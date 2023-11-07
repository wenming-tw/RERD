import os
import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from Process.process import loadMixDataGclSemi
from Process.dataset import MixGraphDataset
from tools.evaluate import evaluationclass
from tools.earlystopping import EarlyStopping
from main import GCN_Net
from tqdm import tqdm

# https://github.com/uvavision/Curriculum-Labeling/blob/main/methods/entropy/curriculum_labeling.py
class Curriculum_Labeling:
    def __init__(self, args, experiment_id='', device='cpu'):
        self.args = args
        self.experiment_id = experiment_id
        self.device = device

    def init_model(self, reset_model):
        if reset_model:
            self.model = GCN_Net(in_dim_t=768, in_dim_p=11, hid_dim=64, out_dim=64).to(self.device) 
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            self.criterion = nn.CrossEntropyLoss()
            self.early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        else:
            self.early_stopping.early_stop = False
            self.early_stopping.counter = 0

    def gen_pseudo_label(self, data_unlabeled, iteration):
        dataset = MixGraphDataset(data_unlabeled)
        data_loader = DataLoader(dataset, batch_size=self.args.bsize, shuffle=False, follow_batch=['x_t', 'x_p'], num_workers=4)

        self.model.eval()
        with torch.no_grad():
            outs = [self.model(Batch_data.to(self.device)).cpu() for Batch_data in data_loader]
        outs = torch.softmax(torch.cat(outs, dim=0), dim=1)
        max_val, max_idx = outs.max(dim=1)

        percentiles_holder = 100 - (self.args.percentiles_holder) * iteration
        threshold = np.percentile(max_val.numpy(), percentiles_holder)
        print('Actual Threshold: {} - Percentile: {}'.format(threshold, percentiles_holder))

        pseudo_label_dict = {}
        for data, val, idx in zip(data_unlabeled, max_val, max_idx):
            if val >= threshold:
                pseudo_label_dict[data] = idx

        return pseudo_label_dict

    def evaluate(self, data_loader, epoch, verbose=True):
        self.model.eval() 
        pred, true = [], []
        for batch in tqdm(data_loader):
            with torch.no_grad():
                batch.to(self.device)
                out = self.model(batch)
                loss = self.criterion(out, batch.y)

            pred.append(out.cpu())
            true.append(batch.y.cpu())

        pred = torch.cat(pred)
        true = torch.cat(true)
        loss = self.criterion(pred, true).item()
        val_pred = torch.argmax(pred, dim=1)

        accs, acc1, acc2, pre1, pre2, rec1, rec2, F1, F2 = evaluationclass(val_pred, true)
   
        if verbose:
            res = [ 'Epoch {:03d}'.format(epoch), 
                    'acc:{:.4f}'.format(accs),
                    'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(acc1, pre1, rec1, F1),
                    'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(acc2, pre2, rec2, F2)]
            print('results:', res)     

        return loss, accs, pre1, pre2, rec1, rec2, F1, F2 

    def train_iter(self, data_loader):
        self.model.train() 
        for batch in data_loader:
            if batch.y.shape[0]==1:
                break
                        
            batch.to(self.device)
            out, cl_loss = self.model(batch, CL=True)
            loss = self.criterion(out[batch.labeled], batch.y[batch.labeled])
            if (~batch.labeled).sum() > 0:
                loss = loss*(1-self.args.beta) + self.criterion(out[~batch.labeled], batch.y[~batch.labeled])*self.args.beta
            loss += cl_loss*self.args.alpha
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_model(self, data_train_l, data_train_u, data_test, pseudo_label_dict={}):
        if len(pseudo_label_dict) == 0:
            data_train_u = []
        else:
            data_train_u = [data for data in data_train_u if data in pseudo_label_dict]
            
        data_train = data_train_l+data_train_u
        train_dataset, test_dataset = loadMixDataGclSemi(data_train, data_test, pseudo_label_dict, self.args.min_droprate, self.args.max_droprate, self.args.drop_prob, 0., 120.)
        train_loader = DataLoader(train_dataset, batch_size=self.args.bsize, shuffle=True, num_workers=8, follow_batch=['x_t', 'x_t_1', 'x_p', 'x_p_1'])
        test_loader = DataLoader(test_dataset, batch_size=self.args.bsize, shuffle=True, num_workers=8, follow_batch=['x_t', 'x_t_1', 'x_p', 'x_p_1'])     
        
        for epoch in range(self.args.epochs):
            self.train_iter(train_loader)
            loss, accs, pre1, pre2, rec1, rec2, F1, F2  = self.evaluate(test_loader, epoch)
            self.early_stopping(loss, accs, pre1, pre2, rec1, rec2, F1, F2, self.model, self.experiment_id, self.args.ckpt_dir, self.iter, self.fold)
            
            if self.early_stopping.early_stop:
                print("Early stopping")
                accs = self.early_stopping.accs
                pre1 = self.early_stopping.pre1
                pre2 = self.early_stopping.pre2
                rec1 = self.early_stopping.rec1
                rec2 = self.early_stopping.rec2        
                F1 = self.early_stopping.F1
                F2 = self.early_stopping.F2
                self.model.load_state_dict(torch.load(os.path.join(self.args.ckpt_dir, self.experiment_id, 'iter_{}_fold_{}.m'.format(self.iter, self.fold))))
                break

        return accs, pre1, pre2, rec1, rec2, F1, F2
    
    def train_cl(self, data_train_l, data_train_u, data_test, iter=0, fold=0, logger=None):
        self.iter = iter
        self.fold = fold
        self.init_model(True)
        accs, pre1, pre2, rec1, rec2, F1, F2 = self.train_model(data_train_l, data_train_u, data_test, {})
        if logger is not None:
            logger.info("CL {:02d}, Test_Accuracy: {:.4f}|FR Pre: {:.4f}|NR Pre: {:.4f}|FR Rec: {:.4f}|NR Rec: {:.4f}|FR F1: {:.4f}|NR F1: {:.4f}".format(
                0, accs, pre1, pre2, rec1, rec2, F1, F2))           
        best_accs, best_pre1, best_pre2, best_rec1, best_rec2, best_F1, best_F2 = accs, pre1, pre2, rec1, rec2, F1, F2

        iteration = 1
        while True:
            pseudo_label_dict = self.gen_pseudo_label(data_train_u, iteration)
            print('pseudo_label_dict length:', len(pseudo_label_dict))

            self.init_model(self.args.reset_model)
            accs, pre1, pre2, rec1, rec2, F1, F2 = self.train_model(data_train_l, data_train_u, data_test, pseudo_label_dict)
            if logger is not None:
                logger.info("CL {:02d}, Test_Accuracy: {:.4f}|FR Pre: {:.4f}|NR Pre: {:.4f}|FR Rec: {:.4f}|NR Rec: {:.4f}|FR F1: {:.4f}|NR F1: {:.4f}".format(
                    iteration, accs, pre1, pre2, rec1, rec2, F1, F2))  

            if sum([accs, pre1, pre2, rec1, rec2, F1, F2]) > sum([best_accs, best_pre1, best_pre2, best_rec1, best_rec2, best_F1, best_F2]):
                best_accs, best_pre1, best_pre2, best_rec1, best_rec2, best_F1, best_F2 = accs, pre1, pre2, rec1, rec2, F1, F2

            if self.args.percentiles_holder * iteration >= 100:
                print('All dataset used. Process finished.')
                break
            iteration += 1

        return best_accs, best_pre1, best_pre2, best_rec1, best_rec2, best_F1, best_F2