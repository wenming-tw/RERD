import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_mean
from torch_geometric.loader import DataLoader

from models.nn import RGCNConv
from Process.rand5fold import load5foldData
from Process.process import loadMixData
from tools.evaluate import evaluationclass
from tools.earlystopping import EarlyStopping

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class RGCN(nn.Module):
    def __init__(self, in_dim_t, in_dim_p, hid_dim, out_dim):
        super(RGCN, self).__init__()
        self.text_conv1 = RGCNConv(in_dim_t, hid_dim, num_relations=2)
        self.text_conv2 = RGCNConv(hid_dim, out_dim, num_relations=2)
        self.text_bn1 = nn.BatchNorm1d(hid_dim)
        self.text_bn2 = nn.BatchNorm1d(out_dim)
        
        self.prop_conv1 = RGCNConv(in_dim_p, hid_dim, num_relations=2)
        self.prop_conv2 = RGCNConv(hid_dim, out_dim, num_relations=2)        
        self.prop_bn1 = nn.BatchNorm1d(hid_dim)
        self.prop_bn2 = nn.BatchNorm1d(out_dim)

        self.relu = nn.LeakyReLU(negative_slope=0.01)
        
    def forward(self, data):
        ## text graph
        x_t, edge_index_t, edge_type_t = data.x_t, data.edge_index_t, data.edge_type_t
        x_t = self.text_conv1(x_t, edge_index_t, edge_type_t)
        x_t = self.relu(self.text_bn1(x_t))

        x_t = self.text_conv2(x_t, edge_index_t, edge_type_t)
        x_t = self.relu(self.text_bn2(x_t))

        x_t = scatter_mean(x_t, data.x_t_batch, dim=0)

        ## propagation graph
        x_p, edge_index_p, edge_type_p = data.x_p, data.edge_index_p, data.edge_type_p
        x_p = self.prop_conv1(x_p, edge_index_p, edge_type_p)
        x_p = self.relu(self.prop_bn1(x_p))

        x_p = self.prop_conv2(x_p, edge_index_p, edge_type_p)
        x_p = self.relu(self.prop_bn2(x_p))

        x_p = scatter_mean(x_p, data.x_p_batch, dim=0)

        return x_t, x_p

class GCN_Net(nn.Module):
    def __init__(self, in_dim_t, in_dim_p, hid_dim, out_dim):
        super(GCN_Net, self).__init__()
        self.encoder = RGCN(in_dim_t, in_dim_p, hid_dim, out_dim)
        self.fusion = nn.Linear(out_dim*2, out_dim)
        self.fc = nn.Linear(out_dim, 2)
        self.relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, data):
        x_t, x_p = self.encoder(data)
        
        x = torch.cat([x_t, x_p], dim=1)
        x = torch.relu(self.fusion(x))

        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x

def train_GCN(x_test, x_train, args, experiment_id, iter, fold):
    model = GCN_Net(in_dim_t=768, in_dim_p=11, hid_dim=64, out_dim=64).to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    traindata_list, testdata_list = loadMixData(x_train, x_test, args.min_droprate, args.max_droprate, args.drop_prob)
    train_loader = DataLoader(traindata_list, batch_size=args.bsize, shuffle=True, num_workers=8, follow_batch=['x_t', 'x_p'])
    test_loader = DataLoader(testdata_list, batch_size=args.bsize, shuffle=True, num_workers=8, follow_batch=['x_t', 'x_p']) 

    for epoch in range(args.epochs): 
        model.train() 
        for Batch_data in tqdm(train_loader):
            Batch_data.to(device)
            out_labels = model(Batch_data) 
            finalloss = F.nll_loss(out_labels, Batch_data.y)
            loss = finalloss
  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        pred, true = [], []
        for Batch_data in tqdm(test_loader):
            with torch.no_grad():
                Batch_data.to(device)
                val_out = model(Batch_data)

            pred.append(val_out.cpu())
            true.append(Batch_data.y.cpu())

        pred = torch.cat(pred)
        true = torch.cat(true)
        val_loss = F.nll_loss(pred, true)
        _, val_pred = pred.max(dim=1)
        correct = val_pred.eq(true).sum().item()
        val_acc = correct / len(true) 

        Acc_all, Acc1, Acc2, Prec1, Prec2, Recll1, Recll2, F1, F2 = evaluationclass(val_pred, true)

        res = [ 'Epoch {:03d}'.format(epoch), 
                'acc:{:.4f}'.format(Acc_all),
                'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc1, Prec1, Recll1, F1),
                'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc2, Prec2, Recll2, F2)]
        print('results:', res)
        early_stopping(val_loss, val_acc, Prec1, Prec2, Recll1, Recll2, F1, F2, model, experiment_id, args.ckpt_dir, iter, fold)
                     
        accs = val_acc
        pre1 = Prec1
        pre2 = Prec2
        rec1 = Recll1
        rec2 = Recll2
        F1 = F1
        F2 = F2

        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.accs
            pre1 = early_stopping.pre1
            pre2 = early_stopping.pre2
            rec1 = early_stopping.rec1
            rec2 = early_stopping.rec2        
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            break
    return accs, pre1, pre2, rec1, rec2, F1, F2

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='../logs/Ours_wo_gcl/')
    parser.add_argument('--ckpt_dir', default='../ckpt/Ours_wo_gcl/')
    parser.add_argument('--bsize', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--patience', default=40, type=int)
    parser.add_argument('--min_droprate', default=0.1, type=float)
    parser.add_argument('--max_droprate', default=0.4, type=float)
    parser.add_argument('--drop_prob', default=0.8, type=float)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()
    print(vars(args))
    
    experiment_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") 
    print('experiment_id:', experiment_id)

    os.makedirs(os.path.join(args.log_dir), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.log_dir, '{}.log'.format(experiment_id)),
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )    

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    logger = logging.getLogger(__name__)
    logger.info(str(vars(args)))

    test_accs = []
    FR_Pre, FR_Rec, FR_F1 = [], [], []
    NR_Pre, NR_Rec, NR_F1 = [], [], []
    for iter in range(args.iterations):
        fold0_x_test, fold0_x_train, \
        fold1_x_test, fold1_x_train, \
        fold2_x_test, fold2_x_train, \
        fold3_x_test, fold3_x_train, \
        fold4_x_test, fold4_x_train = load5foldData(iter)
        
        accs0, Pre1_0, Pre2_0, Rec1_0, Rec2_0, F1_0, F2_0 = train_GCN(fold0_x_test, fold0_x_train, args, experiment_id, iter, 0)
        accs1, Pre1_1, Pre2_1, Rec1_1, Rec2_1, F1_1, F2_1 = train_GCN(fold1_x_test, fold1_x_train, args, experiment_id, iter, 1)
        accs2, Pre1_2, Pre2_2, Rec1_2, Rec2_2, F1_2, F2_2 = train_GCN(fold2_x_test, fold2_x_train, args, experiment_id, iter, 2)
        accs3, Pre1_3, Pre2_3, Rec1_3, Rec2_3, F1_3, F2_3 = train_GCN(fold3_x_test, fold3_x_train, args, experiment_id, iter, 3)
        accs4, Pre1_4, Pre2_4, Rec1_4, Rec2_4, F1_4, F2_4 = train_GCN(fold4_x_test, fold4_x_train, args, experiment_id, iter, 4)

        test_accs.append((accs0+accs1+accs2+accs3+accs4)/5)

        FR_Pre.append((Pre1_0+Pre1_1+Pre1_2+Pre1_3+Pre1_4)/5)
        FR_Rec.append((Rec1_0+Rec1_1+Rec1_2+Rec1_3+Rec1_4)/5)
        FR_F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)

        NR_Pre.append((Pre2_0+Pre2_1+Pre2_2+Pre2_3+Pre2_4)/5)
        NR_Rec.append((Rec2_0+Rec2_1+Rec2_2+Rec2_3+Rec2_4)/5)
        NR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)

        logger.info("Iteration {:03d}, Test_Accuracy: {:.4f}|FR Pre: {:.4f}|NR Pre: {:.4f}|FR Rec: {:.4f}|NR Rec: {:.4f}|FR F1: {:.4f}|NR F1: {:.4f}".format(
            iter, test_accs[-1], FR_Pre[-1], NR_Pre[-1], FR_Rec[-1], NR_Rec[-1], FR_F1[-1], NR_F1[-1]))  
        
        print("Iteration {:03d}, Test_Accuracy: {:.4f}|FR Pre: {:.4f}|NR Pre: {:.4f}|FR Rec: {:.4f}|NR Rec: {:.4f}|FR F1: {:.4f}|NR F1: {:.4f}".format(
            iter, test_accs[-1], FR_Pre[-1], NR_Pre[-1], FR_Rec[-1], NR_Rec[-1], FR_F1[-1], NR_F1[-1]))

    logger.info("Total, Test_Accuracy: {:.4f}|FR Pre: {:.4f}|NR Pre: {:.4f}|FR Rec: {:.4f}|NR Rec: {:.4f}|FR F1: {:.4f}|NR F1: {:.4f}".format(
            np.mean(test_accs), np.mean(FR_Pre), np.mean(NR_Pre), np.mean(FR_Rec), np.mean(NR_Rec), np.mean(FR_F1), np.mean(NR_F1)))    
    print("Total, Test_Accuracy: {:.4f}|FR Pre: {:.4f}|NR Pre: {:.4f}|FR Rec: {:.4f}|NR Rec: {:.4f}|FR F1: {:.4f}|NR F1: {:.4f}".format(
            np.mean(test_accs), np.mean(FR_Pre), np.mean(NR_Pre), np.mean(FR_Rec), np.mean(NR_Rec), np.mean(FR_F1), np.mean(NR_F1))) 


