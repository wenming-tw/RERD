import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_mean
from torch_geometric.loader import DataLoader

from models.nn import RGCNConv
from Process.rand5fold import load5foldData
from Process.dataset import MixGraphDatasetCount
from tools.evaluate import evaluationclass

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

def evaluate(x, count):
    dataset = MixGraphDatasetCount(x, 0, 0, 0, count=count)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8, follow_batch=['x_t', 'x_p'])     

    model.eval()
    pred, true = [], []
    for Batch_data in data_loader:
        with torch.no_grad():
            Batch_data.to(device)
            val_out = model(Batch_data)

        pred.append(val_out.cpu())
        true.append(Batch_data.y.cpu())

    pred = torch.cat(pred)
    true = torch.cat(true)
    _, val_pred = pred.max(dim=1)
    correct = val_pred.eq(true).sum().item()
    val_acc = correct / len(true) 

    Acc_all, Acc1, Acc2, Prec1, Prec2, Recll1, Recll2, F1, F2 = evaluationclass(val_pred, true)

    return val_acc, Prec1, Prec2, Recll1, Recll2, F1, F2

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', default='')
    parser.add_argument('--experiment_id', default='')
    parser.add_argument('--count', default=1, type=int)    
    args = parser.parse_args()
    print(vars(args))

    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    model = GCN_Net(in_dim_t=768, in_dim_p=11, hid_dim=64, out_dim=64).to(device) 

    test_accs = []
    FR_Pre, FR_Rec, FR_F1 = [], [], []
    NR_Pre, NR_Rec, NR_F1 = [], [], []
    for iter in range(10):
        fold0_x_test, fold0_x_train, \
        fold1_x_test, fold1_x_train, \
        fold2_x_test, fold2_x_train, \
        fold3_x_test, fold3_x_train, \
        fold4_x_test, fold4_x_train = load5foldData(iter)
        
        model.load_state_dict(torch.load(os.path.join('../ckpt', args.modelname, args.experiment_id, 'iter_{}_fold_{}.m'.format(iter, 0))), strict=False)
        result_0 = evaluate(fold0_x_test, args.count)

        model.load_state_dict(torch.load(os.path.join('../ckpt', args.modelname, args.experiment_id, 'iter_{}_fold_{}.m'.format(iter, 1))), strict=False)
        result_1 = evaluate(fold1_x_test, args.count)

        model.load_state_dict(torch.load(os.path.join('../ckpt', args.modelname, args.experiment_id, 'iter_{}_fold_{}.m'.format(iter, 2))), strict=False)
        result_2 = evaluate(fold2_x_test, args.count)

        model.load_state_dict(torch.load(os.path.join('../ckpt', args.modelname, args.experiment_id, 'iter_{}_fold_{}.m'.format(iter, 3))), strict=False)
        result_3 = evaluate(fold3_x_test, args.count)

        model.load_state_dict(torch.load(os.path.join('../ckpt', args.modelname, args.experiment_id, 'iter_{}_fold_{}.m'.format(iter, 4))), strict=False)
        result_4 = evaluate(fold4_x_test, args.count)

        result = np.stack([result_0, result_1, result_2, result_3, result_4], 0).mean(0)
        print(iter, result.round(4))

        test_accs.append(result[0])
        FR_Pre.append(result[1])
        NR_Pre.append(result[2])
        FR_Rec.append(result[3])
        NR_Rec.append(result[4])
        FR_F1.append(result[5])
        NR_F1.append(result[6])

    out = np.mean(test_accs).round(4), np.mean(FR_Pre).round(4), np.mean(NR_Pre).round(4), np.mean(FR_Rec).round(4), np.mean(NR_Rec).round(4), np.mean(FR_F1).round(4), np.mean(NR_F1).round(4)
    print(out)

    df = pd.DataFrame([{
        'model': args.modelname, 
        'experiment_id': args.experiment_id, 
        'count': args.count, 
        'ACC': out[0], 
        'FR_Pre': out[1], 
        'NR_Pre': out[2], 
        'FR_Rec': out[3], 
        'NR_Rec': out[3], 
        'FR_F1': out[5], 
        'NR_F1': out[6], 
    }])

    if os.path.isfile('../logs/early_detection_cnt.csv'):
        df.to_csv('../logs/early_detection_cnt.csv', index=False, mode='a', header=False, encoding='utf-8')
    else:
        df.to_csv('../logs/early_detection_cnt.csv', index=False, mode='w', header=True, encoding='utf-8')


