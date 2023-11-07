import os
import argparse
import numpy as np
import pandas as pd

import torch
from torch_geometric.loader import DataLoader

from Process.rand5fold import load5foldData
from Process.dataset_pert import MixGraphDataset
from tools.evaluate import evaluationclass
from main import GCN_Net

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def evaluate(x, pertrate):
    dataset = MixGraphDataset(x, pertrate=pertrate)
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
    parser.add_argument('--pertrate', default=0., type=float)
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
        result_0 = evaluate(fold0_x_test, args.pertrate)

        model.load_state_dict(torch.load(os.path.join('../ckpt', args.modelname, args.experiment_id, 'iter_{}_fold_{}.m'.format(iter, 1))), strict=False)
        result_1 = evaluate(fold1_x_test, args.pertrate)

        model.load_state_dict(torch.load(os.path.join('../ckpt', args.modelname, args.experiment_id, 'iter_{}_fold_{}.m'.format(iter, 2))), strict=False)
        result_2 = evaluate(fold2_x_test, args.pertrate)

        model.load_state_dict(torch.load(os.path.join('../ckpt', args.modelname, args.experiment_id, 'iter_{}_fold_{}.m'.format(iter, 3))), strict=False)
        result_3 = evaluate(fold3_x_test, args.pertrate)

        model.load_state_dict(torch.load(os.path.join('../ckpt', args.modelname, args.experiment_id, 'iter_{}_fold_{}.m'.format(iter, 4))), strict=False)
        result_4 = evaluate(fold4_x_test, args.pertrate)

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
        'pertrate': args.pertrate, 
        'ACC': out[0], 
        'FR_Pre': out[1], 
        'NR_Pre': out[2], 
        'FR_Rec': out[3], 
        'NR_Rec': out[3], 
        'FR_F1': out[5], 
        'NR_F1': out[6], 
    }])

    if os.path.isfile('../logs/edge_noise.csv'):
        df.to_csv('../logs/edge_noise.csv', index=False, mode='a', header=False, encoding='utf-8')
    else:
        df.to_csv('../logs/edge_noise.csv', index=False, mode='w', header=True, encoding='utf-8')


