import os
import logging
import argparse
import numpy as np
from datetime import datetime

import torch
from Process.rand5fold import load5foldSemiData
from Trainer.curriculum_labeling import Curriculum_Labeling

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='../logs/Semi/Ours_w_CL/')
    parser.add_argument('--ckpt_dir', default='../ckpt/Semi/Ours_w_CL/')
    parser.add_argument('--label_ratio', default=0.5, type=float)
    parser.add_argument('--percentiles_holder', default=25, type=float)
    parser.add_argument('--bsize', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--patience', default=40, type=int)
    parser.add_argument('--min_droprate', default=0.1, type=float)
    parser.add_argument('--max_droprate', default=0.4, type=float)
    parser.add_argument('--drop_prob', default=1., type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--beta', default=0.2, type=float)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--reset_model', action='store_true', default=True)    
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()
    print(vars(args))
    
    experiment_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") 
    print('experiment_id:', experiment_id)

    os.makedirs(os.path.join(args.log_dir), exist_ok=True) 
    logger = setup_logger('logger', os.path.join(args.log_dir, '{}.log'.format(experiment_id)))
    sub_logger = setup_logger('sub_logger', os.path.join(args.log_dir, '{}_sub.log'.format(experiment_id)))
    logger.info(str(vars(args)))

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    test_accs = []
    FR_Pre, FR_Rec, FR_F1 = [], [], []
    NR_Pre, NR_Rec, NR_F1 = [], [], []
    #for iter in range(args.iterations):
    for iter in [7, 8, 9]:
        fold0_x_test, (fold0_x_train_l, fold0_x_train_u), \
        fold1_x_test, (fold1_x_train_l, fold1_x_train_u), \
        fold2_x_test, (fold2_x_train_l, fold2_x_train_u), \
        fold3_x_test, (fold3_x_train_l, fold3_x_train_u), \
        fold4_x_test, (fold4_x_train_l, fold4_x_train_u) = load5foldSemiData(args.label_ratio, iter)
                
        print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train_l), len(fold0_x_train_u))
        print('fold1 shape: ', len(fold1_x_test), len(fold1_x_train_l), len(fold1_x_train_u))
        print('fold2 shape: ', len(fold2_x_test), len(fold2_x_train_l), len(fold2_x_train_u))
        print('fold3 shape: ', len(fold3_x_test), len(fold3_x_train_l), len(fold3_x_train_u))
        print('fold4 shape: ', len(fold4_x_test), len(fold4_x_train_l), len(fold4_x_train_u))

        trainer = Curriculum_Labeling(args, experiment_id, device)
        accs0, Pre1_0, Pre2_0, Rec1_0, Rec2_0, F1_0, F2_0 = trainer.train_cl(fold0_x_train_l, fold0_x_train_u, fold0_x_test, iter, 0, sub_logger)
        accs1, Pre1_1, Pre2_1, Rec1_1, Rec2_1, F1_1, F2_1 = trainer.train_cl(fold1_x_train_l, fold1_x_train_u, fold1_x_test, iter, 1, sub_logger)
        accs2, Pre1_2, Pre2_2, Rec1_2, Rec2_2, F1_2, F2_2 = trainer.train_cl(fold2_x_train_l, fold2_x_train_u, fold2_x_test, iter, 2, sub_logger)
        accs3, Pre1_3, Pre2_3, Rec1_3, Rec2_3, F1_3, F2_3 = trainer.train_cl(fold3_x_train_l, fold3_x_train_u, fold3_x_test, iter, 3, sub_logger)
        accs4, Pre1_4, Pre2_4, Rec1_4, Rec2_4, F1_4, F2_4 = trainer.train_cl(fold4_x_train_l, fold4_x_train_u, fold4_x_test, iter, 4, sub_logger)
        sub_logger.info('====================================================================================')

        test_accs.append((accs0 + accs1 + accs2 + accs3 + accs4) / 5)

        FR_Pre.append((Pre1_0 + Pre1_1 + Pre1_2 + Pre1_3 + Pre1_4) / 5)
        FR_Rec.append((Rec1_0 + Rec1_1 + Rec1_2 + Rec1_3 + Rec1_4) / 5)
        FR_F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)

        NR_Pre.append((Pre2_0 + Pre2_1 + Pre2_2 + Pre2_3 + Pre2_4) / 5)
        NR_Rec.append((Rec2_0 + Rec2_1 + Rec2_2 + Rec2_3 + Rec2_4) / 5)
        NR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)

        logger.info("Iteration {:03d}, Test_Accuracy: {:.4f}|FR Pre: {:.4f}|NR Pre: {:.4f}|FR Rec: {:.4f}|NR Rec: {:.4f}|FR F1: {:.4f}|NR F1: {:.4f}".format(
            iter, test_accs[-1], FR_Pre[-1], NR_Pre[-1], FR_Rec[-1], NR_Rec[-1], FR_F1[-1], NR_F1[-1]))  
        
        print("Iteration {:03d}, Test_Accuracy: {:.4f}|FR Pre: {:.4f}|NR Pre: {:.4f}|FR Rec: {:.4f}|NR Rec: {:.4f}|FR F1: {:.4f}|NR F1: {:.4f}".format(
            iter, test_accs[-1], FR_Pre[-1], NR_Pre[-1], FR_Rec[-1], NR_Rec[-1], FR_F1[-1], NR_F1[-1]))

    logger.info("Total, Test_Accuracy: {:.4f}|FR Pre: {:.4f}|NR Pre: {:.4f}|FR Rec: {:.4f}|NR Rec: {:.4f}|FR F1: {:.4f}|NR F1: {:.4f}".format(
            np.mean(test_accs), np.mean(FR_Pre), np.mean(NR_Pre), np.mean(FR_Rec), np.mean(NR_Rec), np.mean(FR_F1), np.mean(NR_F1)))    
    print("Total, Test_Accuracy: {:.4f}|FR Pre: {:.4f}|NR Pre: {:.4f}|FR Rec: {:.4f}|NR Rec: {:.4f}|FR F1: {:.4f}|NR F1: {:.4f}".format(
            np.mean(test_accs), np.mean(FR_Pre), np.mean(NR_Pre), np.mean(FR_Rec), np.mean(NR_Rec), np.mean(FR_F1), np.mean(NR_F1))) 


