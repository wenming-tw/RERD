import os
import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accs = 0
        self.pre1 = 0
        self.pre2 = 0
        self.rec1 = 0
        self.rec2 = 0
        self.F1 = 0
        self.F2 = 0

    def __call__(self, val_loss, accs, pre1, pre2, rec1, rec2, F1, F2, model, experiment_id, ckpt_dir, iter=0, fold=0):
        score = (accs+pre1+pre2+rec1+rec2+F1+F2)/7
        #score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.accs = accs
            self.pre1 = pre1
            self.pre2 = pre2
            self.rec1 = rec1
            self.rec2 = rec2
            self.F1 = F1
            self.F2 = F2
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("BEST Accuracy: {:.4f}|pre1: {:.4f}|pre2: {:.4f}|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(
                    self.accs, self.pre1, self.pre2, self.rec1, self.rec2, self.F1, self.F2
                ))
        else:
            self.best_score = score
            self.accs = accs
            self.pre1 = pre1
            self.pre2 = pre2
            self.rec1 = rec1
            self.rec2 = rec2
            self.F1 = F1
            self.F2 = F2
            self.save_checkpoint(model, experiment_id, ckpt_dir, iter, fold)
            self.counter = 0

    def save_checkpoint(self, model, experiment_id, ckpt_dir, iter=0, fold=0):
        os.makedirs(os.path.join(ckpt_dir, experiment_id), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, experiment_id, 'iter_{}_fold_{}.m'.format(iter, fold)))
        print('Save.')
