import os
import numpy as np
import torch
import random
from scipy.sparse import vstack
from torch.utils.data import Dataset
from torch_geometric.data import Data

label2idx = {
    'rumours': 0, 
    'non-rumours': 1,
}

def pert_edge_index(edge_index, node_num, pertrate):
    length = edge_index.shape[1]
    pert_num = int(length*pertrate)
    if length > 0 and node_num >= 5 and pert_num > 0:
        pert_index = np.random.choice(list(range(length)), size=pert_num, replace=False)
        pert_value = np.random.choice(list(range(node_num)), size=pert_num, replace=True)
        for idx, val in zip(pert_index, pert_value):
            while val==edge_index[0][idx] or val==edge_index[1][idx]:
                val = np.random.randint(0, node_num)
            edge_index[0][idx] = val   

    return edge_index.tolist()

# bert dataset
class TextGraphDataset(Dataset):
    def __init__(self, fold_x, pertrate=0.): 
        self.fold_x = fold_x
        self.pertrate = pertrate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index): 
        path = self.fold_x[index]
        label, src_tid = path.split('/')[-2:]
        
        # ====================================edge_index========================================
        edge = np.loadtxt(os.path.join(path, 'text_structure.txt'), dtype=str).reshape(-1, 2)
        tid2idx = {tid:idx for idx, tid in enumerate(edge[:, 1])}

        edge_index = np.array([
            [tid2idx[tid] for tid in edge[1:, 0]], 
            [tid2idx[tid] for tid in edge[1:, 1]]
        ])    
        edge_index = pert_edge_index(edge_index, len(tid2idx), self.pertrate)
        new_edge_index = edge_index

        burow = list(edge_index[1])
        bucol = list(edge_index[0])
        bunew_edge_index = [burow, bucol]

        edge_index = torch.LongTensor([new_edge_index[0] + bunew_edge_index[0], new_edge_index[1] + bunew_edge_index[1]])
        edge_type = torch.cat([torch.zeros(len(new_edge_index[0])), torch.ones(len(bunew_edge_index[0]))], dim=0)

        # =========================================X====================================================
        feature = np.load(os.path.join(path, 'bert.npy'), allow_pickle=True).item()
        x = np.vstack([feature[tid] for tid in edge[:, 1]])    

        return Data(x=torch.FloatTensor(x), 
                    edge_index=edge_index, 
                    edge_type=edge_type, 
                    y=torch.LongTensor([label2idx[label]]), 
                    index=index) 

# propagation dataset
class PropGraphDataset(Dataset):
    def __init__(self, fold_x, pertrate=0.): 
        self.fold_x = fold_x
        self.pertrate = pertrate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index): 
        path = self.fold_x[index]
        label, src_tid = path.split('/')[-2:]

        # ====================================edge_index========================================
        edge = np.loadtxt(os.path.join(path, 'prop_structure.txt'), dtype=str).reshape(-1, 2)
        tid2idx = {tid:idx for idx, tid in enumerate(edge[:, 1])}

        edge_index = np.array([
            [tid2idx[tid] for tid in edge[1:, 0]], 
            [tid2idx[tid] for tid in edge[1:, 1]]
        ])    
        edge_index = pert_edge_index(edge_index, len(tid2idx), self.pertrate)
        
        new_edge_index = edge_index

        burow = list(edge_index[1])
        bucol = list(edge_index[0])
        bunew_edge_index = [burow, bucol]

        edge_index = torch.LongTensor([new_edge_index[0] + bunew_edge_index[0], new_edge_index[1] + bunew_edge_index[1]])
        edge_type = torch.cat([torch.zeros(len(new_edge_index[0])), torch.ones(len(bunew_edge_index[0]))], dim=0)

        # =========================================X====================================================
        feature = np.load(os.path.join(path, 'prop.npy'), allow_pickle=True).item()
        x = np.vstack([feature[tid] for tid in edge[:, 1]])    

        return Data(x=torch.FloatTensor(x), 
                    edge_index=edge_index, 
                    edge_type=edge_type, 
                    y=torch.LongTensor([label2idx[label]]), 
                    index=index) 

class MixData(Data):
    def __init__(self, x_t=None, edge_index_t=None, edge_type_t=None, x_p=None, edge_index_p=None, edge_type_p=None, y=None, index=None, labeled=None):
        super().__init__()
        self.x_t = x_t
        self.edge_index_t = edge_index_t
        self.edge_type_t = edge_type_t
        
        self.x_p = x_p
        self.edge_index_p = edge_index_p
        self.edge_type_p = edge_type_p

        self.y = y
        self.index = index
        self.labeled = labeled

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_t':
            return self.x_t.size(0)
        elif key == 'edge_index_p':
            return self.x_p.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

class MixGraphDataset(Dataset):
    def __init__(self, fold_x, pertrate=0.): 
        self.fold_x = fold_x
        self.pertrate = pertrate

        self.text_dataset = TextGraphDataset(fold_x, pertrate)
        self.propag_dataset = PropGraphDataset(fold_x, pertrate)

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index): 
        text_G = self.text_dataset[index]
        x_t, edge_index_t, edge_type_t = text_G.x, text_G.edge_index, text_G.edge_type
        
        propag_G = self.propag_dataset[index]
        x_p, edge_index_p, edge_type_p = propag_G.x, propag_G.edge_index, propag_G.edge_type
        
        y = text_G.y

        return MixData(x_t=x_t, 
                       edge_index_t=edge_index_t, 
                       edge_type_t=edge_type_t, 
                       x_p=x_p, 
                       edge_index_p=edge_index_p, 
                       edge_type_p=edge_type_p, 
                       y=y, 
                       index=index) 
