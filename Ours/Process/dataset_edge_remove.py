import os
import numpy as np
import torch
import random
from scipy.sparse import vstack
from torch.utils.data import Dataset
from torch_geometric.data import Data
from Process.dataset import TextGraphDatasetTime, PropGraphDatasetTime, MixDataGcl

label2idx = {
    'rumours': 0, 
    'non-rumours': 1,
}

def remove_edge_index(edge_index, node_num, remove_rate):
    length = edge_index.shape[1]
    remove_num = int(length*remove_rate)

    if length > 0 and node_num >= 5 and remove_num > 0:
        remove_index = np.random.choice(list(range(length)), size=remove_num, replace=False)
        edge_index = [[edge_index[0][idx], edge_index[1][idx]] for idx in range(length) if idx not in remove_index]
    else:
        edge_index = edge_index.tolist()

    return edge_index

# bert dataset
class TextGraphDataset(Dataset):
    def __init__(self, fold_x, min_droprate=0., max_droprate=0., drop_prob=0., remove_rate=0.): 
        self.fold_x = fold_x
        self.min_droprate = min_droprate
        self.max_droprate = max_droprate
        self.drop_prob = drop_prob
        self.remove_rate = remove_rate
        self.keep_edge_index = dict()

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index): 
        path = self.fold_x[index]
        label, src_tid = path.split('/')[-2:]
        
        # ====================================edge_index========================================
        edge = np.loadtxt(os.path.join(path, 'text_structure.txt'), dtype=str).reshape(-1, 2)
        tid2idx = {tid:idx for idx, tid in enumerate(edge[:, 1])}

        if src_tid not in self.keep_edge_index:
            edge_index = np.array([
                [tid2idx[tid] for tid in edge[1:, 0]], 
                [tid2idx[tid] for tid in edge[1:, 1]]
            ])    
            edge_index = remove_edge_index(edge_index, len(tid2idx), self.remove_rate)
            self.keep_edge_index[src_tid] = edge_index
        else:
            edge_index = self.keep_edge_index[src_tid]
        
        # ========================================= drop edge ===============================================
        droprate = np.random.uniform(self.min_droprate, self.max_droprate)
        if self.drop_prob > 0 and droprate > 0 and np.random.uniform(0, 1) < self.drop_prob:
            row = list(edge_index[0])
            col = list(edge_index[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edge_index = [row, col]
        else:
            new_edge_index = edge_index

        burow = list(edge_index[1])
        bucol = list(edge_index[0])
        droprate = np.random.uniform(self.min_droprate, self.max_droprate)
        if self.drop_prob > 0 and droprate > 0 and np.random.uniform(0, 1) < self.drop_prob:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - droprate)))
            poslist = sorted(poslist)
            burow = list(np.array(burow)[poslist])
            bucol = list(np.array(bucol)[poslist])
            bunew_edge_index = [burow, bucol]
        else:
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
    def __init__(self, fold_x, min_droprate=0., max_droprate=0., drop_prob=0., remove_rate=0.): 
        self.fold_x = fold_x
        self.min_droprate = min_droprate
        self.max_droprate = max_droprate
        self.drop_prob = drop_prob
        self.remove_rate = remove_rate
        self.keep_edge_index = dict()

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index): 
        path = self.fold_x[index]
        label, src_tid = path.split('/')[-2:]

        # ====================================edge_index========================================
        edge = np.loadtxt(os.path.join(path, 'prop_structure.txt'), dtype=str).reshape(-1, 2)
        tid2idx = {tid:idx for idx, tid in enumerate(edge[:, 1])}

        if src_tid not in self.keep_edge_index:
            edge_index = np.array([
                [tid2idx[tid] for tid in edge[1:, 0]], 
                [tid2idx[tid] for tid in edge[1:, 1]]
            ])    
            edge_index = remove_edge_index(edge_index, len(tid2idx), self.remove_rate)
            self.keep_edge_index[src_tid] = edge_index
        else:
            edge_index = self.keep_edge_index[src_tid]

        # ========================================= drop edge ===============================================
        droprate = np.random.uniform(self.min_droprate, self.max_droprate)
        if self.drop_prob > 0 and droprate > 0 and np.random.uniform(0, 1) < self.drop_prob:
            row = list(edge_index[0])
            col = list(edge_index[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edge_index = [row, col]
        else:
            new_edge_index = edge_index

        burow = list(edge_index[1])
        bucol = list(edge_index[0])
        droprate = np.random.uniform(self.min_droprate, self.max_droprate)
        if self.drop_prob > 0 and droprate > 0 and np.random.uniform(0, 1) < self.drop_prob:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - droprate)))
            poslist = sorted(poslist)
            burow = list(np.array(burow)[poslist])
            bucol = list(np.array(bucol)[poslist])
            bunew_edge_index = [burow, bucol]
        else:
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
    def __init__(self, fold_x, min_droprate=0., max_droprate=0., drop_prob=0., remove_rate=0.): 
        self.fold_x = fold_x
        self.min_droprate = min_droprate
        self.max_droprate = max_droprate
        self.drop_prob = drop_prob
        self.remove_rate = remove_rate

        self.text_dataset = TextGraphDataset(fold_x, min_droprate, max_droprate, drop_prob, remove_rate)
        self.propag_dataset = PropGraphDataset(fold_x, min_droprate, max_droprate, drop_prob, remove_rate)

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

class MixGraphDatasetGcl(Dataset):
    def __init__(self, fold_x, min_droprate=0., max_droprate=0., drop_prob=0., min_time_limit=1., max_time_limit=120., remove_rate=0.): 
        self.fold_x = fold_x
        self.min_droprate = min_droprate
        self.max_droprate = max_droprate
        self.drop_prob = drop_prob
       
        self.min_time_limit = min_time_limit
        self.max_time_limit = max_time_limit        
        self.remove_rate = remove_rate

        self.text_dataset = TextGraphDataset(fold_x, min_droprate, max_droprate, drop_prob, remove_rate)
        self.propag_dataset = PropGraphDataset(fold_x, min_droprate, max_droprate, drop_prob, remove_rate)

        self.text_dataset_1 = TextGraphDatasetTime(fold_x, min_droprate, max_droprate, drop_prob, self.max_time_limit)
        self.propag_dataset_1 = PropGraphDatasetTime(fold_x, min_droprate, max_droprate, drop_prob, self.max_time_limit)

    def reset_time_limit(self):
        time_limit = np.random.uniform(self.min_time_limit, self.max_time_limit)
        self.text_dataset_1.time_limit = time_limit
        self.propag_dataset_1.time_limit = time_limit

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index): 
        self.reset_time_limit()        
        text_G = self.text_dataset[index]
        x_t, edge_index_t, edge_type_t = text_G.x, text_G.edge_index, text_G.edge_type
        
        text_G_1 = self.text_dataset_1[index]
        x_t_1, edge_index_t_1, edge_type_t_1 = text_G_1.x, text_G_1.edge_index, text_G_1.edge_type        
        
        propag_G = self.propag_dataset[index]
        x_p, edge_index_p, edge_type_p = propag_G.x, propag_G.edge_index, propag_G.edge_type

        propag_G_1 = self.propag_dataset_1[index]
        x_p_1, edge_index_p_1, edge_type_p_1 = propag_G_1.x, propag_G_1.edge_index, propag_G_1.edge_type        
        
        y = text_G.y

        return MixDataGcl(x_t=x_t, 
                          edge_index_t=edge_index_t, 
                          edge_type_t=edge_type_t, 
                          x_t_1=x_t_1,
                          edge_index_t_1=edge_index_t_1, 
                          edge_type_t_1=edge_type_t_1,                        
                          x_p=x_p, 
                          edge_index_p=edge_index_p, 
                          edge_type_p=edge_type_p, 
                          x_p_1=x_p_1,
                          edge_index_p_1=edge_index_p_1, 
                          edge_type_p_1=edge_type_p_1,                        
                          y=y, 
                          index=index)