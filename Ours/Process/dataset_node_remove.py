import os
import numpy as np
import torch
import random
from scipy.sparse import vstack
from torch.utils.data import Dataset
from torch_geometric.data import Data
from Process.dataset import PropGraphDataset, TextGraphDatasetTime, PropGraphDatasetTime, MixData, MixDataGcl

label2idx = {
    'rumours': 0, 
    'non-rumours': 1,
}

def remove_text_node(edge, remove_rate):
    node_num = len(edge)
    remove_num = int(node_num*remove_rate)

    if node_num >= 5 and remove_num > 0:
        remove_index = np.random.choice(list(range(node_num-1)), size=remove_num, replace=False) + 1
        remove_node = [edge[:, 1][idx] for idx in remove_index]
        edge = np.array([row for row in edge if ((row[0] not in remove_node) and (row[1] not in remove_node))]).reshape(-1, 2)

    return edge

# bert dataset
class TextGraphDataset(Dataset):
    def __init__(self, fold_x, min_droprate=0., max_droprate=0., drop_prob=0., remove_rate=0.): 
        self.fold_x = fold_x
        self.min_droprate = min_droprate
        self.max_droprate = max_droprate
        self.drop_prob = drop_prob
        self.remove_rate = remove_rate
        self.keep_edge = dict()

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index): 
        path = self.fold_x[index]
        label, src_tid = path.split('/')[-2:]
        
        # ====================================edge_index========================================
        if src_tid in self.keep_edge:
            edge = self.keep_edge[src_tid]
        else:
            edge = remove_text_node(np.loadtxt(os.path.join(path, 'text_structure.txt'), dtype=str).reshape(-1, 2), self.remove_rate)
            self.keep_edge[src_tid] = edge

        node_set = np.unique(edge.ravel()[1:])
        tid2idx = {tid:idx for idx, tid in enumerate(node_set)}

        edge_index = [
            [tid2idx[tid] for tid in edge[1:, 0]], 
            [tid2idx[tid] for tid in edge[1:, 1]]
        ]
        
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
        x = np.vstack([feature[tid] for tid in node_set])    

        return Data(x=torch.FloatTensor(x), 
                    edge_index=edge_index, 
                    edge_type=edge_type, 
                    y=torch.LongTensor([label2idx[label]]), 
                    index=index) 

class MixGraphDataset(Dataset):
    def __init__(self, fold_x, min_droprate=0., max_droprate=0., drop_prob=0., remove_rate=0.): 
        self.fold_x = fold_x
        self.min_droprate = min_droprate
        self.max_droprate = max_droprate
        self.drop_prob = drop_prob
        self.remove_rate = remove_rate

        self.text_dataset = TextGraphDataset(fold_x, min_droprate, max_droprate, drop_prob, remove_rate)
        self.propag_dataset = PropGraphDataset(fold_x, min_droprate, max_droprate, drop_prob)

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
        self.propag_dataset = PropGraphDataset(fold_x, min_droprate, max_droprate, drop_prob)

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