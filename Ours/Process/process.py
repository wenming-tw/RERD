import os
from Process.dataset import MixGraphDataset, MixGraphDatasetGcl, MixGraphDatasetGclSemi, MixGraphDatasetGcl_LN

################################# load data ###################################
def loadMixData(fold_x_train, fold_x_test, min_droprate=0, max_droprate=0, drop_prob=0):
    print("loading train set", )
    traindata_list = MixGraphDataset(fold_x_train, min_droprate, max_droprate, drop_prob)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = MixGraphDataset(fold_x_test)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadMixDataGcl(fold_x_train, fold_x_test, min_droprate=0, max_droprate=0, drop_prob=0, min_time_limit=0., max_time_limit=120.):
    print("loading train set", )
    traindata_list = MixGraphDatasetGcl(fold_x_train, min_droprate, max_droprate, drop_prob, min_time_limit, max_time_limit)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = MixGraphDataset(fold_x_test)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list 

def loadMixDataGclSemi(fold_x_train, fold_x_test, pseudo_label_dict={}, min_droprate=0, max_droprate=0, drop_prob=0, min_time_limit=0., max_time_limit=120.):
    print("loading train set", )
    traindata_list = MixGraphDatasetGclSemi(fold_x_train, pseudo_label_dict, min_droprate, max_droprate, drop_prob, min_time_limit, max_time_limit)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = MixGraphDataset(fold_x_test)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

################################# noise data ###################################
def loadMixDataGcl_LN(fold_x_train, fold_x_test, min_droprate=0, max_droprate=0, drop_prob=0, min_time_limit=0., max_time_limit=120., noise_rate=0.05):
    print("loading train set", )
    traindata_list = MixGraphDatasetGcl_LN(fold_x_train, min_droprate, max_droprate, drop_prob, min_time_limit, max_time_limit, noise_rate)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = MixGraphDataset(fold_x_test)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list 

# edge_remove
def loadMixDataGcl_ER(fold_x_train, fold_x_test, min_droprate=0, max_droprate=0, drop_prob=0, min_time_limit=0., max_time_limit=120., remove_rate=0.):
    from Process.dataset_edge_remove import MixGraphDataset, MixGraphDatasetGcl
    print("loading train set", )
    traindata_list = MixGraphDatasetGcl(fold_x_train, min_droprate, max_droprate, drop_prob, min_time_limit, max_time_limit, remove_rate)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = MixGraphDataset(fold_x_test, remove_rate=remove_rate)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list 

def loadMixDataGcl_NR(fold_x_train, fold_x_test, min_droprate=0, max_droprate=0, drop_prob=0, min_time_limit=0., max_time_limit=120., remove_rate=0.):
    from Process.dataset_node_remove import MixGraphDataset, MixGraphDatasetGcl
    print("loading train set", )
    traindata_list = MixGraphDatasetGcl(fold_x_train, min_droprate, max_droprate, drop_prob, min_time_limit, max_time_limit, remove_rate)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = MixGraphDataset(fold_x_test, remove_rate=remove_rate)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list 