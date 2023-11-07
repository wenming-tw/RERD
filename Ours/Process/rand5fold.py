import random
from random import shuffle
from glob import glob

label2idx = {
    'rumours': 0, 
    'non-rumours': 1,
}

def load5foldData(seed=-1):
    FR, NR = [], []
    l1, l2 = 0, 0
    labelDic = {}
    for path in sorted(glob('../all-rnr-annotated-threads/*/*/*')):
        eid, label = path.split('/')[-1], path.split('/')[-2]

        labelDic[eid] = label2idx[label]
        if labelDic[eid]==0:
            FR.append(path)
            l1 += 1
        elif labelDic[eid]==1:
            NR.append(path)
            l2 += 1
        
    print(len(labelDic))
    print(l1, l2)

    if seed < 0:
        random.shuffle(FR)
        random.shuffle(NR)
    else:
        random.Random(seed).shuffle(FR) 
        random.Random(seed).shuffle(NR)    

    fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
    fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
    leng1 = int(l1 * 0.2)
    leng2 = int(l2 * 0.2)

    fold0_x_train.extend(FR[leng1:])
    fold0_x_train.extend(NR[leng2:])
    fold0_x_test.extend(FR[0:leng1])
    fold0_x_test.extend(NR[0:leng2])


    fold1_x_train.extend(FR[0:leng1])
    fold1_x_train.extend(FR[leng1 * 2:])
    fold1_x_train.extend(NR[0:leng2])
    fold1_x_train.extend(NR[leng2 * 2:])
    fold1_x_test.extend(FR[leng1:leng1 * 2])
    fold1_x_test.extend(NR[leng2:leng2 * 2])
    
    fold2_x_train.extend(FR[0:leng1 * 2])
    fold2_x_train.extend(FR[leng1 * 3:])
    fold2_x_train.extend(NR[0:leng2 * 2])
    fold2_x_train.extend(NR[leng2 * 3:])
    fold2_x_test.extend(FR[leng1 * 2:leng1 * 3])
    fold2_x_test.extend(NR[leng2 * 2:leng2 * 3])
    
    fold3_x_train.extend(FR[0:leng1 * 3])
    fold3_x_train.extend(FR[leng1 * 4:])
    fold3_x_train.extend(NR[0:leng2 * 3])
    fold3_x_train.extend(NR[leng2 * 4:])
    fold3_x_test.extend(FR[leng1 * 3:leng1 * 4])
    fold3_x_test.extend(NR[leng2 * 3:leng2 * 4])
    
    fold4_x_train.extend(FR[0:leng1 * 4])
    fold4_x_train.extend(FR[leng1 * 5:])
    fold4_x_train.extend(NR[0:leng2 * 4])
    fold4_x_train.extend(NR[leng2 * 5:])
    fold4_x_test.extend(FR[leng1 * 4:leng1 * 5])
    fold4_x_test.extend(NR[leng2 * 4:leng2 * 5])

    fold0_test = list(fold0_x_test)
    shuffle(fold0_test)
    fold0_train = list(fold0_x_train)
    shuffle(fold0_train)
    fold1_test = list(fold1_x_test)
    shuffle(fold1_test)
    fold1_train = list(fold1_x_train)
    shuffle(fold1_train)
    fold2_test = list(fold2_x_test)
    shuffle(fold2_test)
    fold2_train = list(fold2_x_train)
    shuffle(fold2_train)
    fold3_test = list(fold3_x_test)
    shuffle(fold3_test)
    fold3_train = list(fold3_x_train)
    shuffle(fold3_train)
    fold4_test = list(fold4_x_test)
    shuffle(fold4_test)
    fold4_train = list(fold4_x_train)
    shuffle(fold4_train)

    return list(fold0_test), list(fold0_train),\
           list(fold1_test), list(fold1_train),\
           list(fold2_test), list(fold2_train),\
           list(fold3_test), list(fold3_train),\
           list(fold4_test), list(fold4_train)


def load5foldSemiData(label_ratio=1.0, seed=-1):
    FR, NR = [], []
    l1, l2 = 0, 0
    labelDic = {}
    for path in sorted(glob('../all-rnr-annotated-threads/*/*/*')):
        eid, label = path.split('/')[-1], path.split('/')[-2]

        labelDic[eid] = label2idx[label]
        if labelDic[eid]==0:
            FR.append(path)
            l1 += 1
        elif labelDic[eid]==1:
            NR.append(path)
            l2 += 1
        
    print(len(labelDic))
    print(l1, l2)

    if seed < 0:
        random.shuffle(FR)
        random.shuffle(NR)
    else:
        random.Random(seed).shuffle(FR) 
        random.Random(seed).shuffle(NR)    

    fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
    fold0_x_train_l, fold1_x_train_l, fold2_x_train_l, fold3_x_train_l, fold4_x_train_l = [], [], [], [], []
    fold0_x_train_u, fold1_x_train_u, fold2_x_train_u, fold3_x_train_u, fold4_x_train_u = [], [], [], [], []
    
    leng1, leng1_l = int(l1 * 0.2), int((l1 - int(l1 * 0.2))*label_ratio)
    leng2, leng2_l = int(l2 * 0.2), int((l2 - int(l2 * 0.2))*label_ratio)

    ## ============= fold_0 =================
    temp_FR, temp_NR = FR[leng1:], NR[leng2:]
    fold0_x_train_l.extend(temp_FR[:leng1_l]) 
    fold0_x_train_l.extend(temp_NR[:leng2_l])
    fold0_x_train_u.extend(temp_FR[leng1_l:]) 
    fold0_x_train_u.extend(temp_NR[leng2_l:])

    fold0_x_test.extend(FR[0:leng1])
    fold0_x_test.extend(NR[0:leng2])

    ## ============= fold_1 =================
    temp_FR, temp_NR = FR[0:leng1]+FR[leng1 * 2:], NR[0:leng2]+NR[leng2 * 2:]
    fold1_x_train_l.extend(temp_FR[:leng1_l]) 
    fold1_x_train_l.extend(temp_NR[:leng2_l])
    fold1_x_train_u.extend(temp_FR[leng1_l:]) 
    fold1_x_train_u.extend(temp_NR[leng2_l:])

    fold1_x_test.extend(FR[leng1:leng1 * 2])
    fold1_x_test.extend(NR[leng2:leng2 * 2])

    ## ============= fold_2 =================
    temp_FR, temp_NR = FR[0:leng1 * 2]+FR[leng1 * 3:], NR[0:leng2 * 2]+NR[leng2 * 3:]
    fold2_x_train_l.extend(temp_FR[:leng1_l]) 
    fold2_x_train_l.extend(temp_NR[:leng2_l])
    fold2_x_train_u.extend(temp_FR[leng1_l:]) 
    fold2_x_train_u.extend(temp_NR[leng2_l:])

    fold2_x_test.extend(FR[leng1 * 2:leng1 * 3])
    fold2_x_test.extend(NR[leng2 * 2:leng2 * 3])
    
    ## ============= fold_3 =================
    temp_FR, temp_NR = FR[0:leng1 * 3]+FR[leng1 * 4:], NR[0:leng2 * 3]+NR[leng2 * 4:]
    fold3_x_train_l.extend(temp_FR[:leng1_l]) 
    fold3_x_train_l.extend(temp_NR[:leng2_l])
    fold3_x_train_u.extend(temp_FR[leng1_l:]) 
    fold3_x_train_u.extend(temp_NR[leng2_l:])

    fold3_x_test.extend(FR[leng1 * 3:leng1 * 4])
    fold3_x_test.extend(NR[leng2 * 3:leng2 * 4])
    
    ## ============= fold_4 =================
    temp_FR, temp_NR = FR[0:leng1 * 4]+FR[leng1 * 5:], NR[0:leng2 * 4]+NR[leng2 * 5:]
    fold4_x_train_l.extend(temp_FR[:leng1_l]) 
    fold4_x_train_l.extend(temp_NR[:leng2_l])
    fold4_x_train_u.extend(temp_FR[leng1_l:]) 
    fold4_x_train_u.extend(temp_NR[leng2_l:])

    fold4_x_test.extend(FR[leng1 * 4:leng1 * 5])
    fold4_x_test.extend(NR[leng2 * 4:leng2 * 5])

    fold0_test = list(fold0_x_test)
    shuffle(fold0_test)
    fold0_train_l = list(fold0_x_train_l)
    shuffle(fold0_train_l)
    fold0_train_u = list(fold0_x_train_u)
    shuffle(fold0_train_u)

    fold1_test = list(fold1_x_test)
    shuffle(fold1_test)
    fold1_train_l = list(fold1_x_train_l)
    shuffle(fold1_train_l)
    fold1_train_u = list(fold1_x_train_u)
    shuffle(fold1_train_u)
    
    fold2_test = list(fold2_x_test)
    shuffle(fold2_test)
    fold2_train_l = list(fold2_x_train_l)
    shuffle(fold2_train_l)
    fold2_train_u = list(fold2_x_train_u)
    shuffle(fold2_train_u)
    
    fold3_test = list(fold3_x_test)
    shuffle(fold3_test)
    fold3_train_l = list(fold3_x_train_l)
    shuffle(fold3_train_l)
    fold3_train_u = list(fold3_x_train_u)
    shuffle(fold3_train_u)

    fold4_test = list(fold4_x_test)
    shuffle(fold4_test)
    fold4_train_l = list(fold4_x_train_l)
    shuffle(fold4_train_l)
    fold4_train_u = list(fold4_x_train_u)
    shuffle(fold4_train_u)

    return list(fold0_test), (list(fold0_train_l), list(fold0_train_u)),\
           list(fold1_test), (list(fold1_train_l), list(fold1_train_u)),\
           list(fold2_test), (list(fold2_train_l), list(fold2_train_u)),\
           list(fold3_test), (list(fold3_train_l), list(fold3_train_u)),\
           list(fold4_test), (list(fold4_train_l), list(fold4_train_u))
