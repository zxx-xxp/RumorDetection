import os
import os
import pickle
import torch
from sklearn.metrics import classification_report
import numpy as np
import datetime
import pickle
import torch
from sklearn.metrics import classification_report
from FakeNewsDetection.model.Mymodel_1 import PGAN
from FakeNewsDetection.dataset.twitter16.tarinusercred import gettrainusercred
from HGATRD.utils import load_vocab_len, load_user_tweet_graph, accuracy, evaluation_4class, convert_to_one_hot
from HGATRD.utils import load_data as load_data1
from collections import defaultdict
import numpy as np
from Process.process import *
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv
import copy
def delete(fold0_x_train, fold0_x_test, task):
    if task == 'twitter15':
        path = 'H:\\zxx\\BiGCN\\data\\Twitter15graph'
    else:
        path = 'H:\\zxx\\BiGCN\\data\\Twitter16graph'
    filelist = os.listdir(path)
    realfilelisr = []
    index1=[]
    index2=[]
    for i in filelist:
        realfilelisr.append(i.split('.')[0])
    train = fold0_x_train.tolist()
    test = fold0_x_test.tolist()
    for i in train:
        if i in realfilelisr:
            continue
        else:
            if i not in index1:
                index1.append(i)
            train.remove(i)
    for i in test:
        if i in realfilelisr:
            continue
        else:
            if i not in index2:
                index2.append(i)
            test.remove(i)
    for i in train:
        if i in realfilelisr:
            continue
        else:
            if i not in index1:
                index1.append(i)
            train.remove(i)
    for i in test:
        if i in realfilelisr:
            continue
        else:
            if i not in index2:
                index2.append(i)
            test.remove(i)
    id_index1 = []
    id_index2 = []
    count1 =0
    count2 = 0
    indexpath = 'H:\\zxx\\BiGCN\\FakeNewsDetection\\dataset\\'+task+'\\'+task+'.txt'
    indexidlist = []
    file = open(indexpath, encoding='utf-8')
    for line in file:
        line = line.split('	')
        indexidlist.append(line[1])
    for i in indexidlist:
        if i in index1:
            id_index1.append(count1)
        else:
            count1 = count1+1
    for i in indexidlist:
        if i in index2:
            id_index2.append(count2)
        else:
            count2 = count2+1


    return np.array(train), np.array(test),id_index1 ,id_index2


#     删除多的
def  getuseredgecred(X_train_user_id,X_train_ruid,y_train_cred,y_train_rucred):
    len1 = len(X_train_ruid)
    newuserdic = defaultdict(list)
    edge0 = []
    edge1 = []
    for i in range(0,len1):
        for j in X_train_ruid[i]:
            if j == 0:
                break
            else:
                edge0.append(X_train_user_id[i])
                edge1.append(j)
                # if

    uids = pickle.load(
        open("dataset/" + task + "/Uids.pkl", 'rb'))
    return edge0,edge1

def load_dataset(task, i):
    # mydataset
    if task == 'twitter16':
        datasetname = 'Twitter16'
    else:
        datasetname = 'Twitter15'
    spiltdata = np.load(
        os.path.join("H:\\zxx\\BiGCN\\FakeNewsDetection", 'dataset/spiltdata/' + datasetname + '_same.npz'),
        allow_pickle=True)
    fold0_x_test = spiltdata['test']
    fold0_x_train = spiltdata['train']
    fold0_x_val = spiltdata['val']



    tweet_word_adj, features_index, labels, idx_train, idx_val, idx_test = load_data1(task)
    vocab_size = load_vocab_len(task) + 1
    train_idx, dev_idx, test_idx, user_tweet_adj = load_user_tweet_graph(task, 60,
                                                                         500)

    idx_train, idx_val, idx_test = idx_train.tolist(), idx_val.tolist(), idx_test.tolist()
    train_idx, dev_idx, test_idx = train_idx.tolist(), dev_idx.tolist(), test_idx.tolist()



    idx_train, idx_val, idx_test =torch.LongTensor(idx_train),torch.LongTensor(idx_val),torch.LongTensor(idx_test)
    train_idx, dev_idx, test_idx =torch.LongTensor(train_idx),torch.LongTensor(dev_idx),torch.LongTensor(test_idx)




    A_us, A_uu = pickle.load(open("dataset/" + task + "/relations.pkl", 'rb'))
    X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred, word_embeddings = pickle.load(
        open("dataset/" + task + "/train.pkl", 'rb'))
    X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev = pickle.load(
        open("dataset/" + task + "/dev.pkl", 'rb'))
    X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid, y_test = pickle.load(
        open("dataset/" + task + "/test.pkl", 'rb'))

    # t  = pickle.load(
    #     open("dataset/" + task + "/vocab.pkl", 'rb'))
    uids = pickle.load(
        open("dataset/" + task + "/Uids.pkl", 'rb'))

    config['maxlen'] = len(X_train_source_wid[0])
    # mydataset
    # train
    # fold0_x_train,fold1_x_train,fold2_x_train,fold3_x_train,fold4_x_train = def origintonew(fold0_x_train,fold1_x_train,fold2_x_train,fold3_x_train,fold4_x_train, X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred, word_embeddings)
    # mydataset
    new_source_wid = X_train_source_wid + X_dev_source_wid + X_test_source_wid
    new_source_id = np.append(np.append(X_train_source_id, X_dev_source_id), X_test_source_id)
    new_user_id = X_train_user_id + X_dev_user_id + X_test_user_id
    new_ruid = X_train_ruid + X_dev_ruid + X_test_ruid
    new_y = y_train + y_dev + y_test
    # fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train ,fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test= origintonew(fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train,fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test,new_source_wid,new_source_id,new_user_id,new_ruid,new_y)
    fold0_train_source_wid, fold0_train_source_id, fold0_train_user_id, fold0_train_ruid, fold0_y_train, \
    fold0_test_source_wid, fold0_test_source_id, fold0_test_user_id, fold0_test_ruid, fold0_y_test = origintonew(
        fold0_x_train, fold0_x_test, new_source_wid, new_source_id, new_user_id, new_ruid, new_y)

    fold0_train_source_wid, fold0_train_source_id, fold0_train_user_id, fold0_train_ruid, fold0_y_train, \
    fold0_val_source_wid, fold0_val_source_id, fold0_val_user_id, fold0_val_ruid, fold0_y_val = origintonew(
        fold0_x_train, fold0_x_val, new_source_wid, new_source_id, new_user_id, new_ruid, new_y)

    flod0_train_cred, y0_train_rucred = gettrainusercred(fold0_train_user_id, fold0_train_ruid, uids, task)

    flod0_test_cred, y0_test_rucred = gettrainusercred(fold0_test_user_id, fold0_test_ruid, uids, task)
    flod0_val_cred, y0_val_rucred = gettrainusercred(fold0_val_user_id, fold0_val_ruid, uids, task)


    # print('A_us', A_us)
    # print('A_uu', A_uu)
    # print('word_embeddings', word_embeddings)
    # 打印看看pkl里面是什么东西2
    if task == 'twitter15':
        config['n_heads'] = 10
    elif task == 'twitter16':
        config['n_heads'] = 8
    else:
        config['n_heads'] = 7
        config['batch_size'] = 128
        config['num_classes'] = 2
        config['target_names'] = ['NR', 'FR']
    # print(config)

    config['embedding_weights'] = word_embeddings
    config['A_us'] = A_us
    config['A_uu'] = A_uu
    # return X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred, \
    #        X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev, \
    #        X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid, y_test
    return fold0_train_source_wid, fold0_train_source_id, fold0_train_user_id, fold0_train_ruid, fold0_y_train, flod0_train_cred, y0_train_rucred, \
           fold0_val_source_wid, fold0_val_source_id, fold0_val_user_id, fold0_val_ruid, fold0_y_val, flod0_val_cred, y0_val_rucred, \
           fold0_test_source_wid, fold0_test_source_id, fold0_test_user_id, fold0_test_ruid, fold0_y_test, flod0_test_cred, y0_test_rucred, \
           idx_train, idx_val, idx_test,train_idx, dev_idx, test_idx



def train_and_test(model, task, iter, totaliter):
    model_suffix = model.__name__.lower().strip("text")

    # y_train_rucred随便整的
    print("iter:", iter)

    for i in range(0,1):
        print("==flodi==", i)
        X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred, \
        X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev, y_dev_cred, y_dev_rucred, \
        X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid, y_test, y_test_cred, y_test_rucred, \
        flod_x_train, idx_val, idx_test,train_idx, dev_idx, test_idx,f0_train_emotion, f0_train_replyemotion ,f0_val_emotion, f0_val_replyemotion,f0_test_emotion, f0_test_replyemotion,\
        word2_train_idx, word2_dev_idx,word2_test_idx= load_dataset(
            task, i)
        config['save_path'] = 'checkpoint/weights.best.' + str(iter) + task + "." + model_suffix

        nn = model(config)
        nn.fit(X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred,
               y_train_rucred,
               X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid, y_test, y_test_cred, y_test_rucred, iter,
               task,  train_idx, idx_test, test_idx,
              )

    #

    print("==============test================")
    nn.load_state_dict(torch.load(config['save_path'],map_location=torch.device('cpu')))
    # X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_test_cred, y_test_rucred, flod_x_test, task
    y_pred = nn.predict(X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid,y_test_cred, y_test_rucred, task, idx_test, test_idx)
    print(classification_report(y_test, y_pred, target_names=config['target_names'], digits=3))
    filename = 'result_samedata_test_' + task + '.txt'
    with open(filename, 'a') as file_object:
        file_object.write('iter--' + str(iter))
        file_object.write("\n")
        file_object.write(classification_report(y_test, y_pred, target_names=config['target_names'], digits=3))
        file_object.write("\n")
        file_object.write("--------------------------------")


config = {
    'lr': 1e-3,
    'reg': 1e-6,
    'embeding_size': 100,
    'batch_size': 15,
    'nb_filters': 100,
    'kernel_sizes': [3, 4, 5],
    'dropout': 0.5,
    'epochs': 20,
    'num_classes': 4,
    'target_names': ['NR', 'FR', 'TR', 'UR']
}

if __name__ == '__main__':
    # task = 'twitter16'
    task = 'twitter15'
    model = PGAN
    iter = 18
    batch_size = 1
    for i in range(0, iter):
        totaliter = []
        train_and_test(model, task, i, totaliter)


