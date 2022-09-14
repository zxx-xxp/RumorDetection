
import torch
import numpy as np
from scipy.sparse import coo_matrix
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, vocab_size, nfeat, nhid, gat_hidden_dim, joint_dim, features_index, tweet_word_adj, user_tweet_adj, nclass, dropout, alpha):
        super(Model, self).__init__()
        in_feats = 300
        hid_feats = 300
        out_feats = 300
        self.vocab_size = vocab_size
        self.nfeat = nfeat
        self.nhid = nhid
        self.gat_hidden_dim = gat_hidden_dim
        self.joint_dim = joint_dim
        self.features_index = features_index
        # self.features = features
        self.tweet_word_adj = tweet_word_adj.cuda()
        self.user_tweet_adj = user_tweet_adj.cuda()
        # self.wV = tweet_word_adj.shape[0]
        self.uV = user_tweet_adj.shape[0]

        self.user_tweet_embedding = nn.Embedding(self.uV, 300, padding_idx=0)
        nn.init.xavier_uniform_(self.user_tweet_embedding.weight)
        self.word_embedding = nn.Parameter(torch.zeros(size=(vocab_size, nfeat)))
        nn.init.normal(self.word_embedding.data, std=0.1)

        self.nclass = nclass
        self.dropout = dropout
        self.alpha = alpha


        # self.lamda = nn.Parameter(torch.rand(1))
        self.weight_W = nn.Parameter(torch.Tensor(joint_dim, joint_dim))

        self.weight_W1 = nn.Parameter(torch.Tensor(64, 64))
        self.weight_proj = nn.Parameter(torch.Tensor(joint_dim, 1))
        self.weight_proj1 = nn.Parameter(torch.Tensor(64, 1))

        # self.weight_proj = nn.Parameter(torch.Tensor(joint_dim, joint_dim))
        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

        # self.out1 = nn.Linear(2*joint_dim, joint_dim)
        # self.out1 = nn.Linear(joint_dim, 100)
        # self.out2 = nn.Linear(100, nclass)
        # self.relu = nn.ReLU()
        self.out = nn.Linear(joint_dim, nclass)
        self.init_weight()
        # print(self)

    def init_weight(self):
        torch.nn.init.xavier_normal_(self.out.weight)
        # torch.nn.init.xavier_normal_(self.out1.weight)
        # torch.nn.init.xavier_normal_(self.out2.weight)
    def getadj(self,adj):
        tadj = adj.to_dense().cpu().numpy()
        list0 = []
        list1 = []
        len = adj.shape[0]
        for i in range(0, len):
            for j in range(0, i):
                if tadj[i][j] > 0:
                    list0.append(i)
                    list1.append(j)
        list = []
        list.append(list0)
        list.append(list1)
        adjlist = torch.IntTensor(list)
        return adjlist



    def forward(self, tw_graph_idx, ut_graph_idx):




        twt_X_list = []
        for index in self.features_index:
            feature = torch.sum(self.word_embedding[index,:], 0).float().view(1,-1)/len(index)
            twt_X_list.append(feature)
        twt_X = torch.cat([feature for feature in twt_X_list], 0)


        tw_X = self.twt_gat(twt_X, self.tweet_word_adj)
        return tw_X[tw_graph_idx,:]
