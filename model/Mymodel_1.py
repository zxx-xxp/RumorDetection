
import torch.nn.init as init
from model.Transformer import *
from model.NeuralNetwork_1 import *

import os
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PGAN(NeuralNetwork):

    def __init__(self, config):
        super(PGAN, self).__init__()

        self.transformer = Transformer(10, 10, 0, 0, device=device).to(device)
        self.config = config
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        self.n_heads = config['n_heads']
        # A_us、uu是relations.pkl打开读到的东西
        # 要打印看看3

        self.A_us = config['A_us']
        self.A_uu = config['A_uu']
        embeding_size = config['embeding_size']
        fuse_dim = 300

        self.weight_W = nn.Parameter(torch.Tensor(fuse_dim, fuse_dim))
        self.weight_proj = nn.Parameter(torch.Tensor(fuse_dim, 1))
        self.word_embedding = torch.nn.Embedding(V, D, padding_idx=0, _weight=torch.from_numpy(embedding_weights).to(device))

        self.user_embedding = torch.nn.Embedding(config['A_us'].shape[0], embeding_size, padding_idx=0).to(device)
        self.source_embedding = torch.nn.Embedding(config['A_us'].shape[1], embeding_size).to(device)
        self.gat_embedding = torch.nn.Embedding(config['A_us'].shape[1], 300).to(device).to(device)
        self.fuseuser_embedding = torch.nn.Embedding(config['A_us'].shape[0], 300, padding_idx=0).to(device)


        # 卷积
        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])

        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=config['maxlen'] - K + 1) for K in config['kernel_sizes']])



        self.Wcm = [nn.Parameter(torch.FloatTensor(embeding_size, embeding_size)) for _ in
                    range(self.n_heads)]
        self.Wecm = [nn.Parameter(torch.FloatTensor(embeding_size, embeding_size)) for _ in
                    range(self.n_heads)]
        self.Wam = [nn.Parameter(torch.FloatTensor(embeding_size, embeding_size)) for _ in
                    range(self.n_heads)]
        # 逐元素计算平方根 根号d--点embedding的维度
        self.scale = torch.sqrt(torch.FloatTensor([embeding_size])).to(device) #  // self.n_heads

        self.W1 = nn.Parameter(torch.FloatTensor(embeding_size * self.n_heads, embeding_size)).to(device)
        self.W2 = nn.Parameter(torch.FloatTensor(embeding_size * self.n_heads, embeding_size)).to(device)
        self.We = nn.Parameter(torch.FloatTensor(embeding_size * self.n_heads, embeding_size)).to(device)

        # self.linear = nn.Linear(400, 200).to(device)
        self.linear = nn.Linear(200, 200).to(device)

        self.xlnetlayerlinear = nn.Linear(1024, 256).to(device)

        self.dropout = nn.Dropout(config['dropout']).to(device)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.fc = torch.nn.Linear((64 + 64) * 2, 100)

        self.linear_fuse = nn.Linear(600, 1)
        self.linear_fuse2 = nn.Linear(400, 1)
        self.linear_fuse3 = nn.Linear(400, 1)
        self.fc_out = nn.Sequential(
            nn.Linear(700, 100),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(100, config["num_classes"])
        ).to(device)
        self.fc_user_out = nn.Sequential(
            nn.Linear(embeding_size, 100),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(100, 3)
        ).to(device)
        self.fc_ruser_out = nn.Sequential(
            nn.Linear(embeding_size, 100),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(100, 3)
        ).to(device)
        print(self)
        self.init_weights()

        hid_dim = 5000
        n_heads = 6

        # 强制 hid_dim 必须整除 h
        # assert hid_dim % n_heads == 0
        # 定义 W_q 矩阵
        self.w_q = nn.Linear(hid_dim, hid_dim).to(device)
        # 定义 W_k 矩阵
        self.w_k = nn.Linear(hid_dim, hid_dim).to(device)
        # 定义 W_v 矩阵
        self.w_v = nn.Linear(hid_dim, hid_dim).to(device)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(0.1)
        self.newscale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def init_weights(self):
        # xavier初始化方法中服从正态分布
        init.xavier_normal_(self.user_embedding.weight)
        init.xavier_normal_(self.source_embedding.weight)
        # 循环次数为head的值
        for i in range(self.n_heads):
            init.xavier_normal_(self.Wcm[i])
            init.xavier_normal_(self.Wecm[i])
            init.xavier_normal_(self.Wam[i])

        init.xavier_normal_(self.W1)
        init.xavier_normal_(self.W2)
        init.xavier_normal_(self.We)
        init.xavier_normal_(self.linear.weight)
        for name, param in self.fc_out.named_parameters():
            if name.__contains__("weight"):
                init.xavier_normal_(param)
        for name, param in self.fc_user_out.named_parameters():
            if name.__contains__("weight"):
                init.xavier_normal_(param)
        for name, param in self.fc_ruser_out.named_parameters():
            if name.__contains__("weight"):
                init.xavier_normal_(param)
        print('')

    def publisher_encoder(self, X_user, X_user_id):
        m_hat = []
        for i in range(self.n_heads):
            m_hat.append(self.user_multi_head(X_user, X_user_id, self.Wcm[i]))

        m_hat = torch.cat(m_hat, dim=-1).matmul(self.W1)
        m_hat = self.elu(m_hat)
        m_hat = self.dropout(m_hat)

        U_hat = m_hat + X_user  # bsz x d
        return U_hat





    def text_representation(self, X_word):
        # 16 50==max 300
        X_word = X_word.permute(0, 2, 1)
        conv_block = []
        # cnn卷积
        for Conv, max_pooling in zip(self.convs, self.max_poolings):
            act = self.relu(Conv(X_word))
            pool = max_pooling(act).squeeze()
            conv_block.append(pool)

        features = torch.cat(conv_block, dim=1)
        features = self.dropout(features)
        # 16 300



        return features



    def getHGAT(self,task,X_source_id,X_user_id):



        word_output  = self.HGAT(X_source_id, X_user_id)
        fc_word_out = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(word_output.shape[-1], 300)
        ).to(device)
        word_output = fc_word_out(word_output)
        return word_output

    def gloabal_bi(self,x,edge,X_user_id):
        BU_x = self.BUrumorGCN(x, edge, 0)
        edgenew = torch.zeros((edge.shape[0], edge.shape[1]),dtype=torch.int32)

        edgenew[0] = edge[1]
        edgenew[1] = edge[0]
        edgenew = edgenew.to(device)
        TD_x = self.BUrumorGCN(x,edgenew,0)
        x = th.cat((BU_x, TD_x), 1)
        fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(x.shape[-1], 300)
        ).to(device)
        x = fc(x)
        newuserid = []
        for j in X_user_id:
            newuserid.append(int(j) - 1)

        x = x[newuserid, :]
        return x
    def forward(self, X_source_wid, X_source_id, X_user_id, X_ruser_id,batch_y_cred,batch_y_rucred,task, idx_text,    idx_user):  # , X_composer_id, X_reviewer_id


        robertreply=self.getrobertreply(X_source_id,task)
        data = np.load(os.path.join('H:\\zxx\\BiGCN\\FakeNewsDetection', 'dataset/' + task + '/' + 'usergcn.npz'),
                       allow_pickle=True)
        usergat = torch.LongTensor(data['usergat']).to(device)
        useredge = torch.LongTensor(data['useredge']).to(device)
        word_out= self.getHGAT(task, idx_text, idx_user)


        usergat_embedding = self.gat_embedding(usergat).to(device)
        gloabal_bi = self.gloabal_bi(usergat_embedding,useredge,X_user_id)


        X_source_wid = X_source_wid.to(device)

        X_word = self.word_embedding(X_source_wid).to(device)

        # cnn
        X_text = self.text_representation(X_word)



        y_cred = self.user_embedding(batch_y_cred)

        ucred_rep = self.publisher_encoder(y_cred, batch_y_cred)

        word_reply_fuse = torch.cat([word_out, robertreply], dim=-1)
        alpha = torch.sigmoid(self.linear_fuse(word_reply_fuse))
        X_word_reply = alpha * word_out + (1 - alpha) * robertreply
        #
        alpha1 = torch.sigmoid(self.linear_fuse(X_word_reply))
        ruser_user = alpha1 * X_word_reply + (1 - alpha1) * gloabal_bi

        tweet_rep = torch.cat([ruser_user, X_text], dim=-1)
        tweet_rep = torch.cat([tweet_rep, ucred_rep], dim=-1)


        Xt_logit = self.fc_out(tweet_rep)



        return Xt_logit

