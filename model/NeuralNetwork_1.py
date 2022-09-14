import torch
import torch.nn as nn
import torch.nn.utils as utils
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import sys,os
sys.path.append(os.getcwd())
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
from gatmodel.models import Model as HGAT
from gatmodel.utils import load_data, load_vocab_len, load_user_tweet_graph, accuracy, evaluation_4class, convert_to_one_hot

from torch_geometric.nn import GCNConv, GATConv,SAGEConv
import copy
# 将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BUrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GATConv(in_feats, hid_feats)
        self.conv2 = GATConv(hid_feats, out_feats)

    def forward(self, x,edge,batchid):
        x, edge_index = x.to(device), edge.to(device)
        x = F.dropout(x)

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.best_acc = 0
        self.patience = 0
        self.init_clip_max_norm = 5.0
        self.device = torch.device('cpu')
        in_feats = 5000
        hid_feats = 64
        out_feats = 64
        self.TDrumorGCN =TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(300, 128, 300)
        task = 'twitter16'
        tweet_word_adj, features_index, labels, idx_train, idx_val, idx_test = load_data(task)
        vocab_size = load_vocab_len(task) + 1
        train_idx, dev_idx, test_idx, user_tweet_adj = load_user_tweet_graph(task,240,
                                                                             500)
        self.HGAT = HGAT(vocab_size=3098,
                         nfeat=300,
                         nhid=16,
                         gat_hidden_dim=16,
                         joint_dim=300,
                         features_index=features_index,
                         tweet_word_adj=tweet_word_adj,
                         user_tweet_adj=user_tweet_adj,
                         nclass=4,
                         dropout=0.5,
                         alpha=0.3).cuda()





    def forward(self):
        # 它的意思是如果这个方法没有被子类重写，但是调用了，就会报错。
        raise NotImplementedError
    # 最重要的
    def fit(self, X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred,
            X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev,y_test_cred,y_test_rucred,iter,task,idx_train,train_idx, idx_val, dev_idx):



        if torch.cuda.is_available():

            self.cuda()

        batch_size = self.config['batch_size']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], weight_decay=self.config['reg'], amsgrad=True) #
        # self.optimizer = torch.optim.Adadelta(self.parameters(), weight_decay=self.config['reg'])

        X_train_source_wid = torch.LongTensor(X_train_source_wid)
        X_train_source_id = torch.LongTensor(X_train_source_id)
        X_train_user_id = torch.LongTensor(X_train_user_id)
        X_train_ruid = torch.LongTensor(X_train_ruid)
        y_train = torch.LongTensor(y_train)
        y_train_cred = torch.LongTensor(y_train_cred)
        y_train_rucred = torch.LongTensor(y_train_rucred )
        y_test_cred = torch.LongTensor(y_test_cred)
        y_test_rucred = torch.LongTensor(y_test_rucred)





        # 把数据打包，每行分别有这些
        dataset = TensorDataset(X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred,idx_train, train_idx)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


        loss_func2 = nn.CrossEntropyLoss(ignore_index=3)
        for epoch in range(1, self.config['epochs']+1):
            print("\nEpoch ", epoch, "/", self.config['epochs'])
            self.train()
            avg_loss = 0
            avg_acc = 0
            loss_func = nn.CrossEntropyLoss()
            for data in zip(dataloader):
                    with torch.no_grad():
                        X_source_wid, X_source_id, X_user_id, X_ruid, batch_y, batch_y_cred, batch_y_rucred, idx_train, train_idx= (
                        item.to(device) for item in data)

                    # 清空过往梯度
                    self.optimizer.zero_grad()
                    logit = self.forward(X_source_wid, X_source_id, X_user_id, X_ruid,batch_y_cred,batch_y_rucred,task,idx_train,train_idx,f0_train_emotion, f0_train_replyemotion,word2_train_idx)
                    loss = loss_func(logit, batch_y)


                    # 反向传播，计算当前梯度
                    loss.backward()
                    # 更新所有的参数
                    self.optimizer.step()

                    corrects = (torch.max(logit, 1)[1].view(batch_y.size()).data == batch_y.data).sum()
                    accuracy = 100 * corrects / len(batch_y)
                    print('Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(0, loss.item(), accuracy, corrects,
                                                                                 batch_y.size(0)))

                    avg_loss += loss.item()
                    avg_acc += accuracy

                    if self.init_clip_max_norm is not None:
                        utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)

            cnt = y_train.size(0) // batch_size + 1
            print("Average loss:{:.6f} average acc:{:.6f}%".format(avg_loss/cnt, avg_acc/cnt))
            if epoch > self.config['epochs']//4 and self.patience > 2: #
                print("Reload the best model...")
                self.load_state_dict(torch.load(self.config['save_path']))
                now_lr = self.adjust_learning_rate(self.optimizer)
                print(now_lr)
                self.patience = 0
            self.evaluate(X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev,y_test_cred,y_test_rucred,epoch,task , idx_val,   dev_idx)



    def adjust_learning_rate(self, optimizer, decay_rate=.5):
        now_lr = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            now_lr = param_group['lr']
        return now_lr


    def evaluate(self, X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev,y_test_cred,y_test_rucred, epoch,task, idx_val,   dev_idx):
        y_pred = self.predict(X_dev_source_wid, X_dev_source_id, X_dev_user_id,X_dev_ruid,y_test_cred,y_test_rucred,task, idx_val,   dev_idx)
        acc = accuracy_score(y_dev, y_pred)
        print("Val set acc:", acc)
        print("Best val set acc:", self.best_acc)

        if acc > self.best_acc:  #
            self.best_acc = acc
            self.patience = 0
            torch.save(self.state_dict(),self.config['save_path'])
            print(classification_report(y_dev, y_pred, target_names=self.config['target_names'], digits=5))
            restr = classification_report(y_dev, y_pred, target_names=self.config['target_names'], digits=5)
            print("save model!!!")
            filename = 'result_samedata'+task+'.txt'
            with open(filename, 'a') as file_object:
                file_object.write('flodi==' + 'iter--' + str(iter) + '  epoch--' + str(epoch) + '      ')
                file_object.write(str(self.best_acc))
                file_object.write("\n")
            filename = 'result_samedata_more_'+task+'.txt'
            with open(filename, 'a') as file_object:

                file_object.write('flodi=='+ 'iter--' + str(iter) + '  epoch--' + str(epoch) + '      ')
                file_object.write(str(self.best_acc))
                file_object.write("\n")
                file_object.write(restr)
                file_object.write("\n")
                file_object.write("--------------------------------")
                file_object.write("\n")

        else:
            self.patience += 1


    def predict(self, X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid,y_test_cred,y_test_rucred,task, idx_val,   dev_idx):
        if torch.cuda.is_available():
            self.cuda()

        self.eval()
        y_pred = []
        X_dev_source_wid = torch.LongTensor(X_dev_source_wid)
        X_dev_source_id = torch.LongTensor(X_dev_source_id)
        X_dev_user_id = torch.LongTensor(X_dev_user_id)
        X_dev_ruid = torch.LongTensor(X_dev_ruid)
        y_test_cred = torch.LongTensor(y_test_cred)
        y_test_rucred = torch.LongTensor(y_test_rucred)

        dataset = TensorDataset(X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid,y_test_cred,y_test_rucred, idx_val, dev_idx )
        dataloader = DataLoader(dataset, batch_size=64)

        for data in zip(dataloader):
            with torch.no_grad():
                X_source_wid, X_source_id, X_user_id, \
                X_ruid, y_test_cred,y_test_rucred, idx_val, dev_idx= (item.to(device) for item in data)

            logits, _, _ = self.forward(X_source_wid, X_source_id, X_user_id, X_ruid,y_test_cred,y_test_rucred,task, idx_val,   dev_idx )
            predicted = torch.max(logits, dim=1)[1]
            y_pred += predicted.data.cpu().numpy().tolist()
        return y_pred
