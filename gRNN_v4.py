import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
from torch.nn import CosineSimilarity

class GRNN(nn.Module):
    def __init__(self,input_dim,hidden1_dim,hidden2_dim,hidden3_dim,output_dim,num_head1,num_head2,
                 alpha,device,type,reduction):
        super(GRNN, self).__init__()
        self.num_head1 = num_head1
        self.num_head2 = num_head2
        self.device = device
        self.alpha = alpha
        self.type = type
        self.reduction = reduction
        self.time_point = 4

        # self.attention = nn.Sequential(
        #     nn.Linear(128, 1),
        #     nn.Tanh(),
        #     nn.Softmax(dim=1)
        # )

        if self.reduction == 'mean':
            self.hidden1_dim = hidden1_dim
            self.hidden2_dim = hidden2_dim
        elif self.reduction == 'concate':
            self.hidden1_dim = num_head1*hidden1_dim
            self.hidden2_dim = num_head2*hidden2_dim

        ## 怎么每一层gat都是三个头部，并且没看到平均啊
        ## gat1
        self.ConvLayer1 = [AttentionLayer(input_dim,hidden1_dim,alpha) for _ in range(num_head1)]
        for i, attention in enumerate(self.ConvLayer1):
            ## ConvLayer1_AttentionHead1，ConvLayer1_AttentionHead2，ConvLayer1_AttentionHead3
            self.add_module('ConvLayer1_AttentionHead{}'.format(i),attention)
        ## gat2
        self.ConvLayer2 = [AttentionLayer(self.hidden1_dim,hidden2_dim,alpha) for _ in range(num_head2)]
        for i, attention in enumerate(self.ConvLayer2):
            self.add_module('ConvLayer2_AttentionHead{}'.format(i),attention)

        ## tf和target分开放入一个mlp里面
        ## MLP1
        self.tf_linear1 = nn.Linear(hidden2_dim,hidden3_dim)
        self.target_linear1 = nn.Linear(hidden2_dim,hidden3_dim)
        ## MLP2
        self.tf_linear2 = nn.Linear(hidden3_dim,output_dim)
        self.target_linear2 = nn.Linear(hidden3_dim, output_dim)

        ## 输入维度统一  特征重构
        # hesc2
        # self.nomal_linear1 = nn.Linear(92, input_dim)
        # self.nomal_linear2 = nn.Linear(102, input_dim)
        # self.nomal_linear3 = nn.Linear(66, input_dim)
        # self.nomal_linear4 = nn.Linear(172, input_dim)
        # self.nomal_linear5 = nn.Linear(138, input_dim)
        # self.nomal_linear6 = nn.Linear(188, input_dim)
        # mesc1
        #self.nomal_linear = nn.Linear(384, input_dim)
        # hesc1

        # self.nomal_linear1 = nn.Linear(81, input_dim)
        # self.nomal_linear2 = nn.Linear(190, input_dim)
        # self.nomal_linear3 = nn.Linear(377, input_dim)
        # self.nomal_linear4 = nn.Linear(415, input_dim)
        # self.nomal_linear5 = nn.Linear(466, input_dim)
        # mesc2
        self.nomal_linear1 = nn.Linear(933, input_dim)
        self.nomal_linear2 = nn.Linear(303, input_dim)
        self.nomal_linear3 = nn.Linear(683, input_dim)
        self.nomal_linear4 = nn.Linear(798, input_dim)


        ## 输出层
        if self.type == 'MLP':
            self.linear = nn.Linear(2*output_dim, 2) ## 两个低维表示做连接后输入到这层然后输出两个分类节点

        ## lstm层
        # self.rnn = nn.LSTM(  # LSTM 效果要比 nn.RNN() 好多了
        #     input_size=32,  # 图片每行的数据像素点
        #     hidden_size=256,  # rnn hidden unit
        #     num_layers=2,  # 有几层 RNN layers
        #     batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        # )

        ## gru层
        self.gru = nn.GRU(input_size=32, hidden_size=128, num_layers=2, batch_first=True)
        self.attention = Attention(128)


        #self.out = nn.Linear(128, 1)  # 输出层
        # self.out = nn.Linear(128,10)
        # self.fout = nn.Linear(10,1)

        self.out = nn.Linear(128, 16)
        self.final_out = nn.Linear(128,1)


        self.reset_parameters()

    def reset_parameters(self):
        for attention in self.ConvLayer1:
            attention.reset_parameters() ## attentionlayer类里面的函数

        for attention in self.ConvLayer2:
            attention.reset_parameters()

        nn.init.xavier_uniform_(self.tf_linear1.weight,gain=1.414)
        nn.init.xavier_uniform_(self.target_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.tf_linear2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear2.weight, gain=1.414)


    def encode(self,x,adj):
        ## 看att(x,adj)的输出维度吧，如果是一列，下面的操作就很迷？？盲猜是一行
        if self.reduction =='concate':
            x = torch.cat([att(x, adj) for att in self.ConvLayer1], dim=1) # 对列进行拼接，dim=0是对行进行拼接
            x = F.elu(x)

        elif self.reduction =='mean':
            x = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer1]), dim=0) # 相当于竖着拼接成的一条向量所有元素求均值？
            x = F.elu(x)

        else:
            raise TypeError


        with torch.no_grad():
            out = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer2]),dim=0) #修改过，加了一个梯度消除，不然内存不够

        return out


    def decode(self,tf_embed,target_embed):

        # if self.type =='dot':
        #     ## mul()两个张量对应元素相乘
        #     prob = torch.mul(tf_embed, target_embed)
        #     prob = torch.sum(prob,dim=1).view(-1,1) ##整个batch的预测结果，是一列tensor
        #
        #
        #     return prob
        #
        # elif self.type =='cosine':
        #     prob = torch.cosine_similarity(tf_embed,target_embed,dim=1).view(-1,1)
        #
        #     return prob
        #
        # elif self.type == 'MLP':
        #     h = torch.cat([tf_embed, target_embed],dim=1)
        #     prob = self.linear(h)
        #
        #     return prob
        # else:
        #     raise TypeError(r'{} is not available'.format(self.type))
        t_p = self.time_point
        h_t = []
        for i in range(t_p):
            h1 = torch.cat([tf_embed[i], target_embed[i]], dim=1)
            h_t.append(h1)
        # h1 = torch.cat([tf_embed[0], target_embed[0]], dim=1)
        # h2 = torch.cat([tf_embed[1], target_embed[1]], dim=1)
        # h3 = torch.cat([tf_embed[2], target_embed[2]], dim=1)
        # h4 = torch.cat([tf_embed[3], target_embed[3]], dim=1)
        #h5 = torch.cat([tf_embed[4], target_embed[4]], dim=1)
        #h6 = torch.cat([tf_embed[5], target_embed[5]], dim=1)
        h_x = torch.cat(h_t, dim = 1).view(-1,t_p,16*2)
        #prob = self.linear(h)
        ## 放到lstm里面去
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.gru(h_x, None)  # None 表示 hidden state 会用全0的 state 就是128维
        attention_output = self.attention(r_out)
        #attention_output = self.attention(r_out)
        linear_output = self.final_out(attention_output)
        # 这个地方选择lstm_output[-1]，也就是相当于最后一个输出，因为其实每一个cell（相当于图中的A）都会有输出，但是我们只关心最后一个
        # 选取最后一个时间点的 r_out 输出 r_out[:, -1, :] 的值，也是 h_n 的值
        # torch.Size([64, 28, 128])->torch.Size([64,128])
        #out = self.out(r_out[:,-1, :])  # torch.Size([64, 128])-> torch.Size([64, 10])    ##这里应该是256*2
        #out = F.relu(out)
        #out = F.dropout(out,p=0.5)
        #out = self.final_out(out)
        #out = F.leaky_relu(out)

        ## 多输出
        # l_out = []
        # for i in range(t_p):
        #     out = self.out(r_out[:, i, :])
        #     out = F.relu(out)
        #     l_out.append(out)
        # ll_out = torch.cat(l_out, dim=1)
        # #ll_out = self.out(ll_out)
        # #F.dropout(ll_out,p=0.01)
        # lll_out = self.final_out(ll_out)

        ## 池化操作
        # l_out = []
        # for i in range(t_p):
        #     out = self.out(r_out[:, i, :])
        #     #out = F.relu(out)
        #     #out = out.cpu().detach().numpy()
        #     l_out.append(out)
        #
        # feature_vectors = l_out
        # # 将所有特征向量按列连接成一个二维矩阵
        # features_matrix = torch.stack(feature_vectors,dim=1)
        # pooled_features = torch.mean(features_matrix,dim=1)
        #
        # lll_out = self.final_out(pooled_features)
        return linear_output

    ## 将训练数据输入网络求解pred
    def forward(self,x,adj,train_sample):
        ## 经过两层gat后的特征向量
        train_tf_total = []
        train_target_total = []
        for i in range(self.time_point):
            if i == 0:
                e = self.nomal_linear1(x[i])
            elif i == 1:
                e = self.nomal_linear2(x[i])
            elif i == 2:
                e = self.nomal_linear3(x[i])
            # elif i == 3:
            #     e = self.nomal_linear4(x[i])
            # # elif i == 4:
            # #     e = self.nomal_linear5(x[i])
            else:
                e = self.nomal_linear4(x[i])
            #e = self.nomal_linear(x[i])
            embed = self.encode(e,adj)

            tf_embed = self.tf_linear1(embed)
            tf_embed = F.leaky_relu(tf_embed)
            tf_embed = F.dropout(tf_embed,p=0.01)
            tf_embed = self.tf_linear2(tf_embed)
            tf_embed = F.leaky_relu(tf_embed)

            target_embed = self.target_linear1(embed)
            target_embed = F.leaky_relu(target_embed)
            target_embed = F.dropout(target_embed, p=0.01)
            target_embed = self.target_linear2(target_embed)
            target_embed = F.leaky_relu(target_embed)

            ## 这个是所有的基因经过网络层的低维向量化表示
            self.tf_ouput = tf_embed
            self.target_output = target_embed

            ## 提取出这个batch的数据，tf,target分离
            train_tf = tf_embed[train_sample[:,0]] # 取得是tf基因对应的特征向量（这个batch的所有的tf）
            train_target = target_embed[train_sample[:, 1]] # 取得是target基因对应的特征向量
            train_tf_total.append(train_tf)
            train_target_total.append(train_target)
        ## 返回最终lstm输出的结果
        pred = self.decode(train_tf_total, train_target_total)

        return pred

    def get_embedding(self):
        return self.tf_ouput, self.target_output

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim # gru隐藏层输出128维
        self.linear = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        # inputs: batch_size x seq_length x input_dim
        # 对输入进行线性变换
        linear_output = self.linear(inputs)
        # 计算注意力分布
        attention_weights = self.softmax(linear_output)
        # 利用注意力分布加权求和
        attention_output = torch.sum(attention_weights * inputs, dim=1)
        return attention_output


class AttentionLayer(nn.Module):
    def __init__(self,input_dim,output_dim,alpha=0.2,bias=True):
        super(AttentionLayer, self).__init__()

        self.input_dim = input_dim #细胞个数（一个基因的维度），421
        self.output_dim = output_dim #16
        self.alpha = alpha

        ## w = (output_dim*input_dim)
        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.weight_interact = nn.Parameter(torch.FloatTensor(self.input_dim,self.output_dim))
        ## a的维度是多少？？应该是1*2F
        self.a = nn.Parameter(torch.zeros(size=(2*self.output_dim,1))) # g1,g2做连接后与a相乘


        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)

        ## 初始化参数，保持每层输入和输出的方差相同
        self.reset_parameters()


    def reset_parameters(self):
        ## 初始化，xavier:通过网络层时，保持输入和输出的方差相同
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_interact.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def _prepare_attentional_mechanism_input(self, x):

        Wh1 = torch.matmul(x, self.a[:self.output_dim, :])
        Wh2 = torch.matmul(x, self.a[self.output_dim:, :])
        e = F.leaky_relu(Wh1 + Wh2.T,negative_slope=self.alpha)
        return e

    ############（data_feature,adj）att()
    def forward(self,x,adj):
        ## matmul()若两个tensor都是一维的，则返回两个向量的点积运算结果；若不是则返回两个矩阵相乘结果
        h = torch.matmul(x, self.weight) # x:1120*421   weight:16*421
        e = self._prepare_attentional_mechanism_input(h) ## 在

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense()>0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        # attention = F.softmax(e, dim=1)

        attention = F.dropout(attention, training=self.training)
        h_pass = torch.matmul(attention, h)

        output_data = h_pass


        output_data = F.leaky_relu(output_data,negative_slope=self.alpha)
        output_data = F.normalize(output_data,p=2,dim=1)


        if self.bias is not None:
            output_data = output_data + self.bias
        ############## 注意这里的输出维度
        return output_data













