from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam
from gRNN_v4 import GRNN
from torch.optim.lr_scheduler import StepLR
#import scipy.sparse as sp
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation,  Network_Statistic
import pandas as pd
#from torch.utils.tensorboard import SummaryWriter
#from Code.PytorchTools import EarlyStopping
import numpy as np
import random
#import glob
import os
from sklearn import metrics

#import time
import argparse
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
time_point = 4
d_type = 'mesc2'
d_short = 'm2_expression'
model_version = 'v4_' ## 输入的时候利用一个线性层映射到最长维度的一个特征空间，同时结合rnn的多个输出,并且最终用注意力机制加权来输出任意多个时间点的结合特征向量，从而达到目的

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
#parser.add_argument('--epochs', type=int, default= 90, help='Number of epoch.')
parser.add_argument('--epochs', type=int, default= 5, help='Number of epoch.')
parser.add_argument('--num_head', type=list, default=[3,3], help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden_dim', type=int, default=[128,64,32], help='The dimension of hidden layer')
parser.add_argument('--output_dim', type=int, default=16, help='The dimension of latent layer')
parser.add_argument('--batch_size', type=int, default=512, help='The size of each batch')
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--Type',type=str,default='dot', help='score metric')
parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('--reduction',type=str,default='concate', help='how to integrate multihead attention')
#parser.add_argument('--time_point', type=int, default= 4, help='Number of epoch.')
args = parser.parse_args()
seed = args.seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


"""
Non-Specific
mHSC-L learning rate = 3e-5
"""
data_type = 'hESC'
num = 500


def embed2file(tf_embed,tg_embed,gene_file,tf_path,target_path):
    tf_embed = tf_embed.cpu().detach().numpy()
    tg_embed = tg_embed.cpu().detach().numpy()

    gene_set = pd.read_csv(gene_file)

    tf_embed = pd.DataFrame(tf_embed,index=gene_set['Gene'].values)
    tg_embed = pd.DataFrame(tg_embed, index=gene_set['Gene'].values)

    tf_embed.to_csv(tf_path)
    tg_embed.to_csv(target_path)


net_type = "Specific"
## 返回网络密度target/tf*gene
# density = Network_Statistic(data_type,num,net_type)
# exp_file = './'+data_type+'/TFs+'+str(num)+'/BL--ExpressionData.csv'
# tf_file = './'+data_type+'/TFs+'+str(num)+'/TF.csv'
# target_file = './'+data_type+'/TFs+'+str(num)+'/Target.csv'

density = 1/36
# exp_file1 = './raw_data/mesc1/m1_expression0.csv'
# exp_file2 = './raw_data/mesc1/m1_expression1.csv'
# exp_file3 = './raw_data/mesc1/m1_expression2.csv'
# exp_file4 = './raw_data/mesc1/m1_expression3.csv'
# exp_file5 = './raw_data/mesc1/m1_expression4.csv'
# exp_file6 = './raw_data/mesc1/m1_expression5.csv'
# exp_file7 = './raw_data/mesc1/m1_expression6.csv'
# exp_file8 = './raw_data/mesc1/m1_expression7.csv'
# exp_file9 = './raw_data/mesc1/m1_expression8.csv'
tf_file = 'processed_data/'+d_type+'/TF.csv'
target_file = 'processed_data/'+d_type+'/Target.csv'

#data_input = pd.read_csv(exp_file,index_col=0)
# data1_input = pd.read_csv(exp_file1,index_col=0)
# data2_input = pd.read_csv(exp_file2,index_col=0)
# data3_input = pd.read_csv(exp_file3,index_col=0)
# data4_input = pd.read_csv(exp_file4,index_col=0)
# data5_input = pd.read_csv(exp_file1,index_col=0)
# data2_input = pd.read_csv(exp_file2,index_col=0)
# data3_input = pd.read_csv(exp_file3,index_col=0)
# data4_input = pd.read_csv(exp_file4,index_col=0)
#data5_input = pd.read_csv(exp_file5,index_col=0)
#data6_input = pd.read_csv(exp_file6,index_col=0)
#data_input = data1_input.append(data2_input).append(data3_input).append(data4_input).append(data5_input).append(data6_input)
#print(data_input)
## 实例化这个load_data类
# loader1 = load_data(data1_input)
# loader2 = load_data(data2_input)
# loader3 = load_data(data3_input)
# loader4 = load_data(data4_input)
#loader5 = load_data(data5_input)
#loader6 = load_data(data6_input)
## 可以调用类里面得函数了
# feature1 = loader1.exp_data() # 标准化后的数据，1120(gene)*421(cell)
# feature2 = loader2.exp_data()
# feature3 = loader3.exp_data()
# feature4 = loader4.exp_data()
#feature5 = loader5.exp_data()
#feature6 = loader6.exp_data()

#feature = np.array([feature1,feature2,feature3,feature4,feature5,feature6])
#print(feature)
tf = pd.read_csv(tf_file)['index'].values.astype(np.int64) #tf的索引(全部基因(target)是0-1119)张量
target = pd.read_csv(target_file)['index'].values.astype(np.int64) #相当于所有基因
# feature1 = torch.from_numpy(feature1)
# feature2 = torch.from_numpy(feature2)
# feature3 = torch.from_numpy(feature3)
# feature4 = torch.from_numpy(feature4)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
## 直接用循环
# time_point = 6
# d_type = 'hesc2'
# d_short = 'h2_expression'
data_feature = []
for i in range(time_point):
    exp_file = './processed_data/' + d_type + '/' + d_short + str(int(i)) + '.csv'
    data_input = pd.read_csv(exp_file, index_col=0)
    loader = load_data(data_input)
    feature = loader.exp_data()
    feature = torch.from_numpy(feature)
    d_feature = feature.to(device)
    data_feature.append(d_feature)


#feature5 = torch.from_numpy(feature5)
#feature6 = torch.from_numpy(feature6)
tf = torch.from_numpy(tf)

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#feature = [feature1,feature2,feature3,feature4,feature5,feature6]
# data_feature1 = feature1.to(device)
# data_feature2 = feature2.to(device)
# data_feature3 = feature3.to(device)
# data_feature4 = feature4.to(device)
# #data_feature5 = feature5.to(device)
# #data_feature6 = feature6.to(device)
# data_feature = [data_feature1,data_feature2,data_feature3,data_feature4]
tf = tf.to(device)

## data_feature(tensor),tf(tensor),target(numpy)

## 已经划分好的数据集
# train_file = './Train_validation_test/'+data_type+' '+str(num)+'/Train_set.csv'
# test_file = './Train_validation_test/'+data_type+' '+str(num)+'/Test_set.csv'
# val_file = './Train_validation_test/'+data_type+' '+str(num)+'/Validation_set.csv'

train_file = 'processed_data/'+d_type+'/Train_set.csv'
test_file = 'processed_data/'+d_type+'/Test_set.csv'
val_file = 'processed_data/'+d_type+'/Validation_set.csv'

## 所有基因经过网络层最后的低维向量化表示？？？（tf与target）
tf_embed_path = r'Result/'+data_type+' '+str(num)+'/Channel1.csv'
target_embed_path = r'Result/'+data_type+' '+str(num)+'/Channel2.csv'
if not os.path.exists('Result/'+data_type+' '+str(num)):
    os.makedirs('Result/'+data_type+' '+str(num))

train_data = pd.read_csv(train_file, index_col=0).values
validation_data = pd.read_csv(val_file, index_col=0).values
test_data = pd.read_csv(test_file, index_col=0).values

## 实例化这个类，都是套路
train_load = scRNADataset(train_data, data_feature[0].shape[0], flag=args.flag)
## 调用类里面的函数处理问题
adj = train_load.Adj_Generate(tf,loop=args.loop) # 用train_set组成的矩阵（这是叫做先验知识嘛？？）阿哲


adj = adj2saprse_tensor(adj)

## (tf,target,label)
train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)
val_data = torch.from_numpy(validation_data)

torch.cuda.empty_cache()
## 网络结构
## h1 = 100 h2 = 64 m1 = 450 m2 = 185

model = GRNN(input_dim=450, # 这个输入维度是什么操作，细胞的个数,正确的
                #time_point = args.time_point,
                hidden1_dim=args.hidden_dim[0],
                hidden2_dim=args.hidden_dim[1],
                hidden3_dim=args.hidden_dim[2],
                output_dim=args.output_dim,
                num_head1=args.num_head[0],
                num_head2=args.num_head[1],
                alpha=args.alpha,
                device=device,
                type=args.Type,
                reduction=args.reduction
                #time_point = time_point
                )


## 指定设备
# device_ids = [0,1,2,3]
# model = torch.nn.DataParallel(model, device_ids=device_ids)

adj = adj.to(device)
model = model.to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)
validation_data = val_data.to(device)

## 指定优化方法
optimizer = Adam(model.parameters(), lr=args.lr)
## 学习率衰减函数，每训练1个epoch，学习率衰减为原来的0.99
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

model_path = 'model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)


## 封装到dataloadr里面，每一次取出batch进行迭代训练，优化网络参数
for epoch in range(args.epochs):
    running_loss = 0.0

    ## DataLoader就是用来包装所使用的数据，每次从数据库train_load抛出batch_size个数据
    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        model.train()
        optimizer.zero_grad()

        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1) # view()相当于reshape()


        # train_y = train_y.to(device).view(-1, 1)
        #model = model.module
        pred = model(data_feature, adj, train_x) # model.forward(data_feature,adj,train_x)


        #pred = torch.sigmoid(pred)
        if args.flag:
            pred = torch.softmax(pred, dim=1)
        else:
            pred = torch.sigmoid(pred)
        loss_BCE = F.binary_cross_entropy(pred, train_y)


        loss_BCE.backward() # 计算梯度
        optimizer.step() #更新参数
        scheduler.step() #更新学习率衰减（学习率会越来越小）

        running_loss += loss_BCE.item()


    model.eval()
    score = model(data_feature, adj, validation_data)
    ## 是否进行因果推断
    if args.flag:
        score = torch.softmax(score, dim=1)
    else:
        score = torch.sigmoid(score)

    # score = torch.sigmoid(score)

    AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=validation_data[:, -1],flag=args.flag)
    #AUC, AUPR, AUPR_norm = Evaluation(y_true=validation_data[:, -1], y_pred=score, flag=args.flag)
        #
    print('Epoch:{}'.format(epoch + 1),
            'train loss:{}'.format(running_loss),
            'AUC:{:.3F}'.format(AUC),
            'AUPR:{:.3F}'.format(AUPR))
## 只保存模型参数，加载模型进行预测的时候还需要定义同一网络结构
torch.save(model.state_dict(), model_path + data_type+' '+str(num)+'.pkl')

## 同时保存模型参数和网络结构
torch.save(model,model_path +model_version + d_type + '_' + str(time_point) + '.pkl')

model.load_state_dict(torch.load(model_path + data_type+' '+str(num)+'.pkl'))
model.eval()
tf_embed, target_embed = model.get_embedding()
embed2file(tf_embed,target_embed,target_file,tf_embed_path,target_embed_path)

score = model(data_feature, adj, test_data)
if args.flag:
    score = torch.softmax(score, dim=1)
else:
    score = torch.sigmoid(score)
# score = torch.sigmoid(score)

# y_p = score.cpu().detach().numpy()
# y_p = y_p.flatten()
# y_t = test_data[:, -1].cpu().numpy().flatten().astype(int)
# fpr, tpr, thresholds = metrics.roc_curve(test_data[:, -1], score, pos_label=1)
# plt.plot(fpr, tpr)
# plt.grid()
# plt.plot([0, 1], [0, 1])
# plt.xlabel('FP')
# plt.ylabel('TP')
# plt.title('t_roc')
# plt.ylim([0, 1])
# plt.xlim([0, 1])
# plt.show()
AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=test_data[:, -1],flag=args.flag)


## 阈值选取
# score = model(data_feature, adj, validation_data)
# if args.flag:
#     score = torch.softmax(score, dim=1)
# else:
#     score = torch.sigmoid(score)
#
# fpr, tpr, thresholds = metrics.roc_curve(validation_data[:, -1], score, pos_label=1)
# plt.plot(fpr, tpr)
# plt.grid()
# plt.plot([0, 1], [0, 1])
# plt.xlabel('FP')
# plt.ylabel('TP')
# plt.title('v_roc')
# plt.ylim([0, 1])
# plt.xlim([0, 1])
# plt.show()
# thired =


print('AUC:{}'.format(AUC),
     'AUPRC:{}'.format(AUPR))



