## 划分训练集-验证集-测试集
## tf.csv,target.csv,label.csv     均用序号表示
import pandas as pd

data_type = 's4'
# dic = {} ## 所有基因对应的编号,模拟数据需要按照表达数据来读
# count = 0
# s=open('./processed_data/'+data_type+'/' + 'simulation4_gene_list.txt')#'mmukegg_new_new_unique_rand_labelx.txt')#)   ### read the gene pair and label file
# for line in s:
#     #print(line)
#     g_line = line.strip("\n").split()
#     #print(g_line[0])
#     g_line[1] = count
#     dic[g_line[0]] = count
#     count = count +1
#
# s.close()

## 按照表达数据读
dic = {} ## 所有基因对应的编号,模拟数据需要按照表达数据来读
count = 0
store = pd.HDFStore('./processed_data/' + 's4' + '/' + 'sim4' + '/' + 'ST_t' + str(0) + '.h5')
data = store['STrans'].T
data_index = data.index
tf_list = []
for i in data_index:
    tf_list.append(i.lower())
print(tf_list)
for line in tf_list :
    dic[line] = count
    count = count +1
print(dic)




## 写入tf.csv
dic2 = {}
label_list = []
all_label_list = []
# dic3,target序号集合
dic3 = {}
ss=open('./processed_data/'+data_type+'/' + 'simulation4_gene_pairs.txt')#'mmukegg_new_new_unique_rand_labelx.txt')#)   ### read the gene pair and label file
for line in ss:
    #print(line)
    #ll = [0,0]
    ## 读全部的标签
    ll = [0,0,0]

    t_line = line.strip("\n").split()
    if t_line[2] != '2':
        dic2[t_line[0]] = dic[t_line[0]]
        dic3[t_line[1]] = dic[t_line[1]]

        ## 标签全写入
        ll[0] = dic[t_line[0]]
        ll[1] = dic[t_line[1]]
        ll[2] = t_line[2]
        all_label_list.append(ll)

        #print('。。。。。')
        if t_line[2] == '1':
            print('------------------------')
            ll[0] = dic[t_line[0]]
            ll[1] = dic[t_line[1]]
            label_list.append(ll)
ss.close()
import os
import pandas as pd
#
path2 = './processed_data/'+data_type+'/TF.csv'
# 一次写入一行
c = 0
for d in dic2.keys():
    df = pd.DataFrame(data=[[d, dic2[d]]])
    #print(df)
    #解决追加模式写的表头重复问题
    if c == 0:
        df.to_csv(path2, header=['TF', 'index'], index=False, mode='a')
    else:
        df.to_csv(path2, header=False, index=False, mode='a')
    c = c+1
print(c)
## 写入lable
path3 = './processed_data/'+data_type+'/label.csv'
c = 0
for dd in all_label_list:
    df = pd.DataFrame(data=[[dd[0], dd[1] , dd[2]]])
    if c == 0:
        df.to_csv(path3, header=['TF', 'Target', 'Label'], index=False, mode='a')
    else:
        df.to_csv(path3, header=False, index=False, mode='a')
    c = c+1


import pandas as pd
#
c = 0
path = './processed_data/'+data_type+'/Target.csv'
# 一次写入一行
for d in dic.keys():
    df = pd.DataFrame(data=[[d, dic[d]]])
    #print(df)
    #解决追加模式写的表头重复问题
    if c == 0:
        df.to_csv(path, header=['Gene', 'index'], index=False, mode='a')
    else:
        df.to_csv(path, header=False, index=False, mode='a')
    c = c+1

