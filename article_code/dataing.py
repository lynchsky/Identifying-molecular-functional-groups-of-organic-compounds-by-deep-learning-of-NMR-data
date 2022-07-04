import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *##plt中文
mpl.rcParams['font.sans-serif'] = ['SimHei']

##只有函数没有运行
def data_make():#读取training_set,training_label
    f1 = csv.reader(open('new_training_set.csv'))
    training_X = []
    for i in f1:
        #i的类型<class 'list'>
        #print(i[0])
        #print(i)
        #print(type(i))
        X = []  # 网络输入
        for j in i:
            # print(j)
            b = j.strip(" [']").split(' ')
            #b的类型<class 'list'>
            #print(b)
            #print(type(b))

            x = []  # 神经元输入
            for k in b:
                # print(k)
                x.append(float(k))
            #print(x)
            X.append(x)
        #print(X)
        training_X.append(X)
    #print(training_X)
    #print(len(training_X))
    #print(type(training_X[0][0][0]))

    f2 = csv.reader(open('new_training_label.csv'))
    training_Y = []
    for i1 in f2:
        #print(k)
        #print(i1[0])
        #print(type(i1))
        #a = i1
        Y = []
        for j in i1:
            Y.append(int(j))
        #print(Y)
        #print(type(Y))
        #print(type(Y[0]))
        training_Y.append(Y)
    #print(training_Y)
    #print(len(training_Y))

    ##删除最长的特征向量，剩下都在200以内

    return (training_X,training_Y)


def data_make_samelen():#读取training_slen_set,training_slen_label

    f1 = csv.reader(open('training_slen_set.csv'))
    training_X = []
    for i in f1:
        #i的类型<class 'list'>
        #print(i[0])
        #print(i)
        #print(type(i))
        X = []  # 网络输入
        for j in i:
            # print(j)
            X.append(round(float(j),4))
            #b的类型<class 'list'>
            #print(b)
            #print(type(b))

            #x = []  # 神经元输入
            #for k in b:
                # print(k)
                #x.append(float(k))
            #print(x)
            #X.append(x)
        #print(X)
        training_X.append(X)
    #print(training_X)
    #print(len(training_X))
    #print(type(training_X[0][0][0]))

    f2 = csv.reader(open('training_slen_label.csv'))
    training_Y = []
    for i1 in f2:
        #print(k)
        #print(i1[0])
        #print(type(i1))
        #a = i1
        Y = []
        for j1 in i1:
            Y.append(int(j1))
        #print(Y)
        #print(type(Y))
        #print(type(Y[0]))
        training_Y.append(Y)
    #print(training_Y)
    #print(len(training_Y))

    return (training_X,training_Y)

################查看数据的形式###############
'''x,y=data_make()
#print(len(x)) #516个元素
train_set = []
for i in range(len(x)):
    train_set.append([x[i],y[i]])
train_set = np.array(train_set,dtype=object)
frame = pd.DataFrame(train_set,columns=['set','label'])
frame['set_len'] = frame['set'].apply(lambda x : len(x))
print(frame.head(10))
print(frame['set_len'].describe())
_ = plt.hist(frame['set_len'],bins=100)
#bins参数是bin的个数，也就是总共要画多少条条状图
set_len = list(frame['set_len'])##pandas的series转为list
print(set_len.index(max(set_len)))##位置345

plt.xlabel('数据特征向量长度')
plt.title('数据特征直方图')
plt.show()'''

'''x,y=data_make_samelen()
print(len(x)) #516个元素
for i in range(10):
    print(x[i],y[i])'''









