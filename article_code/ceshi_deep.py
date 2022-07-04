import dataing ##读取数据
import data_same_length ##处理成定长数据


import tensorflow as tf
from tensorflow.keras.layers import Dense

import numpy as np
import pandas as pd
import random


#1.读入数据,打乱顺序
x,y = dataing.data_make()
train_set = []
for i in range(len(x)):
    train_set.append([x[i],y[i]])
random.shuffle(train_set)###数据重排
x_set = [e[0] for e in train_set]#特征数据
y_set = [f[1] for f in train_set]#标签

#2.数据长度归一化
length_x = []
for i in x_set:
    length_x.append(len(i))
max_length = max(length_x)
print('训练数据最大长度是：',max_length)
x_set = data_same_length.same_length(x_set,max_length)
print(x_set[0])
print(len(x_set[0]))

##测试1，每个神经元都是一维输入
'''x_set = []
for i in x:
    linshi_x = []
    for j in i:
        linshi_x.append(j[0])
        linshi_x.append(j[1])
    x_set.append(linshi_x)'''

'''print(x_set)
print(len(x_set))
print(x_set[0])'''

#训练集
x_tr = np.array(x_set)
y_tr = np.array(y_set)
#print(x_set)
x_train = x_tr[0:500,:,0]
y_train = y_tr[0:500,3]
#print(x_set.shape)
#测试集
x_test=x_tr[501:,:,0]
y_test=y_tr[501:,3]




#计算标签中0-1个数
def number_label(arr):
    length = len(arr)
    number_list = [0]*length
    number_0 = 0
    number_1 = 0
    for i in range(length):
        if arr[i] == number_list[i]:
            number_0 += 1
        else:
            number_1 += 1
    print('0的数目是：', number_0 , '   ', '1的数目是：', number_1)

number_label(y_train)


'''x_test=x_tr[501:,:,0]
y_test=y_tr[501:,3]'''
'''for _ in range(20):
    x_test.append(random.choice(x_train))
    y_test.append(random.choice(y_train))
x_test = np.array(x_test)
y_test = np.array(y_test)'''
#print(x_test,y_test)


print('1',x_train.shape)
print(y_train.shape)

def deep_model(feature_dim,label_dim):
    model = tf.keras.Sequential()
    print("create model. feature_dim ={}, label_dim ={}".format(feature_dim, label_dim))
    model.add(Dense(300, activation='relu', input_dim=feature_dim))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(30,activation='relu'))
    model.add(Dense(label_dim, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_deep(x_train,y_train,x_test,y_test):
    feature_dim = x_train.shape[1]
    label_dim =1 #y_train.shape[0]
    model = deep_model(feature_dim,label_dim)
    model.summary()
    model.fit(x_train,y_train,batch_size=1, epochs=8,validation_data=(x_test,y_test))

train_deep(x_train,y_train,x_test,y_test)

