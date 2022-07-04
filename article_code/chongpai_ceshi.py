# -*- coding: utf-8 -*-
#@Author  : lynch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense,Dropout,LSTM, SpatialDropout1D, Bidirectional


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn import metrics
import tensorflow.keras.backend as K

#在自己的库函数
import dataing ##读取数据
import data_same_length ##处理成定长数据
import loss_function



##平衡数据：0第一类{1: 330, 0: 187}，1第二类{1: 393, 0: 124}，2第三类{1: 347, 0: 170}
########： 3第四类{0: 430, 1: 87}，4第五类{1: 316, 0: 201}，5第六类{0: 482, 1: 35}
########： 6第七类{0: 444, 1: 73}，7第八类{0: 432, 1: 85}，8第九类{0: 422, 1: 95}，9第十类{0: 502, 1: 15}

#读取数据
x,y = dataing.data_make()##不定长特征
#x,y = dataing.data_make_samelen()##定长特征

train_set = []
for i in range(len(x)):
    random.shuffle(x[i])
    #print(x[i])
    #a =x[i].copy()
    train_set.append([x[i],y[i]])

random.shuffle(train_set)###数据重排
x_set1 = [e[0] for e in train_set]#特征数据
y_set1 = [f[1] for f in train_set]#标签


'''x1 = x.copy()
y1 = np.array(y)[:,0].tolist()
print(len(x))
print(type(y1))

def chongpai(xdata,ydata,number_label):
    for epoch in range(number_label):
        for i in range(len(ydata)):
            if ydata[i] == 0:
                data = list(reversed(xdata[i]))
                #print(data)
                label = ydata[i]
                xdata.append(data)
                ydata.append(label)
    return xdata,ydata

x_set,y_set =chongpai(x1,y1,1)
#print(len(x))
print(len(x_set))
print(len(y_set))

#数据打乱处理
train_set = []
for i in range(len(x_set)):
    train_set.append([x_set[i],y_set[i]])
random.shuffle(train_set)###数据重排
x_set1 = [e[0] for e in train_set]#特征数据
y_set1 = [f[1] for f in train_set]#标签
print(type(x_set1))'''

##训练集测试集划分
max_length =268
x_set2 = keras.preprocessing.sequence.pad_sequences(x_set1, maxlen=max_length, dtype='float64',padding='post',value = [0.0,0.0])
y_set2 = np.array(y_set1)
x_train = x_set2[0:450]
y_train = y_set2[0:450]
x_test = x_set2[450:]
y_test = y_set2[450:]

model = keras.models.Sequential()
# 添加一个Masking层
model.add(layers.Masking(mask_value=0.0, input_shape=(2, 2)))
# 添加一个RNN层
rnn_layer =layers.SimpleRNN(150, return_sequences=False)
model.add(rnn_layer)
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30,activation='relu'))
#多个标签
#model.add(Dense(10, activation='sigmoid'))
#单个标签
model.add(Dense(10, activation='sigmoid'))

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy']) ###,'top_k_categorical_accuracy'])#metrics.hanming_loss
model.summary()#####loss_function.multilabel_categorical_crossentropy
history = model.fit(x_train,y_train,batch_size=16, epochs=50,validation_data=(x_test,y_test))


print(model.predict(x_test[0:10]))
print(y_test[0:10])

# 绘制训练 & 验证的准确率值
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model accuracy and loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Epoch')
plt.legend(['train_acc', 'test_acc','train_loss','test_loss'], loc='upper left')
plt.show()
