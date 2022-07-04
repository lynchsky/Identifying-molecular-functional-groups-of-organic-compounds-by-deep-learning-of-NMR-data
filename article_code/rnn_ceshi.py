# -*- coding: utf-8 -*-
# @Author  : lynch

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import random
from sklearn import metrics
import tensorflow.keras.backend as K
#在自己的库函数
import dataing ##读取数据
import data_same_length ##处理成定长数据
import loss_function

#1.读入数据,打乱顺序
x,y = dataing.data_make()
x_set = np.array(x,dtype=object)
train_set = []
for i in range(len(x)):
    train_set.append([x[i],y[i]])
random.shuffle(train_set)###数据重排
x_set = [e[0] for e in train_set]#特征数据
y_set = [f[1] for f in train_set]#标签

##训练数据最大长度,可以在dataing程序中查看，目前是276
max_length = 268

##keras.preprocessing.sequence.pad_sequences将多个序列截断或补齐为相同长度，返回numpy数组
X = keras.preprocessing.sequence.pad_sequences(x_set, maxlen=max_length, dtype='float64',padding='post',value = [0.0,0.0])
#print(X[0])
print(X.shape)

x_train = X[0:450]
y_train = np.array(y_set)[0:450]
print(y_train.shape)
x_test = X[450:]
y_test = np.array(y_set)[450:]
print(y_test[1])
print(x_test[1])
'''import 查看数据平衡
x,y= 查看数据平衡.oversamp()
x_train = x[0:600]
y_train = y[0:600]
x_test = x[600:]
y_test = y[600:]'''


model = keras.models.Sequential()
# 添加一个Masking层
model.add(layers.Masking(mask_value=0.0, input_shape=(2, 2)))
# 添加一个RNN层
rnn_layer =layers.SimpleRNN(50, return_sequences=False)
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

adam = keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','binary_accuracy']) ###,'top_k_categorical_accuracy'])#metrics.hanming_loss
model.summary()#####loss_function.multilabel_categorical_crossentropy
history = model.fit(x_train,y_train,batch_size=16, epochs=30,validation_data=(x_test,y_test))



def result_trans(x_pred):
    x_result = []
    for i in x_pred:
        result = [0] * 10
        for j in range(10):
            if i[j] >= 0.5:
                result[j] = 1
            else:
                result[j] = 0
        x_result.append(result)
    return x_result

def duibi_result(y_pred,y_true):
    result = 0
    '''for i in range(len(y_pred)):
        if y_pred[i]==y_true[i]:
            result += 1'''
    for i in range(7):
        if y_pred[i] == y_true[i]:
            result += 10
    for j in range(7,10):
        if y_pred[j] == y_true[j]:
            result += 1

    return result/73

x_pred = model.predict(x_test[0:2])
y_pred = result_trans(x_pred)
for i in range(len(y_pred)):
    print(duibi_result(y_pred[i],y_test[0:2][i]))

print(model.predict(x_test[0:2]))
print(y_test[0:2])

# 绘制训练 & 验证的准确率值
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model accuracy and loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Epoch')
plt.legend(['train_acc', 'test_acc','train_loss','test_loss'], loc='upper left')
plt.savefig('./test2.jpg')
plt.show()

# 绘制训练 & 验证的损失值
'''plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()'''

'''#plot_model(model, to_file='model.png')
acc = history.history['binary_accuracy']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)

plt.title('Accuracy and Loss')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, loss, 'blue', label='Validation loss')
plt.legend()
plt.show()'''

#查看权重
'''U=model.get_weights()[0] #输入层和循环层之间的权重，维度为（20*32）
W=model.get_weights()[1]  #循环层与循环层之间的权重，维度为（32*32）
bias=model.get_weights()[2] #隐藏层的偏置项，32个
print(U)
print(W)
print(bias)'''
