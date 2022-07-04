# -*- coding: utf-8 -*-
#@Author  : lynch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense,Dropout,LSTM,Bidirectional


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


def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


############################################
'''#数据读取
x,y = dataing.data_make()##不定长特征
#x,y = dataing.data_make_samelen()##定长特征

#数据预处理
x_set = np.array(x,dtype=object)
train_set = []
for i in range(len(x)):
    train_set.append([x[i],y[i]])
random.shuffle(train_set)###数据重排
x_set1 = [e[0] for e in train_set]#特征数据
y_set1 = [f[1] for f in train_set]#标签

##训练数据最大长度,可以在dataing程序中查看，目前是268
max_length = 268
##keras.preprocessing.sequence.pad_sequences将多个序列截断或补齐为相同长度，返回numpy数组
x_set2 = keras.preprocessing.sequence.pad_sequences(x_set1, maxlen=max_length, dtype='float64',padding='post',value = [0.0,0.0])
y_set2 = np.array(y_set1)
#print(x_set2.shape)

##测试集训练集划分
x_train = x_set2[0:450]
y_train = y_set2[0:450,0]
print(y_train.shape)
print(x_train.shape)
x_test = x_set2[450:517]
y_test = y_set2[450:517,0]
#print(y_test[1])
#print(x_test[1])'''


################################################
#数据重采样技术，单个标签在原模块中
import 查看数据平衡
x,y= 查看数据平衡.oversamp()
x_set = np.array(x,dtype=object)

#数据打乱处理
train_set = []
for i in range(len(x)):
    train_set.append([x[i],y[i]])

print(train_set[0])
random.shuffle(train_set)###数据重排
x_set1 = [e[0] for e in train_set]#特征数据
y_set1 = [f[1] for f in train_set]#标签

##训练集测试集划分
x_set2 = np.array(x_set1)
y_set2 = np.array(y_set1)
x_train = x_set2[0:550]
y_train = y_set2[0:550]
x_test = x_set2[550:]
y_test = y_set2[550:]



model = keras.models.Sequential()
# 添加一个Masking层
model.add(layers.Masking(mask_value=0.0, input_shape=(2, 2)))
# 添加一个RNN层
rnn_layer =layers.SimpleRNN(50, return_sequences=False)
model.add(rnn_layer)
#model.add(Bidirectional(LSTM(32)))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(30,activation='relu'))
#多个标签
#model.add(Dense(10, activation='sigmoid'))
#单个标签
model.add(Dense(1, activation='sigmoid'))

adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)#binary_focal_loss(gamma=2, alpha=0.25)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy']) ###,'top_k_categorical_accuracy'])#metrics.hanming_loss
model.summary()#####loss_function.multilabel_categorical_crossentropy
history = model.fit(x_train,y_train,batch_size=8, epochs=20,validation_data=(x_test,y_test))


print(model.predict(x_test[0:10]))
print(y_test[0:10])

'''# 绘制训练 & 验证的损失值和准确率（统一画图）
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model accuracy and loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Epoch')
plt.legend(['train_acc', 'test_acc','train_loss','test_loss'], loc='upper left')
plt.show()'''

# 绘制双图
# 绘制训练损失值和评估
plt.plot(history.history['loss'])
plt.ylabel('test_loss_Value')
plt.xlabel('Epoch')
plt.title('Model training')
plt.legend(['test_loss'],loc = 'upper left')
plt.savefig('./train_loss.jpg')
plt.show()


plt.plot(history.history['binary_accuracy'])
plt.ylabel('test_accuracy_Value')
plt.xlabel('Epoch')
plt.title('Model training')
plt.legend(['test_accuracy'],loc = 'upper left')
plt.savefig('./train_accuracy.jpg')
plt.show()

# 绘制测试损失值和评估
plt.plot(history.history['val_binary_accuracy'])
plt.ylabel('test_accuracy_Value')
plt.xlabel('Epoch')
plt.title('Model testing')
plt.legend(['test_accuracy'],loc = 'upper left')
plt.savefig('./test_accuracy.jpg')
plt.show()

plt.plot(history.history['val_loss'])
plt.ylabel('test_loss_Value')
plt.xlabel('Epoch')
plt.title('Model testing')
plt.legend(['test_loss'],loc = 'upper left')
plt.savefig('./test_loss.jpg')
plt.show()

# 绘制测试损失值和评估
'''plt.plot(history.history['val_loss'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('Model testing')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend(['test_loss', 'test_evaluate'], loc='upper left')
plt.show()'''

#绘制模型图
#plot_model(model, to_file='model.png')

#查看权重
'''U=model.get_weights()[0] #输入层和循环层之间的权重，维度为（20*32）
W=model.get_weights()[1]  #循环层与循环层之间的权重，维度为（32*32）
bias=model.get_weights()[2] #隐藏层的偏置项，32个
print(U)
print(W)
print(bias)'''

'''acc = history.history['binary_accuracy']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)

plt.title('Accuracy and Loss')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, loss, 'blue', label='Validation loss')
plt.legend()
plt.show()'''

