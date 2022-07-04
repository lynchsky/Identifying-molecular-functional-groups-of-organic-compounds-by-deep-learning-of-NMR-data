# -*- coding: utf-8 -*-
#@Author  : lynch


from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Dropout

def create_model():
    model = keras.models.Sequential()
    # 添加一个Masking层
    model.add(layers.Masking(mask_value=0.0, input_shape=(2, 2)))
    # 添加一个普通RNN层
    rnn_layer = layers.SimpleRNN(50, return_sequences=False)
    model.add(rnn_layer)
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='relu'))
    # 多个标签
    # model.add(Dense(10, activation='sigmoid'))
    # 单个标签
    model.add(Dense(10, activation='sigmoid'))

    adam = keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_accuracy'])
    return model


# Create a basic model instance
model = create_model()
plot_model(model, to_file='model.png',show_shapes=True)