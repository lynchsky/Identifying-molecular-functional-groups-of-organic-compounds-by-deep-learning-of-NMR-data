# -*- coding: utf-8 -*-
#@Author  : lynch
from tensorflow.keras import backend as K
from tensorflow import keras

##测试函数部分
a = K.constant(0.6)
b =K.constant(1)
print(K.clip(a,0,1))
#print(keras.metrics.binary_accuracy(b,a))






def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    '''
    1.K.epsilon
    用法：返回数值表达式中使用的模糊因子的值。返回：一个浮点数。
    keras.backend.epsilon() , 输出结果：1e-07
    '''

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



