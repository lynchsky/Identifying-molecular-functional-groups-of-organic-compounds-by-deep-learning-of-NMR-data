# -*- coding: utf-8 -*-
#@Author  : lynch

import tensorflow as tf
import numpy as np
from sklearn import metrics
import tensorflow.keras.backend as K


def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    y_pred_neg = K.exp(y_pred_neg)+1
    y_pred_pos = K.exp(y_pred_pos)+1
    neg_loss = K.log(y_pred_neg)
    pos_loss = K.log(y_pred_pos)
    return neg_loss + pos_loss

'''y1 = [0,1,1,1,0]
x1 = [1,1,1,0,0]
y2 = [1,0,0,1,1]
x2 = [1,0,0,0,1]
y3 = [1,1,0,0,0]
x3 = [1,0,1,0,1]
y_t = np.array([y1,y2,y3])
y_p = np.array([x1,x2,x3])
print(y_t)
b = tf.constant(y_t)
c = tf.constant(y_p)

print(multilabel_categorical_crossentropy(b,c))'''


def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed