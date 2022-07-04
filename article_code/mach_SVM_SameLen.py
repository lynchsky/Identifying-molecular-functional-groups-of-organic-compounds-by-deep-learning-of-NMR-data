# -*- coding: utf-8 -*-
#@Author  : lynch
"""
功能：实现线性分类支持向量机
说明：可以用于二类分类，也可以用于多类分类
作者：唐天泽
博客：http://write.blog.csdn.net/mdeditor#!postId=76188190
日期：2017-08-09
"""

# 导入本项目所需要的包

import pandas as pd
import numpy as np
import dataing
import random
from sklearn import datasets
from sklearn import svm
from tensorflow import keras
# 使用交叉验证的方法，把数据集分为训练集合测试集
from sklearn.model_selection import train_test_split


#1.加载数据
x,y = dataing.data_make_samelen()##分别读取读取‘training_slen_set.csv’和‘training_slen_label.csv’文件，定长
train_set = []
for i in range(len(x)):
    train_set.append([x[i],y[i]])
random.shuffle(train_set)###数据重排

x_set= np.array([e[0] for e in train_set])#特征数据,shape=(517,2000)
y_set = np.array([f[1] for f in train_set])#标签,shape=(517,10)

#训练集
x_train = x_set[0:450]
y_train = y_set[0:450,9]
#测试集
x_test = x_set[451:]
y_test = y_set[451:517,9]



# 2.模型训练，使用LinearSVC考察线性分类SVM的预测能力
def test_LinearSVC(X_train,X_test,y_train,y_test):

    cls = svm.LinearSVC()
    cls.fit(X_train,y_train)
    print('Coefficients:%s, intercept %s'%(cls.coef_,cls.intercept_))
    print('Score: %.2f' % cls.score(X_test, y_test))

test_LinearSVC(x_train,x_test,y_train,y_test)