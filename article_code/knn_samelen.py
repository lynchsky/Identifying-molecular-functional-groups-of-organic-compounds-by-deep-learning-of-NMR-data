# -*- coding: utf-8 -*-
#@Author  : lynch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn import metrics
import tensorflow.keras.backend as K

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
import sklearn.metrics as sm

# 数据重采样技术，单个标签在原模块中
import 查看数据平衡

x, y = 查看数据平衡.oversamp_samelen()
x_set = np.array(x, dtype=object)

# 数据打乱处理
train_set = []
for i in range(len(x)):
    train_set.append([x[i], y[i]])
random.shuffle(train_set)  ###数据重排
x_set1 = [e[0] for e in train_set]  # 特征数据
y_set1 = [f[1] for f in train_set]  # 标签

################训练测试集划分#################
# 训练集改成一维
print(len(x_set1),len(y_set1))
print('same len')
x_set2 = np.array(x_set1)
y_set2=np.array(y_set1)
x_train = x_set2[0:740]
y_train = y_set2[0:740]
x_test = x_set2[740:]
y_test = y_set2[740:]

## 训练并预测，其中选取k=2
clf = neighbors.KNeighborsClassifier(2, 'distance')
clf.fit(x_train, y_train)
Z = clf.predict(x_test)
print('准确率:', clf.score(x_test, y_test))
y_pred = clf.predict(x_test)
f1_score = sm.f1_score(y_test, y_pred)
print('f1_score', f1_score)
pre_score = sm.precision_score(y_test, y_pred)
print('precision', pre_score)
reca_score = sm.recall_score(y_test, y_pred)
print('recall', reca_score)
con_matrix = sm.confusion_matrix(y_test, y_pred)
print('con_matrix', con_matrix)