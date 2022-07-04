# -*- coding: utf-8 -*-
#@Author  : lynch
import numpy as np
a=[1,2,3]
b=[4,5]
c=[[[6,7],[8,9]],[[1,2],[2,1]]]
d=[[10,11],[12,13]]
print('在一维数组a后添加values,结果如下：{}'.format(np.append(a,b,axis=0)))
print('沿二维数组c的行方向添加values结果如下：'.format(np.append(c,d,axis=0)))
print('沿二维数组c的列方向添加values结果如下：'.format(np.append(c,d,axis=1)))
print('使用了axis，若arr和values的形状不同，则报错：'.format(np.append(a,c,axis=0)))



