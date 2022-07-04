# -*- coding: utf-8 -*-
#@Author  : lynch
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#定义四个函数，1.预测15~-2(32768)；2.预测16~-4(32768);
# 3.xpf16~-4(65536且需要挪移)；4.xpf14~-2(65536需要挪移，且穿插在3数据中)
a = np.array(list(range(1600,-400,-1)))/100
normal_list = a.tolist()##[16.47 16.46 16.45 ... -3.98 -3.99]
print(normal_list)
print(len(normal_list))


f1 = csv.reader(open('C:\\Users\\lcc\\Desktop\\data\\xpfdata\\xujt\\3ab.csv'))
#f1 = csv.reader(open('C:\\Users\\lcc\\Desktop\\data\\nova预测\\NMRpicture_1_18\\90_05_1.csv'))
#f1 = csv.reader(open('C:\\Users\\lcc\\Desktop\\data\\日本数据库\\917_54_4.txt'))
'''for first_value in f1:
     loca_value = int(float(first_value[0].strip().split()[1]))
     break'''

'''print(loca_value)
loca_index = normal_list.index(loca_value)
print(loca_index)
j = loca_index'''
'''duquzhi = [0]*2000
duquzhi_x = []
duquzhi_y = []
for i in f1:
    a = i[0].strip().split()
    duquzhi_x.append(float(a[1]))
    duquzhi_y.append(float(a[2]))

# 归一化数据
y_min = min(duquzhi_y)
y_max = max(duquzhi_y)
for j in range(len(duquzhi_y)):
    if y_max == y_min:
        duquzhi_y[j] = round(5, 4)
    else:
        duquzhi_y[j] = round(((duquzhi_y[j] - y_min) * 4 / (y_max - y_min)) + 1, 4)

data = [0]*2000
for k in range(len(duquzhi_y)):
    loca_value = round(duquzhi_x[k],2)
    loca_index = normal_list.index(loca_value)
    data[loca_index] = duquzhi_y[k]
    if a-normal_list[j] <= 0.0004 :
        #print(normal_list[j])
        duquzhi[j] = float(i[0].split('\t')[1])
        j = j + 1
    if j>1999:
        break


print(data)
#print('duquzhi',duquzhi)
#print('data',data)
#plt.scatter(normal_list,duquzhi,s=1)#原始数据画图，蓝色散点图
#plt.scatter(normal_list,data,s=2,c='r')
plt.plot(normal_list,data)
plt.show()'''

duquzhi_x = []
duquzhi_y = []
for i, j in zip(f1, range(1, 1000000)):
    data = i[0].split('\t')
    duquzhi_x.append(float(data[0]))
    duquzhi_y.append(float(data[1]))

print(len(duquzhi_y))
#归一化原始数据[0-5]
y_min = min(duquzhi_y)
y_max = max(duquzhi_y)
print(y_min,y_max)
duquzhi_y1 =duquzhi_y.copy()
for j in range(len(duquzhi_y)):
    duquzhi_y1[j] = round((duquzhi_y[j]-y_min)*5/(y_max-y_min),4)

print(min(duquzhi_y1))
#plt.xlim(0, 8)
plt.scatter(duquzhi_x,duquzhi_y1,s=1,alpha=0.5)###原始数据画图，蓝色散点图
#plt.show()
print(duquzhi_y)

loca_value = 0
for first_value in duquzhi_x:
    loca_value = round(first_value, 0)
    break
loca_index = normal_list.index(loca_value)
j = loca_index
duquzhi = [0] * 2000
for i in range(len(duquzhi_x)):
    if duquzhi_x[i] - normal_list[j] <= 0.0004:
        # print(normal_list[j])
        duquzhi[j] = duquzhi_y[i]
        j = j + 1
    if j > 1999:
        break
print(duquzhi)

# 归一化原始数据[0-5]
y_min = min(duquzhi)
y_max = max(duquzhi)
print(y_min,y_max)
for j in range(len(duquzhi)):
    duquzhi[j] = (duquzhi[j] - y_min) * 5 / (y_max - y_min)
# 进一步处理数据，去掉杂点数据
data = duquzhi.copy()
for j in range(len(duquzhi)):
    if data[j] < 0.2:
        data[j] = 0

plt.scatter(normal_list,data,s=15,c='r',marker='*')
plt.show()