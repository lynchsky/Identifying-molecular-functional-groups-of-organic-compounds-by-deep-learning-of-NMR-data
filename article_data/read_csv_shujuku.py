import csv
import numpy as np
import matplotlib.pyplot as plt


f1 = csv.reader(open('C:\\Users\\lcc\\Desktop\\data\\日本数据库\\624-64-6.txt'))
duquzhi_x = []
duquzhi_y = []
x = []
j=0
for i in f1:
    print(type(i))
    #print(i[0])
    j += 1
    print(j)
    #print(type(i[0]))
    data=i[0].strip().split()
    print(data)
    duquzhi_x.append(float(data[1]))
    duquzhi_y.append(float(data[2]))
print(duquzhi_x,duquzhi_y)
print('数据长度是：',len(duquzhi_y))
#归一化原始数据[0-5]
y_min = min(duquzhi_y)
y_max = max(duquzhi_y)
for j in range(len(duquzhi_y)):
    if y_max == y_min:
        duquzhi_y[j] = round(1, 4)
    else:
        duquzhi_y[j] = round((duquzhi_y[j] - y_min) * 5 / (y_max - y_min), 4)

print(y_min)
print(y_max)
print(max(duquzhi_y))
#plt.xlim(0, 8)
plt.scatter(duquzhi_x,duquzhi_y,s=1)###原始数据画图，蓝色散点图
plt.show()

#第一次处理原始数据，根据y值，提取峰值
'''x_1 = []
y_1 = []
datas = []
for i in range(len(duquzhi_y)-1):
    if duquzhi_y[i] != 0 and duquzhi_y[i]>duquzhi_y[i+1]:
        if i == 0:##由于不等于0的判断条件，一般情况下，第一个数据就排除掉了，不会导致第一个i-1<0这种情况
            x_1.append(duquzhi_x[i])
            y_1.append(duquzhi_y[i])
            data = [duquzhi_x[i],duquzhi_y[i]]
        elif duquzhi_y[i]>duquzhi_y[i-1]:
            x_1.append(duquzhi_x[i])
            y_1.append(duquzhi_y[i])
            data = ' '.join([str(duquzhi_x[i]), str(duquzhi_y[i])])
            datas.append(data)'''


'''#del y_1[0]
print(datas)
print(type(datas))
#print(datas[0])
#print(type(datas[0]))
print(len(x_1))
#print(x_1)
#print(y_1)
plt.scatter(x_1,y_1,s=2,c='r')
plt.show()'''




'''#第二次处理数据，根据x值，去掉相邻很近但峰值比较低的数据
x_2 = []
y_2 = []
i = 0
while i < len(x_1):
    xvalue = x_1[i]
    yvalue = y_1[i]
    j = 1
    while i+j<len(x_1) and x_1[i+j]-xvalue<0.01:
        if y_1[i+j]>yvalue:
            yvalue = y_1[i+j]
            xvalue = x_1[i+j]
        j = j + 1
        x_2.append(xvalue)
        y_2.append(yvalue)
    i = i+j'''

