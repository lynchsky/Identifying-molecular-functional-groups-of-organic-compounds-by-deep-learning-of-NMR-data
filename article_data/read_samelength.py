# -*- coding: utf-8 -*-
#@Author  : lynch

##加载模块
import numpy as np
import csv
import numpy as np
import glob
import sys
import os

#定义四个函数，1.预测15~-2(32768)；2.预测16~-4(32768);
# 3.xpf16~-4(65536且需要挪移)；4.xpf14~-2(65536需要挪移，且穿插在3数据中)
a = np.array(list(range(1600,-400,-1)))/100
normal_list = a.tolist()##[16.47 16.46 16.45 ... -3.98 -3.99 -4.0]

def write_csv(file,data):
    '''
    写入完整的训练数据集
    :param file: 要写入的文件名称
    :param data: 谱图数据
    ###:param label: 谱图标签,改进不要，仅存谱图数据
    '''
    csv_writer = csv.writer(file)
    csv_writer.writerow(data)
    file.close()


##读取yuce，按照时间顺序
csvfile_list=sorted(glob.glob('C:\\Users\\lcc\\Desktop\\data\\nova预测\\NMRpicture_1_18\\*.csv'), key=os.path.getmtime)
##读取xpf,按照时间顺序
#csvfile_list=sorted(glob.glob('C:\\Users\\lcc\\Desktop\\data\\xpfdata\\zhaoqs_20_cc\\*.csv'), key=os.path.getmtime)
##读取riben,按照时间顺序
#csvfile_list=sorted(glob.glob('C:\\Users\\lcc\\Desktop\\data\\日本数据库\\*.txt'), key=os.path.getmtime)

print('总共发现%s个CSV文件' % len(csvfile_list))

#函数1.预测15~-2(32768)；2.预测16~-4(32768)
def yuce_1_2(csvfile_list):
    for csvfile in csvfile_list:
        print(csvfile)
        # 读取每个文件原始数据
        f_data = csv.reader(open(csvfile))
        for first_value in f_data:
            loca_value = round(float(first_value[0].split('\t')[0]), 2)
            break

        # print(loca_value)
        loca_index = normal_list.index(loca_value)
        # print(loca_index)
        j = loca_index
        data = [0] * 2000
        for i in f_data:

            a = float(i[0].split('\t')[0])

            if a - normal_list[j] <= 0.0004:
                # print(normal_list[j])
                data[j] = float(i[0].split('\t')[1])
                j = j + 1
            if j > 1999:
                break
        # print(data)

        file = open('training_set.csv', 'a+', encoding='utf-8', newline='')
        write_csv(file, data)
        data.clear()

def xpf_3_4(csvfile_list):
    for csvfile in csvfile_list:
        print(csvfile)
        f_data = csv.reader(open(csvfile))
        for first_value in f_data:
            loca_value = round(float(first_value[0].split('\t')[0]),0)
            break

        loca_index = normal_list.index(loca_value)
        j = loca_index
        duquzhi = [0] * 2000
        for i in f_data:
            a = float(i[0].split('\t')[0])
            if a - normal_list[j] <= 0.0004:
                # print(normal_list[j])
                duquzhi[j] = float(i[0].split('\t')[1])
                j = j + 1
            if j > 1999:
                break

        # 归一化原始数据[0-5]
        y_min = min(duquzhi)
        y_max = max(duquzhi)
        for j in range(len(duquzhi)):
            duquzhi[j] = (duquzhi[j] - y_min) * 5 / (y_max - y_min)
        # 进一步处理数据，去掉杂点数据
        data = duquzhi.copy()
        for j in range(len(duquzhi)):
            if data[j] < 0.2:
                data[j] = 0


        file = open('training_set.csv', 'a+', encoding='utf-8', newline='')
        write_csv(file, data)
        data.clear()

def riben_5(csvfile_list):
    for csvfile in csvfile_list:
        print(csvfile)
        f_data = csv.reader(open(csvfile))
        duquzhi_x = []
        duquzhi_y = []
        for i in f_data:
            a = i[0].strip().split()
            print(a)
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

        data = [0] * 2000
        for k in range(len(duquzhi_y)):
            loca_value = round(duquzhi_x[k], 2)
            loca_index = normal_list.index(loca_value)
            data[loca_index] = duquzhi_y[k]

        file = open('training_set.csv', 'a+', encoding='utf-8', newline='')
        write_csv(file, data)
        data.clear()



##xpf_3_4(csvfile_list)
#riben_5(csvfile_list)
yuce_1_2(csvfile_list)