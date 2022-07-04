import csv
import numpy as np
import glob
import sys
import os

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



#csv.field_size_limit(500 * 1024 * 1024)###读取限制大小
csvfile_list=sorted(glob.glob('C:\\Users\\lcc\\Desktop\\data\\日本数据库\\*.txt'), key=os.path.getmtime)
#按照时间顺序
print('总共发现%s个CSV文件' %len(csvfile_list))
label_data= csv.reader(open('C:\\Users\\lcc\\Desktop\\data\\日本数据库\\data_label.csv'))

length = []
datas = []
for csvfile,label in zip(csvfile_list,label_data):
    print(csvfile,label)

    #读取每个文件原始数据
    f_data = csv.reader(open(csvfile))
    duquzhi_x = []
    duquzhi_y = []
    for i in f_data:
        a =i[0].strip().split()
        duquzhi_x.append(float(a[1]))
        duquzhi_y.append(float(a[2]))
    #归一化数据
    y_min = min(duquzhi_y)
    y_max = max(duquzhi_y)
    for j in range(len(duquzhi_y)):
        if y_max==y_min:
            duquzhi_y[j] = round(5,4)
        else:
            duquzhi_y[j] = round(((duquzhi_y[j] - y_min) * 4 / (y_max - y_min))+1,4)
    #print(min(duquzhi_y),max(duquzhi_y))

    for k in range(len(duquzhi_x)):
        data = ' '.join([str(duquzhi_x[k]), str(duquzhi_y[k])])
        datas.append(data)

    duquzhi_x.clear()
    duquzhi_y.clear()

    print('数据长度是：',len(datas))
    print('数据：',datas)
    length.append(len(datas))
    file = open('training_set.csv', 'a+', encoding='utf-8', newline='')
    write_csv(file,datas)
    datas.clear()

##查看预测数据情况
print('预测数据中最大长度是',max(length))
print('预测数据中长度大于150的有(个):',sum(i > 150 for i in length))