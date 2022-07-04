import csv
import numpy as np
import glob
import sys
import os

#数据规范化
a = np.linspace(-4,16,2001)
print(type(a))
b = a.tolist()

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
csvfile_list=sorted(glob.glob('C:\\Users\\lcc\\Desktop\\data\\nova预测\\NMRpicture_1_18\\*.csv'), key=os.path.getmtime)
#按照时间顺序
print('总共发现%s个CSV文件' % len(csvfile_list))

label_data= csv.reader(open('D:\\python\PycharmProjects\\article_data\\label_data.csv'))

datas = []
for csvfile,label in zip(csvfile_list,label_data):
    print(csvfile,label)
    #读取每个文件原始数据
    f_data = csv.reader(open(csvfile))
    duquzhi_x = []
    duquzhi_y = []
    for i in f_data:
        a = i[0].split('\t')
        duquzhi_x.append(float(a[0]))
        duquzhi_y.append(float(a[1]))
    #归一化数据
    y_min = min(duquzhi_y)
    y_max = max(duquzhi_y)
    for j in range(len(duquzhi_y)):
        duquzhi_y[j] = round((duquzhi_y[j] - y_min) * 5 / (y_max - y_min),4)
    print(min(duquzhi_y),max(duquzhi_y))
    # 处理原始数据，根据y值，提取峰值，并且直接转换为字符串形式便于保存为csv文件
    for i in range(len(duquzhi_y)-1):
        if duquzhi_y[i] != 0 and duquzhi_y[i]>duquzhi_y[i+1] and duquzhi_y[i]>0.2:
            if i == 0:##由于不等于0的判断条件，一般情况下，第一个数据就排除掉了，不会导致第一个i-1<0这种情况
                data = ' '.join([str(duquzhi_x[i]), str(duquzhi_y[i])])
                datas.append(data)
            elif duquzhi_y[i]>duquzhi_y[i-1]:
                data = ' '.join([str(duquzhi_x[i]), str(duquzhi_y[i])])
                datas.append(data)
    duquzhi_x.clear()
    duquzhi_y.clear()

    print(len(datas))
    file = open('training_set.csv', 'a+', encoding='utf-8', newline='')
    write_csv(file,datas)
    datas.clear()