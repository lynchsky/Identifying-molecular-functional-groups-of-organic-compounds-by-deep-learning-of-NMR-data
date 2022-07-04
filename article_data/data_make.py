import csv
import numpy as np
import glob
import sys
import os

#csv.field_size_limit(500 * 1024 * 1024)###读取限制大小
csvfile_list=sorted(glob.glob('C:\\Users\lcc\Desktop\\NMRpicture\\*.csv'), key=os.path.getmtime)
#按照时间顺序
print('总共发现%s个CSV文件' % len(csvfile_list))

def write_csv(file,data,label):
    '''
    写入完整的训练数据集
    :param file: 要写入的文件名称
    :param data: 谱图数据
    :param label: 谱图标签
    '''
    csv_writer = csv.writer(file)
    csv_writer.writerow((data, label))
    file.close()

##新建csv文件，第一次运行时打开
'''f = open('training_set','w',encoding='utf-8',newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(('data','label'))
f.close()'''
##新建csv_label文件，第一次运行时打开
'''f = open('label_data','w',encoding='utf-8',newline='')
csv_writer = csv.writer(f)
csv_writer.writerow((''))
f.close()'''

label_data= csv.reader(open('D:\\python\\PycharmProjects\\article_data\\label_data.csv'))


datas = []
for csvfile,label in zip(csvfile_list,label_data):
    print(csvfile,label)

    f_data = csv.reader(open(csvfile))
    for i, j in zip(f_data, range(1, 1000000)):
        # print(i[0])
        if j % 2 == 0:
            a = i[0].split('\t')
            # print(a)
            data = ' '.join(a)
            # print(len(data))
            datas.append(data)
            # print(type(data))
    # 标签（-CH_3，-CH_2,-CH,-OH,-苯环，-CHO,-COOH,-NH_2,-NH,-SH）
    file = open('training_set.csv', 'a+', encoding='utf-8', newline='')
    write_csv(file,datas,label)
    datas.clear()



