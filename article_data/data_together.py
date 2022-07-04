# -*- coding: utf-8 -*-
#@Author  : lynch
import csv

f1 = csv.reader(open('D:\\python\\PycharmProjects\\article\\training_slen_set.csv'))
f2 = csv.reader(open('D:\\python\\PycharmProjects\\article\\training_slen_label.csv'))


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



for i,j in zip(f1,f2):
    print(j)
    print(type(j))
    print(j[0])
    print(type(j[0]))
    a = ' '.join(i)
    b = ' '.join(j)
    data = i+j
    file = open('training_slen_together.csv', 'a+', encoding='utf-8', newline='')
    write_csv(file,data)
