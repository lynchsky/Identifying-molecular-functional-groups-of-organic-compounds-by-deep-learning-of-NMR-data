import csv
import numpy as np

f = csv.reader(open('C:\\Users\lcc\Desktop\\NMRpicture\\35_04_1.csv'))
datas = []
for i,j in zip(f,range(1,1000000)):
    #print(i[0])
    if j%16==0:
        a = i[0].split('\t')#
        #print(a)
        data = ' '.join(a)
        #print(len(data))
        datas.append(data)
        #print(type(data))

print(len(datas))

'''f = open('data.csv','w',encoding='utf-8',newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(('data','label'))
f.close()'''

#标签（-CH_3，-CH_2,-CH,-OH,-苯环，-CHO,-COOH,-NH_2,-NH,-SH）

label = [1,0,0,1,1,0,0,0,0,0]

def write_csv(f):
    csv_writer = csv.writer(f)
    csv_writer.writerow((datas,label))
    f.close()

file = open('data.csv','a+',encoding='utf-8',newline='')
write_csv(file)






