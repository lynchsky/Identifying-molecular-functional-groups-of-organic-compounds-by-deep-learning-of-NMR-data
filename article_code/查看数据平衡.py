import dataing #读取数据
import numpy as np
from collections import Counter  #collecton模块
from tensorflow import keras

################过采样方法##############
def oversamp():
    x, y = dataing.data_make()  ##返回的都是数值类型
    #print(x[317],y[317])
    x_set = np.array(x, dtype=object)  ##由于是不定长数据，需要带上dtype
    y_set = np.array(y)
    '''print(x_set[0])
    print(x_set.shape)  # （516，）
    print(y_set.shape)  # （516，10）'''

    max_length = 193
    X = keras.preprocessing.sequence.pad_sequences(x_set, maxlen=max_length, dtype='float64', padding='post',
                                                   value=[0.0, 0.0]).tolist()

    x = []
    for i in X:
        k = [n for a in i for n in a]
        x.append(k)
    #print(x[0])


    y = y_set[:, 0].tolist()###设置单标签
    #print('y_set',len(y_set))
    print(Counter(y))
    ##平衡数据：0第一类{1: 330, 0: 187}，1第二类{1: 393, 0: 124}，2第三类{1: 347, 0: 170}
    ########： 3第四类{0: 430, 1: 87}，4第五类{1: 316, 0: 201}，5第六类{0: 482, 1: 35}
    ########： 6第七类{0: 444, 1: 73}，7第八类{0: 432, 1: 85}，8第九类{0: 422, 1: 95}，9第十类{0: 502, 1: 15}


    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=10)
    x_resampled, y_resampled = ros.fit_resample(x, y)
    print(sorted(Counter(y_resampled).items()))
    #print(x_resampled[0])
    #print(len(y_resampled))
    result = []
    for i in x_resampled:
        b = np.array(i).reshape(193, 2).tolist() # reshape(列的长度，行的长度)
        result.append(b)

    #print(type(result),type(y_resampled))##返回的都是等长的列表类型
    return result,y_resampled

def oversamp_samelen():
    x, y = dataing.data_make_samelen()  ##返回的都是数值类型
    #print(x[317],y[317])
    x_set = np.array(x, dtype=object)  ##由于是不定长数据，需要带上dtype
    y_set = np.array(y)
    '''print(x_set[0])
    print(x_set.shape)  # （516，）
    print(y_set.shape)  # （516，10）'''


    x_data = x_set.tolist()
    y_data = y_set[:, 6].tolist()###设置单标签
    #print('y_set',len(y_set))
    print(Counter(y_data))
    ##平衡数据：0第一类{1: 330, 0: 187}，1第二类{1: 393, 0: 124}，2第三类{1: 347, 0: 170}
    ########： 3第四类{0: 430, 1: 87}，4第五类{1: 316, 0: 201}，5第六类{0: 482, 1: 35}
    ########： 6第七类{0: 444, 1: 73}，7第八类{0: 432, 1: 85}，8第九类{0: 422, 1: 95}，9第十类{0: 502, 1: 15}


    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=10)
    x_resampled, y_resampled = ros.fit_resample(x_data, y_data)
    print(sorted(Counter(y_resampled).items()))
    #print(x_resampled[0])
    #print(len(y_resampled))


    #print(type(result),type(y_resampled))##返回的都是等长的列表类型
    return x_resampled,y_resampled

'''x,y=oversamp_samelen()
print(type(x),type(y))
print(x[517],y[517])
#print(x)
#print(y)
print(x[317],y[317])
print(type(x))
print(type(x[0]))'''