def same_length(data,max_length):
    '''
    :param data:
    :param max_length:
    :return:
    '''
    result = []
    data_ceshi = [[0, 0]]
    for i in data:
        #print(i)
        length = len(i)
        #print(length)
        add_length =  max_length - length
        result.append(i+data_ceshi*add_length)
        #print(result)

    return result

'''a = [[[1,2],[1,3]],[[11,2]]]
b= same_length((a),2)
print(b)'''