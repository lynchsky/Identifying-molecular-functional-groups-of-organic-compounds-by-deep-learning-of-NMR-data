import numpy as np

hidden_size = 50 #隐藏层神经元个数
vocab_size = 4 #词汇表大小

def init_orthogoanl(param):
    '''正交初始化'''
    if param.ndim < 2:
        raise ValueError('参数维度必须大于2')
    rows,cols =param.shape
    new_param = np.random.randn(rows,cols)
    if rows<cols:
        new_param =new_param.T

    #QR矩阵分解，q为正交矩阵，r为上三角矩阵
    q,r=np.linalg.qr(new_param)
    #让q均匀分布，https://arxiv.org/pdf/math-ph/0609050.pdf
    d = np.diag(r,0)
    ph = np.sign(d)
    q *= ph
    if rows<cols:
        q = q.T
    new_param =q
    return new_param

def init_rnn(hidden_size,vocab_size):
    '''
    初始化循环神经网络参数
    Args:
    hidden_size: #隐藏神经元个数
    vocab_size: #词汇表大小
    '''
    #输入到隐藏层权重矩阵
    U = np.zeros((hidden_size,vocab_size))
    #隐藏层到隐藏层权重矩阵
    V = np.zeros((hidden_size,hidden_size))
    #隐藏层到输出层权重矩阵
    W = np.zeros((vocab_size,hidden_size))
    #隐层bias
    b_hidden = np.zeros((hidden_size,1))
    #输出层bias
    b_out = np.zeros((vocab_size,1))
    #权重初始化
    U = init_orthogoanl(U)
    V = init_orthogoanl(V)
    W = init_orthogoanl(W)

    return U,V,W,b_hidden,b_out


##激活函数
def sigmoid(x,derivative = False):
    '''
    Args:
    :param x:输入数据
    :param derivative:如果为true则计算梯度
    :return:
    '''
    x_safe = x + 1e-12
    f = 1/(1+np.exp(-x_safe))
    if derivative:
        return f*(1-f)
    else:
        return f

def tanh(x,derivative = False):
    x_safe = x +1e-12
    f = (np.exp(x_safe)-np.exp(-x_safe))/(np.exp(x_safe)+np.exp(-x_safe))
    if derivative:
        return 1-f**2
    else:
        return f

def softmax(x,derivative = False):
    x_safe = x + 1e-12
    f = np.exp(x_safe)/np.sum(np.exp(x_safe))

    if derivative:
        pass#本文不加算softmax函数导数
    else:
        return f

##RNN前向传播
def forward_pass(inputs ,hidden_state,params):
    '''
    RNN前向传播计算
    :param inputs:输入序列
    :param hidden_state: 初始化后的隐藏状态参数
    :param params: RNN参数
    :return:
    '''
    U,V,W,b_hidden,b_out = params
    outputs,hidden_states = [],[]

    for t in range(len(inputs)):
        #计算隐藏状态
        hidden_state = tanh(np.dot(U,inputs[t])+np.dot(V,hidden_state)+b_hidden)
        #计算输出
        out = softmax(np.dot(W,hidden_state)+b_out)
        #保存中间结果
        outputs.aappend(out)
        hidden_state.append(hidden_state.copy())
    return outputs ,hidden_states

##RNN后向传播
def clip_gradient_norm(grads,max_norm=0.25):
    '''
    梯度剪裁防止梯度爆炸
    :param grads:
    :param max_norm:
    :return:
    '''
    max_norm = float(max_norm)
    total_norm = 0
    for grad in grads:
        grad_norm = np.sum(np.power(grad,2))
        total_norm = np.sqrt(total_norm)
        clip_coef = max_norm/(total_norm+1e-6)
        if clip_coef<1:
            for grad in grads:
                grad *= clip_coef
        return grads
def backward_pass(inputs,outputs,hidden_states,targets,params):
    '''
    #后向传播
    :param inputs:序列输入
    :param outputs: 输出
    :param hidden_states:隐藏状态
    :param targets: 预测目标
    :param params: RNN参数
    :return:
    '''
    U,V,W,b_hidden,b_out = params
    d_U,d_V,d_W = np.zeros_like(U),np.zeros_like(V),np.zeros_like(W)
    d_b_hidden ,d_b_out = np.zeros_like(b_hidden),np.zeros_like(b_out)
    ##跟踪隐藏层偏导及损失
    d_h_next = np.zeros_like(hidden_states[0])
    loss = 0
    #对于输出学列当中每一个元素，反向遍历s.t. t= N,N-1,...,1,0
    for t in reversed((range(len(outputs)))):
        #交叉熵损失
        loss += -np.mean(np.log(outputs[t]+1e-12)*targets[t])
        d_o = outputs[t].copy()
        d_o[np.argmax(targets[t])] -= 1
        d_W += np.dot(d_o,hidden_states[t].T)
        d_b_out += d_o
        d_h = np.dot(W.T,d_o) + d_h_next

        d_f = tanh(hidden_states[t],derivative=True) *d_h
        d_b_hidden += d_f

        d_U += np.dot(d_f,inputs[t].T)

        d_V += np.dot(d_f,hidden_states[t-1].T)
        d_h_next = np.dot(V.T,d_f)
    grads = d_U,d_V,d_W,d_b_hidden,d_b_out
    #梯度裁剪
    grads = clip_gradient_norm(grads)
    return loss,grads

##梯度优化
def update_paramters(params,grads,lr=1e-3):
    '''
    :param params:
    :param grads:
    :param lr: 学习率
    :return:
    '''
    for param,grad in zip(params,grads):
        param -= lr*grad
    return params
