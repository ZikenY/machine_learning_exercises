import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
'''
https://www.leiphone.com/news/201705/zW49Eo8YfYu9K03J.html

    输入数据X：在时间t，Xt的值有50%的概率为1，50%的概率为0； 
    输出数据Y：在时间t，Yt的值有50%的概率为1，50%的概率为0，
                除此之外，如果`Xt-3 == 1`，Yt为1的概率增加50%，
                如果`Xt-8 == 1`，则Yt为1的概率减少25%， 
                如果上述两个条件同时满足，则Yt为1的概率为75%。

可知，Y与X有两个依赖关系，一个是t-3，一个是t-8。我们实验的目的就是检验RNN能否捕捉到Y与X之间的这两个依赖关系。
'''

'''
https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
'''

# Global config variables
num_steps = 5       # number of truncated backprop steps ('n' in the discussion above)
batch_size = 200
num_classes = 2
state_size = 4      # length of RNN's state
learning_rate = 0.1

# 生成实验数据
def gen_data(size = 100000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5

        #判断X[i-3]和X[i-8]是否为1，修改阈值
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25

        #生成随机数，以threshold为阈值给Yi赋值
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)

    return X, np.array(Y)

'''
    将生成的数据按照模型参数设置进行切分，
    这里需要用得到的参数主要包括：batch_size   - 批量数据大小
                             num_steps    - RNN每层rnn_cell循环的次数，
                                             也就是下图中Sn中n的大小
'''        
def gen_batch(raw_data, batch_size, num_steps):
    #raw_data是使用gen_data()函数生成的数据，分别是X和Y
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # 首先将数据切分成batch_size份，0-batch_size，batch_size-2*batch_size。。。
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype = np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype = np.int32)

    # data_x, data_y分成batch_size份 --> data_x[0 ~ batch_size-1]
    #                        data_y --> data_y[0 ~ batch_size-1]
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i : batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i : batch_partition_length * (i + 1)]

    # --- 注意这里的epoch_size和模型训练过程中的epoch不同 --- 
    # 因为RNN模型一次只处理num_steps个数据，
    # 所以将每个batch_size在进行切分成epoch_size份，每份num_steps个数据。
    epoch_size = batch_partition_length // num_steps

    # x是0-num_steps， batch_partition_length -batch_partition_length +num_steps。。。共batch_size个
    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield(x, y)
        '''
        yield 的作用就是把一个函数变成一个 generator，带有 yield 的函数不再是一个普通函数
        Python 解释器会将其视为一个 generator，调用 gen_batch() 不会执行函数，而是返回一个 iterable 对象！
        在 for 循环执行时，每次循环都会执行 fab 函数内部的代码，执行到 yield b 时，fab 函数就返回一个迭代值，
        下次迭代时，代码从 yield b 的下一条语句继续执行，而函数的本地变量看起来和上次中断执行前是完全一样的，于是函数继续执行，
        直到再次遇到 yield。

        一个带有 yield 的函数就是一个 generator，它和普通函数不同，
        生成一个 generator 看起来像函数调用，但不会执行任何函数代码，直到对其调用 next()（在 for 循环中会自动调用 next()）才开始执行。
        虽然执行流程仍按函数的流程执行，但每执行到一个 yield 语句就会中断，并返回一个迭代值，下次执行时从 yield 的下一个语句继续执行。
        看起来就好像一个函数在正常执行的过程中被 yield 中断了数次，每次中断都会通过 yield 返回当前的迭代值。
            https://www.ibm.com/developerworks/cn/opensource/os-cn-python-yield/            
        '''

# 这里的n就是训练过程中用的epoch，即在样本规模上循环的次数
# num_steps    - RNN每层rnn_cell循环的次数
def gen_epochs(n, num_steps):
    for i in range(n):
        # 将数据切分成batch_size份, batch_partition_length = data_length // batch_size
        # 将每个batch_size在进行切分成epoch_size份，每份num_steps个数据。
        # epoch_size = batch_partition_length // num_steps
        yield gen_batch(gen_data(), batch_size, num_steps)

if __name__ == '__main__':
    f = gen_epochs(2, 5)
    x, y = f.__next__().__next__()
    print('x.shape, y.shape:', np.array(x).shape, ', ', np.array(y).shape)
    x, y = f.__next__().__next__()
    print('x.shape, y.shape:', np.array(x).shape, ', ', np.array(y).shape)
#    x, y = f.__next__().__next__()  StopIteration









    
    