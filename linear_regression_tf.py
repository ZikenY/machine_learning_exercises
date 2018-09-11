'''
http://www.jianshu.com/p/6766fbcd43b9
    将计算流程表示成图；
    通过Sessions来执行图计算；
    将数据表示为tensors；
    使用Variables来保持状态信息；
    分别使用feeds和fetches来填充数据和抓取任意的操作结果；
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
'''
        sudo apt-get remove python3-matplotlib
        sudo pip install -U matplotlib
'''
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline

def fit1d():
    # 模拟生成100对数据对, 对应的函数为y = x * 0.1 * (1 + noise1) + 0.3 + noise0
    x_data = np.random.rand(100).astype("float32")
    noise0 = (np.random.rand(100) - 0.5) * 0.02
    noise1 = (np.random.rand(100) - 0.5) * 0.02
    labels = x_data * 0.1 * (1 + noise1) + 0.3 + noise0
    
    # 指定w和b变量的取值范围（注意我们要利用TensorFlow来得到w和b的值）
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    
    # y is the prediction
    y = W * x_data + b
    
    # 最小化均方误差
    loss = tf.reduce_mean(tf.squared_difference(y, labels))
    # loss = tf.reduce_mean(tf.square(y - labels))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    # the goal is to minimize the loss by using gradient descent
    train = optimizer.minimize(loss)
    
    # 初始化TensorFlow参数
    init = tf.global_variables_initializer()
    
    # 运行数据流图（注意在这一步才开始执行计算过程）
    sess = tf.Session()
    sess.run(init)
    
    # 观察多次迭代计算时，w和b的拟合值
    for step in range(0, 201):
        sess.run(train)
#         if step % 20 == 0:
#             print(step, sess.run(W), sess.run(b), end='')
#             print(', loss:', sess.run(loss))

    # draw all samples  
    for i in range(100):
        plt.plot(x_data[i], labels[i], 'ob')

    # draw the classify line  
    min_x = min(x_data)
    max_x = max(x_data)
    min_y = float(sess.run(W) * min_x) + sess.run(b)
    max_y = float(sess.run(W) * max_x) + sess.run(b)
    plt.plot([min_x, max_x], [min_y, max_y], '-g')  
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('linear regression')
    plt.legend()
    plt.show()

def fit2d():
    # 用 NumPy 随机生成 100 个数据
    # input sample#: 100, each sample:(x0, x1)
    # [2x100]
    x_data = np.float32(np.random.rand(2, 100)*10) 

    # labels#: 100
    # [1x2] * [2x100] = [1x100], 0.3 expands to 1x100
    #                 [0.1, 0.2] is also ok  -   [2,] can be treated to [1,2]
    y_labels = np.dot([[0.1, 0.2]], x_data) + 0.3
    y_labels += (np.random.rand(100) - 0.5) * 0.02 # noise

    #  *ideal_W = [[0.1, 0.2]], ideal_b = [[0.3]]
    # we need to train our W, b to as close as [[0.1, 0.2]], [[0.3]]
    
    # 构造一个线性模型
    # b[1x1]
    b = tf.Variable(tf.random_uniform([1,1]))
    # W[1x2]
    W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
    # 拼装 W, b, x, y;  expand b to 1x100
    y = tf.matmul(W, x_data) + b

    # 最小化方差
 #                        tf.square(y - y_labels)) will do the same
    loss = tf.reduce_mean(tf.squared_difference(y, y_labels))

    # use gradient descent to minimize the loss
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # initialize the graph
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # train the graph
    for i in range(0, 3291):
        sess.run(train)
#         if i % 20 == 0:
#             print('iteration#:', i,
#               'W[0]:', sess.run(W)[0][0],  # W: [[W0, W1]]
#               'W[1]', sess.run(W)[0][1],
#               ', loss:', sess.run(b)) 

    print('W: ', sess.run(W)[0][0], sess.run(W)[0][1], 'b: ', sess.run(b))
    
    # prepare the x, y labels
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')

    scatter_x = np.matrix.transpose(np.matrix(x_data, dtype = float)).A
    scatter_y = np.matrix.transpose(np.matrix(y_labels, dtype = float)).A    

    x1 = scatter_x[:, 0]
    x2 = scatter_x[:, 1]
    ax.scatter(x1, x2, scatter_y, c = 'g', marker = 'o', label = 'samples')
    
    # how to draw a plane ?
    X1 = np.arange(0, 10) # np.arange(min(scatter_x[0]), max(scatter_x[0]))
    X2 = np.arange(0, 10) # np.arange(min(scatter_x[1]), max(scatter_x[1]))

    X1, X2 = np.meshgrid(X1, X2)
    Y = sess.run(W)[0][0] * X1 + sess.run(W)[0][1] * X2 + sess.run(b)
    ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap='rainbow')

    plt.legend()
    plt.show

fit1d()
fit2d()

