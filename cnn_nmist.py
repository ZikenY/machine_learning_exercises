# -*- coding: utf-8 -*-

# 基本流程都是y' = Wx + b
#           minimize(loss(y, y'))
#           run(feed inputs to x, feed labels to y)

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# mnist.train.next_batch(batch_size) -> [x], [label]

# -----------------------------------------------------------------------------
# 权值(w)和偏置(b)的初始化过程中需要加入一小部分的噪声以破坏参数整体的对称性，同时避免梯度为0.
# 由于我们使用ReLU激活函数，所以我们通常将这些参数初始化为很小的正值：
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0,1, shape=shape)
    return initial
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 卷积操作使用滑动步长为1的窗口，使用0进行填充，所以输出规模和输入的一致；
# 池化操作是在2 * 2的窗口内采用最大池化技术(max-pooling)
def conv2d(x, W):
    # x: input image;  W: kernel which is weight
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
# -----------------------------------------------------------------------------


sess = tf.InteractiveSession()

# convolution layer1
# -----------------------------------------------------------------------------
# 卷积层1将要计算出32个feature map)，5 * 5的patch, 它的权值tensor的大小为[5, 5, 1, 32].
#                                前两维是patch的大小，第三维时输入通道的数目，最后一维是输出通道的数目
# kernel(patch) and bias for convolution layer-1: 
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# x is input images per batch  [BATCH_SIZE, 28x28]
x = tf.placeholder(tf.float32, [None, 784])

# 为了使得图片与计算层匹配，我们首先reshape输入图像x为4维的tensor，
#           第2、3维对应图片的宽和高，最后一维对应颜色通道的数目。
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 使用weight tensor对x_image进行卷积计算，加上bias
convolution1 = conv2d(x_image, W_conv1) + b_conv1

# use relu() as activation function
h_conv1 = tf.nn.relu(convolution1)
# -----------------------------------------------------------------------------

# pooling layer1, output: [-1,14,14,32]
h_pool1 = max_pool_2x2(h_conv1)

# convolution layer2
# -----------------------------------------------------------------------------
# 第二层将会有64个特征，对应每个5 * 5的patch
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
convolution1 = conv2d(h_pool1, W_conv2) + b_conv2
h_conv2 = tf.nn.relu(convolution1)
# -----------------------------------------------------------------------------

# pooling layer1, output: [-1,7,7,64]
h_pool2 = max_pool_2x2(h_conv2)

# full connection, output: [-1, 1024]
# -----------------------------------------------------------------------------
# 到目前为止，图像的尺寸被缩减为7x7，我们最后加入一个神经元数目为1024的全连接层来处理所有的图像上
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# 将最后的pooling层的输出reshape为一个一维向量，与权值相乘，加上偏置
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
full_connection = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
h_fc1 = tf.nn.relu(full_connection)
# -----------------------------------------------------------------------------

# dropout, 在训练过程中使用，而在测试过程中不使用
# output: [-1, 1024]
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer: softmax, output: [-1, 10]
# -----------------------------------------------------------------------------
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
output = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv = tf.nn.softmax(output)
# -----------------------------------------------------------------------------

# we will compare y_ and y_conv. that is loss function
y_ = tf.placeholder(tf.float32, [None, 10])

# loss function:
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv), name = "loss_function")

optimizer = tf.train.AdamOptimizer(1e-4)
# optimizer = tf.train.GradientDescentOptimizer(1e-5)
train_step = optimizer.minimize(cross_entropy)

# compute training accuracy
correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess.run(init)

for i in range(20000):
    # batch size
    batch = mnist.train.next_batch(50)

    #--------- output the accuracy during training progress ------------
    if i % 500 == 0:
        #                                                                  在测试过程中不使用dropout
        train_accuacy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuacy))
    #-------------------------------------------------------------------

    train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})

# accuacy on test
print("test accuracy %g"%(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))









