# -*- coding: utf-8 -*-

# 基本流程都是y' = Wx + b
#           minimize(loss(y, y'))
#           run(feed inputs to x, feed labels to y)

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 100
lamda = 50.0 # λ

# the number of images for each BATCH can be arbitrary
x = tf.placeholder(shape=[None, 784], dtype = tf.float32, name='x')
# the image# must match x
# each image has a one_hot vector[10]
labels = tf.placeholder(shape=[None, 10], dtype = tf.float32, name='labels')

# each sample has 784 pixels. it can be one of 10 classes
W = tf.Variable(tf.truncated_normal([784, 10]), name='W')
# 10 classes in total; 10 neurons in the 2nd layer
b = tf.Variable(tf.truncated_normal([10]), name='b')

# prediction y has 10 probabilities
# b will expand to [?, 10], ? depends on x(placeholder)
y_ = tf.nn.softmax(tf.matmul(x, W) + b)

# loss w/ L2 regularization
tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
regularizer = tf.contrib.layers.l2_regularizer(scale = lamda / (2 * batch_size))
reg_term = tf.contrib.layers.apply_regularization(regularizer)
# loss func
cross_entropy = -tf.reduce_sum(labels * tf.log(y_), name='cross_entropy')
loss = cross_entropy + reg_term

# the goal is using gradient descent to minimize the loss function
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)

sess = tf.Session()

# init
init = tf.global_variables_initializer()
sess.run(init)

#train
for i in range(1000):
    batch_xs, batch_labels = mnist.train.next_batch(batch_size)
    # feed the samples and the labels                      batch_xs:     [batch_size, 784]
    #                                                      batch_labels: [batch_size, 10]
    _, l = sess.run([train_step, cross_entropy], feed_dict={x:batch_xs, labels:batch_labels})
    if i % 100 == 0:
        print('iteration', i, ': loss:', l, end='  ')
        
        # ============ 模型的评价 =============
        # a label: one-hot-vectors[10]; labels: [55000x10]
        # 这里实际上构建了一个小graph, 以y和labels作为输入, 计算equality
        # 而y从哪里来呢?, 用mnist.test.images作为输入算出来 (这里实际上不需要labels to feed placeholder)
        #
        # 利用tf.argmax()函数来得到预测和实际的图片label值, 再用一个tf.equal()函数来判断预测值和真实值是否一致
        # labels每一行是一个one-hot[10], 所以用axis=1取出一行中最大的(唯一的1)
        # y的每一行是softmax输出, 最大值对应predicted的那一类
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(labels, 1))
        # cast correct_prediction to float, and compute the mean of the differeces among all the images
        cast_eval = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
        # feed x for computing y. labels in here is no use
        accuracy = sess.run(cast_eval, feed_dict={x: mnist.test.images, labels: mnist.test.labels})
        print("accuracy:", accuracy)


correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(labels, 1))
cast_eval = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
accuracy = sess.run(cast_eval, feed_dict={x: mnist.test.images, labels: mnist.test.labels})
print("final accuracy:", accuracy)

