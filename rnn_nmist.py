# -*- coding: utf-8 -*-

# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
This code is a modified version of the code from this link:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
His code is a very good one for RNN beginners. Feel free to check it out.
"""
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class rnn:
    def __init__(self, batch_size, n_inputs, n_steps, n_hidden_units, n_classes):
        self.batch_size = batch_size
        self.n_inputs = n_inputs
        self.n_steps = n_steps
        self.n_hidden_units = n_hidden_units
        self.n_classes = n_classes

        # 记住，x和y的最后一维都是vector length
        # tf Graph input              batch#, step#,   input_vector_length
        self.inputs = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
        self.labels = tf.placeholder(tf.float32, [None, n_classes])

        self.logits = self._cell()
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.labels))

    def _cell(self):
        # Define weights
        self.weights = {
            # (28, 128)  这里的28是单个input vector的长度, hidden layer有128个neurons
            'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
            # (128, 10)  输出output vector长度为10
            'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
        }

        self.biases = {
            # (128, )
            'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
            # (10, )
            'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
        }

        # ----------------------------------------------------
        # hidden layer from input to cell 这里对应变量集U
        # ----------------------------------------------------
        # transpose the inputs shape to 2-Dimension
        # X ==> 128 batch * 28 steps, 28 input_vector_length
        X = tf.reshape(self.inputs, [-1, self.n_inputs])

        # into hidden
        # X_in = (128 batch * 28 steps, 128 hidden)
        X_in = tf.matmul(X, self.weights['in']) + self.biases['in']
        
        # 为什么要reshape成3-Dimension？ 因为后面RNN要将第二维的steps顺序feed in，而第一维batch中每个对应一个y
        # X_in ==> (128 batch, 28 steps, 128 hidden)
        X_in = tf.reshape(X_in, [-1, self.n_steps, self.n_hidden_units])

        # ----------------------------------------------------
        # basic LSTM Cell.
        # ----------------------------------------------------
        cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units)

        # lstm cell is divided into two parts (c_state, h_state)
        init_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        # You have 2 options for following step.
        # 1: tf.nn.rnn(cell, inputs);
        # 2: tf.nn.dynamic_rnn(cell, inputs).
        # If use option 1, you have to modified the shape of X_in, go and check out this:
        # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
        # In here, we go for option 2.
        # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
        # Make sure the time_major is changed accordingly.
        outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

        # hidden layer for output as the final results
        #############################################
        logits = tf.matmul(final_state[1], self.weights['out']) + self.biases['out']
        '''
        # # or    
        # unpack to list [(batch, outputs)..] * steps
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
        logits = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']    # shape = (128, 10)
        '''
        return logits

if __name__ == '__main__':
    batch_size = 128
    n_inputs = 28   # MNIST data input (img shape: 28*28)
    n_steps = 28    # time steps
    n_hidden_units = 128   # neurons in hidden layer
    n_classes = 10      # MNIST classes (0-9 digits)
    rnn1 = rnn(batch_size, n_inputs, n_steps, n_hidden_units, n_classes)

    # set random seed for comparing the two result calculations
    tf.set_random_seed(1)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # hyperparameters
    lr = 0.001
    training_iters = 100000

    train_op = tf.train.AdamOptimizer(lr).minimize(rnn1.cost)

    correct_pred = tf.equal(tf.argmax(rnn1.logits, 1), tf.argmax(rnn1.labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        step = 0
        while step * batch_size < training_iters:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
            feed_dict={
                rnn1.inputs: batch_xs,
                rnn1.labels: batch_ys,
            }
            sess.run([train_op], feed_dict=feed_dict)
            if step % 20 == 0:
                print(sess.run(accuracy, feed_dict=feed_dict))
                sys.stdout.flush()
            step += 1
