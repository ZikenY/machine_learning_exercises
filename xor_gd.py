# -*- coding: utf-8 -*-
"""
@author: ziken
"""

import numpy as np
import tensorflow as tf

def main():
    
    # train_dataset: 4x[1x2]   - 2 features, 4 samples
    train_dataset = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]], dtype=np.float32)
    # labels: 4x[1x1],
    train_labels = np.array([[[0]], [[1]], [[1]], [[0]]], dtype=np.float32)
    
    print('\ntrain_dataset定义成4个(1x2)')
    print('train_dataset.shape: ', train_dataset.shape)
    print('train_labels.shape: ', train_labels.shape, '\n')

    model = tf.Graph()
    with model.as_default():

        # 为什么要定义input为1x2？ 因为便于train_dataset定义成4个(1x2)。
        input_1x2 = tf.placeholder(shape=[1,2], dtype=tf.float32, name='input_1x2')
        label_1x1 = tf.placeholder(shape=[1,1], dtype=tf.float32, name='input_1x1')
        
        # read input [2x1]    <<<  注意乘法的顺序： W.t * input.t  >>>
        input_2x1 = tf.transpose(input_1x2)

        # init W,c,w,b
        init_Weight1    = tf.truncated_normal([2, 2])
        init_bias_c     = tf.truncated_normal([2, 1])      # = tf.zeros([2, 1])
        init_weight2    = tf.truncated_normal([2, 1])
        init_bias_b     = tf.truncated_normal([1])

        # these are things needed to be trained
        W_layer1        = tf.Variable(init_Weight1, name = "weight_layer1")
        bias_layer1     = tf.Variable(init_bias_c,  name = "bias_layer1")
        w_layer2        = tf.Variable(init_weight2, name = "weight_layer2")
        bias_layer2     = tf.Variable(init_bias_b,  name = "bias_layer2")



        # ====   combine activation functions for layer1   ====
        # from input_2x1, W_layer1_T(2x2), bias_layer1 to relu()
        #
        # [2x2]
        W_layer1_T = tf.transpose(W_layer1)
        
        # [2x2] * [2x1] + [2x1]
        net_in_layer1 = tf.matmul(W_layer1_T, input_2x1) + bias_layer1

        # relu activation function for layer1
        activation_h1 = tf.nn.relu(net_in_layer1)
        # output size: [2x1]
        

        # ====   layer2   ====
        # from activation_h1(2x1), w_layer2_T(1x2), bias_layer2 to net_in_layer2
        #
        # [1x2]
        w_layer2_T = tf.transpose(w_layer2)
        
        # [1x2] * [2x1] + [1x1]
        net_in_layer2 = tf.matmul(w_layer2_T, activation_h1) + bias_layer2
        # output size: [1x1] 


        # 计算sqr(x-y); compute the differences between y' and labels
        # 在每个维度上计算，但这里实际上只是个数值
        sqr_diff = tf.squared_difference(net_in_layer2, label_1x1, name = "sqr_diff")
#        sqr_diff = tf.square(net_in_layer2 - labels_1x1)



        # compute the mean of the differences
        # .reduce_mean()参数1--input_tensor:待求值的tensor  参数2--reduction_indices:在哪一维上求解
        # 如果不指定第二个参数，那么就在所有的元素中取平均值
        loss = tf.reduce_mean(sqr_diff)
#        loss = sqr_diff    # 注意这里的sqr_diff实际上只有1x1


        #  加入学习率衰减  ==> 套路
        init_rate = tf.constant(0.5, dtype = tf.float32)
        globalstep = tf.Variable(0, trainable = False)       # 用来记录当前是第几个batch
        decay_steps = tf.constant(4, dtype = tf.int32)      # update the rate in every 5 iterations
        decay_rate = tf.constant(0.97 + 0.03, dtype = tf.float32)  # decay coefficient
        learning_rate = tf.train.exponential_decay(init_rate, globalstep, decay_steps, decay_rate, staircase = True)
        
        # use gradient descent to minimize the loss
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        # goal  -->  用刚才定义的optimizer去minimize刚刚才才定义的loss函数
        train_step = optimizer.minimize(loss,  global_step = globalstep)

        '''
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, [0.2])
        train_step = optimizer.apply_gradients(zip(gradients, variables), global_step=globalstep)
        '''

    with tf.Session(graph = model) as sess:
        #tf.global_variables_initializer().run()
        init = tf.initialize_all_variables()        
        sess.run(init)
        
        epoch_count = 13;
        
        for epoch in range(0, epoch_count):
            print("Network (prediction) output=", end='') # 接下来将本次epoch 的4个输入值的输出打印在同一行
            
            for i in range(0, 4):
#                _, l, network_output = sess.run([train_step, loss, net_in_layer2], feed_dict={input_1x2:train_dataset[i], label_1x1:train_labels[i]})

                # model中共有两个输入：input_1x2和label_1x1, 
                # 将train_dataset[i]和train_labels[i]打包成dictionary喂给train_step
                feed_dict = {input_1x2:train_dataset[i], label_1x1:train_labels[i]}   # one of these: [0, 0]->0; [0, 1]->1; [1, 0]->1; [1, 1]->0
                
                # train_step is the goal we just defined
                # train the weights !!! 不信你去掉这一行试试？
                train_step.run(feed_dict)

                # just get the predicted network_output by the inputdata
                # 这一次目的是通过输入feed_dict来计算最终变量net_in_layer2的数值
                pred = sess.run(net_in_layer2, {input_1x2:train_dataset[i]})
                print(pred, ', ', end='')
            
            # 这一轮epoch结束，计算一下loss（也就是predict值与label之间的差值）. 这里直接用[0, 0]->0作为input
            # 看看上面loss的定义，就是the mean of the differences of layer2输出与label1x1
            l = sess.run(loss, feed_dict={input_1x2:train_dataset[0], label_1x1:train_labels[0]})
            print("\nloss at epoch %d is %f" % (epoch, l))
            
            # 记住sess.run()第1个参数是上面定义的你想要计算的变量，
            #               第2个参数就是{为了计算需要feed的输入变量:inputdata, ...}

main()