# -*- coding: utf-8 -*-

import tensorflow as tf

# input image:  size= 3x3; channels#= 5
# kernel:       size= 1x1; channels#= 5
# feature map:  size= 3x3; channels#= 1
# --- kernel dot product with image's each pixel
input_image = tf.Variable(tf.random_normal([1,3,3,5]), dtype=tf.float32)
kernel = tf.Variable(tf.random_normal([1,1,5,1]), dtype=tf.float32)
convolute = tf.nn.conv2d(input_image, kernel, strides = [1,1,1,1], padding = 'SAME')
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    feature_maps = sess.run(convolute)
    print('input_image', input_image.shape, ', kernel', kernel.shape, 'feature_maps shape:', feature_maps.shape)

# input image:  size= 3x3; channels#= 5
# kernel:       size= 3x3; channels#= 5, padding='VALID'
# feature map:  size= 1x1; channels#= 1
# --- kernel dot product with image's each pixel and get the summation
input_image = tf.Variable(tf.random_normal([1,3,3,5]), dtype=tf.float32)
kernel = tf.Variable(tf.random_normal([3,3,5,1]), dtype=tf.float32)
convolute = tf.nn.conv2d(input_image, kernel, strides = [1,1,1,1], padding = 'VALID')
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    feature_maps = sess.run(convolute)
    print('input_image', input_image.shape, ', kernel', kernel.shape, 'feature_maps shape:', feature_maps.shape)

# input image:  size= 5x5; channels#= 5
# kernel:       size= 3x3; channels#= 5, padding='SAME'
# feature map:  size= 5x5; channels#= 1
# --- kernel dot product with image's each pixel and get the summation
input_image = tf.Variable(tf.random_normal([1,5,5,5]), dtype=tf.float32)
kernel = tf.Variable(tf.random_normal([3,3,5,1]), dtype=tf.float32)
convolute = tf.nn.conv2d(input_image, kernel, strides = [1,1,1,1], padding = 'SAME')
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    feature_maps = sess.run(convolute)
    print('input_image', input_image.shape, ', kernel', kernel.shape, 'feature_maps shape:', feature_maps.shape)

# input image:  size= 5x5; channels#= 5
# kernel:       size= 3x3; channels#= 5, padding='VALID'
# feature map:  size= 5x5; channels#= 1
# --- exclude the outter border pixels, return a 3x3 feature map
input_image = tf.Variable(tf.random_normal([1,5,5,5]), dtype=tf.float32)
kernel = tf.Variable(tf.random_normal([3,3,5,1]), dtype=tf.float32)
convolute = tf.nn.conv2d(input_image, kernel, strides = [1,1,1,1], padding = 'VALID')
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    feature_maps = sess.run(convolute)
    print('input_image', input_image.shape, ', kernel', kernel.shape, 'feature_maps shape:', feature_maps.shape)

# input image:  size= 5x5; channels#= 5
# kernel:       size= 3x3; channels#= 5, strides=[1, 2, 2, 1], padding='SAME'
# feature map:  size= 3x3; channels#= 1
# --- return feature maps with size of 3x3. notice this 3x3 is not the same one with padding='VALID'
input_image = tf.Variable(tf.random_normal([1,5,5,5]), dtype=tf.float32)
kernel = tf.Variable(tf.random_normal([3,3,5,1]), dtype=tf.float32)
convolute = tf.nn.conv2d(input_image, kernel, strides = [1,2,2,1], padding = 'SAME')
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    feature_maps = sess.run(convolute)
    print('input_image', input_image.shape, ', kernel', kernel.shape, 'feature_maps shape:', feature_maps.shape)

# input image:  size= 5x5; channels#= 5
# kernel:  7 x (size= 3x3; channels#= 5), padding='SAME'
# feature map:  size= 5x5; channels#= 7
# --- return a feature map with 7 channels
input_image = tf.Variable(tf.random_normal([1,5,5,5]), dtype=tf.float32)
kernel = tf.Variable(tf.random_normal([3,3,5,7]), dtype=tf.float32)
convolute = tf.nn.conv2d(input_image, kernel, strides = [1,1,1,1], padding = 'SAME')
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    feature_maps = sess.run(convolute)
    print('input_image', input_image.shape, ', kernel', kernel.shape, 'feature_maps shape:', feature_maps.shape)

# input image: (size= 5x5; channels#= 5) x 10
# kernel:  7 x (size= 3x3; channels#= 5), padding='SAME'
# feature map: (size= 5x5; channels#= 7) x 10
# --- return 10 feature maps with 7 channels each
input_image = tf.Variable(tf.random_normal([10,5,5,5]), dtype=tf.float32)
kernel = tf.Variable(tf.random_normal([3,3,5,7]), dtype=tf.float32)
convolute = tf.nn.conv2d(input_image, kernel, strides = [1,1,1,1], padding = 'SAME')
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    feature_maps = sess.run(convolute)
    print('input_image', input_image.shape, ', kernel', kernel.shape, 'feature_maps shape:', feature_maps.shape)
