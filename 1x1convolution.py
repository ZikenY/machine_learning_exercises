# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.pyplot as plt
#from scipy import ndimage
#from PIL import Image


# Initialize a 5x5 Gabor filter
wts_a = np.array([[0.0257, -0.0671, 0.0, 0.0671, -0.0257],
        [0.1151, -0.3009, 0.0, 0.3009, -0.1151],
        [0.1897, -0.4961, 0.0, 0.4961, -0.1897],
        [0.1151, -0.3009, 0.0, 0.3009, -0.1151],
        [0.0257, -0.0671, 0.0, 0.0671, -0.0257]], dtype=np.float32) # 'shape:', (5, 5)

# expand the shape: (5, 5, 1)
wts2_3d = np.expand_dims(wts_a, axis=2)

# expand the shape again: (5, 5, 1, 1)
wts2_4d_gabor1 = np.expand_dims(wts2_3d, axis=3)
print('Wts2_4d shape:', wts2_4d_gabor1.shape)

# another 5x5 Gabor filter
wts_b = np.array([[0.0000, 0.0647, -0.0564, -0.0466, 0.0131],
        [-0.0647, 0.0, 0.4781, -0.1534, -0.0466],
        [0.0564, -0.4781, 0.0, 0.4781, -0.0564],
        [0.0466, 0.1534, -0.4781, 0.0, 0.0647],
        [-0.0131, 0.0466, 0.0564, -0.0647, 0.0]], dtype=np.float32) # 'shape:', (5, 5)
wts2_3d = np.expand_dims(wts_b, axis=2) # shape: (5, 5, 1)
wts2_4d_gabor2 = np.expand_dims(wts2_3d, axis=3)  # shape: (5, 5, 1, 1)


wts_a_t = wts_a.transpose()
wts2_3d = np.expand_dims(wts_a_t, axis=2) # shape: (5, 5, 1)
wts2_4d_gabor3 = np.expand_dims(wts2_3d, axis=3)  # shape: (5, 5, 1, 1)

wts_b_t = wts_b.transpose()
wts2_3d = np.expand_dims(wts_b_t, axis=2) # shape: (5, 5, 1)
wts2_4d_gabor4 = np.expand_dims(wts2_3d, axis=3)  # shape: (5, 5, 1, 1)


cwd = os.getcwd()
print(cwd)
filename = 'image_0004_leafCropped.jpg'
print(filename)
#im = ndimage.imread(cwd+'/'+filename) # both ways work
im = plt.imread(cwd+'/'+filename, format = 'jpeg')
plt.imshow(im, cmap = cm.Greys_r)
plt.show()
print('Image shape after reading:',im.shape)    #(451, 451, 3)
print('Image data type:', type(im))             #'numpy.ndarray'



# let im4d be the input data
# original image is (451, 451, 3). make it (1, 451, 451, 3) because there is only 1 input image
im4d = np.expand_dims(im, axis=0)

print('Add new dimension for conv2d compatibility.')
print('New shape: ', im4d.shape)


#im2 = Image.open(filename) # returns 'PIL.JpegImagePlugin.JpegImageFile'
#print(type(im2))
#plt.imshow(im)

# create a new graph for the following session
model = tf.Graph()

# 使用 model.as_default() 设model为默认, 接下来的操作是在model中了, 而不是在原先的系统默认graph中
# 这里的变量定义全部在model中
with model.as_default():

    # perform a 1x1 convolution with 3 channels (only 1 kernel)
    channels = [[0.21], [0.72], [0.07]]     # array shape: 3x1         对应r, g, b各分量在灰度图中的权重
    array_1x3x1 = [channels]
    kernel_1x1x3x1 = [array_1x3x1]          # 注意kernels的最后一维代表kernel的个数, 也就是输出featuremaps的channel数 !!!


    # kernel shape = (1, 1, 3, 1):  size 1x1, 3 channels, only 1 kernel             kernel = tf.constant([[[[0.21], [0.72], [0.07]]]], dtype=tf.float32)
    kernel = tf.constant(kernel_1x1x3x1, dtype = tf.float32)


    #input shape(1, 451, 451, 3): only 1 image, size 451x451, 3 channels
    inputdata = tf.constant(im4d, dtype=tf.float32)


    # gray is a op executing 1x1 convolution
    # inputdata:            (1, 451, 451, 3)        1张图片
    # kernel:                   (1,   1,  3, 1)     size(1x1), 3 channels, 只有1个kernel
    # output featuremaps:   (1, 451, 451, 1)        1个featuremap(来自1张图片), size(451x451), 1 channels(来自1个kernel)
    gray = tf.nn.conv2d(inputdata, kernel, strides = [1, 1, 1, 1], padding='SAME')


    # ===  到这里为止我们用1x1 卷积核将3通道彩色图片转换为1通 道灰度图  ===
    #       gray就是后面convolution操作的输入图片


    # 5x5 Gabor filter kernel with shape(5, 5, 1, 1)
    #                               size(5x5), 1 channel, only 1 kernel
    # 注意这里的用法：将nparray转换为tensor的kernel
    kernel_gabor1  = tf.convert_to_tensor(wts2_4d_gabor1, dtype=tf.float32)
    kernel_gabor2  = tf.convert_to_tensor(wts2_4d_gabor2, dtype=tf.float32)
    kernel_gabor3  = tf.convert_to_tensor(wts2_4d_gabor3, dtype=tf.float32)
    kernel_gabor4  = tf.convert_to_tensor(wts2_4d_gabor4, dtype=tf.float32)
    shape_gabor_tf = kernel_gabor1.get_shape().as_list()
    print('Shape wts2_4d_tf', shape_gabor_tf)


    # vert is a op executing 5x5 convolution
    # inputdata: gray.shape(1, 451, 451, 1): 1张图片, size(451x451), 1 channel
    vert            = tf.nn.conv2d(gray, kernel_gabor1, [1, 1, 1, 1], padding='SAME')
    diagonal_ccw    = tf.nn.conv2d(gray, kernel_gabor2, [1, 1, 1, 1], padding='SAME')
    horizonal       = tf.nn.conv2d(gray, kernel_gabor3, [1, 1, 1, 1], padding='SAME')
    diagonal_clw    = tf.nn.conv2d(gray, kernel_gabor4, [1, 1, 1, 1], padding='SAME')

    
# use 'model' as new graph to execute gray operation
# session用来执行刚才定义的graph（model），sess.run(gray)代表激活运行gray这个tensor，
# 系统会从最开始的inputdata获取输入数据，做卷积，将output featuremaps（size(1,451,451, 1)）返回给output
with tf.Session(graph = model) as sess:
    output = sess.run(gray)


print('Output shape after grayscale conversion: ', output.shape)
output.resize((451, 451))
print('Resized for imshow:', output.shape)
print('Print some matrix values to show it is grayscale.')
print(output)
print('Display the grayscale image.')
plt.imshow(output, cmap = cm.Greys_r)


    
# Run the model to detect vertical edges
with tf.Session(graph=model) as sess:
    output = sess.run(vert)
print('Output shape after grayscale conversion: ', output.shape)
output.resize((451, 451))
print('Resized for imshow:', output.shape)
#print(output.shape)
print('Print some matrix values to show it is grayscale.')
print(output)
print('Display the grayscale feature map with vertical edges.')
plt.figure('feature map with vertical edges')
plt.imshow(output, cmap = cm.Greys_r)
plt.show()

# Run the model to detect horizonal edges
with tf.Session(graph=model) as sess:
    output = sess.run(horizonal)
output.resize((451, 451))
print('Print some matrix values to show the result of horizonal edges detection.')
print(output)
print('Display the grayscale feature map with horizonal edges.')
plt.figure('feature map with horizonal edges')
plt.imshow(output, cmap = cm.Greys_r)
plt.show()

# Run the model to detect diagonal_ccw edges
with tf.Session(graph=model) as sess:
    output = sess.run(diagonal_ccw)
output.resize((451, 451))
print('Print some matrix values to show the result of diagonal_ccw edges detection.')
print(output)
print('Display the grayscale feature map with diagonal_ccw edges.')
plt.figure('feature map with diagonal_ccw edges')
plt.imshow(output, cmap = cm.Greys_r)
plt.show()

# Run the model to detect diagonal_clw edges
with tf.Session(graph=model) as sess:
    output = sess.run(diagonal_clw)
output.resize((451, 451))
print('Print some matrix values to show the result of diagonal_clw edges detection.')
print(output)
print('Display the grayscale feature map with diagonal_clw edges.')
plt.figure('feature map with diagonal_clw edges')
plt.imshow(output, cmap = cm.Greys_r)
plt.show()
    