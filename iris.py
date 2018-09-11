# coding=utf-8

'''
https://zhuanlan.zhihu.com/p/31646426

Iris 数据集 包含了 150 行数据，三种相关的 Iris 品种：
    Iris setosa，Iris virginica 和 Iris versicolor。
每种 Iris 品种有 50 个样本

对于每个花朵样本，每一行都包含了以下数据：萼片长度，萼片宽度，花瓣长度，花瓣宽度和花的品种。
花的品种用整数型数字表示，0表示 Iris setosa，1表示 Iris versicolor。
| Sepal Length | Sepal Width | Petal Length | Petal Width | Species |
|   4.9        | 3.0         | 1.4          | 0.2         | 0       |
|   7.0        | 3.2         | 4.7          | 1.4         | 1       |
|   6.5        | 3.0         | 5.2          | 2.0         | 2       |

Iris 数据随机打乱并划分成两个独立的 CSV 数据集：
    包含 120 个样本的训练集 (iris_training.csv)
    包含 30 个样本的测试集 (iris_test.csv)
    
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf

'''
    定义哪里下载数据和存储数据集：
'''
# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
  # If the training and test sets aren't stored locally, download them.
  if not os.path.exists(IRIS_TRAINING):
    raw = urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "wb") as f:
      f.write(raw)

  if not os.path.exists(IRIS_TEST):
    raw = urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "wb") as f:
      f.write(raw)

  # Load datasets.
  '''
  load_csv_with_header() 方法带有三个必要的参数：
      filename, 它从 CSV 文件得到文件路径
      target_dtype, 它采用数据集目标值的 ef="http://docs.scipy.org/doc/numpy/user/basics.types.html">numpy 数据类型
      features_dtype, 它采用数据集特征值的 ef="http://docs.scipy.org/doc/numpy/user/basics.types.html">numpy 数据类型   
  '''
  '''
    Dataset 在 tf.contrib.learn 中名为元组；你可以通过 data 和 target 字段访问特征数据和目标值。
    这里，training_set.data 和 training_set.target 分别包含了训练集的特征数据和特征值，
    而 test_set.data 和 test_set.target 分别包含了测试集的特征数据和目标值。
  '''
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename      = IRIS_TRAINING,
      target_dtype  = np.int, # 目标(值是你训练的模型的预测)是花的品种，它是一个从 0 到 2 的整数
      features_dtype= np.float32)
  
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename      = IRIS_TEST,
      target_dtype  = np.int,
      features_dtype= np.float32)


  '''
    配置一个深度神经网络分类器模型来适应 Iris 数据。使用 tf.estimator，通过两行代码来实例化你的 DNNClassifier：
    首先定义了模型的特征列，指定了数据集中的特征的数据类型。所有的特征数据都是连续的，所以 numeric_column 用于构造特征列。
    这里有四个特征(萼片宽度，萼片长度，花瓣宽度和花瓣长度)，于是 shape 必须设置为[4] 以适应所有的数据。    
  '''
  # Specify that all features have real-value data
  feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

  '''
    然后，代码使用以下参数创建了一个 DNNClassifier 模型：
  '''
  # Build 3 layer DNN with 10, 20, 10 units respectively.
  dnn_classifier = tf.estimator.DNNClassifier(
    feature_columns = feature_columns,  # 上面定义的一组特征列
    hidden_units    = [10, 20, 10],     # 三个 隐藏层，分别包含 10，20 和 10 神经元
    n_classes       = 3,                # 代表三种 Iris 品种
    model_dir       = "/tmp/iris_model" # TensorFlow 将在模型训练期间保存检测数据和 TensorBoard 摘要的目录
    )
  
  
  '''
    numpy_input_fn 产生输入管道
  '''
  # Define the training inputs
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x = {"x": np.array(training_set.data)},
      y = np.array(training_set.target),
      num_epochs = None,
      shuffle = True
      )

  '''
    # 将 train_input_fn 传递给 input_fn，并设置训练的次数
  '''
  # Train model.
  dnn_classifier.train(input_fn = train_input_fn, steps = 2000)
  '''
    模型的状态是保存在 classifier，这意味着如果你喜欢你可以迭代训练模型。例如，以上代码等同于以下代码：
    classifier.train(input_fn=train_input_fn, steps=1000)
    classifier.train(input_fn=train_input_fn, steps=1000)
  '''


  '''
    你已经在 Iris 训练数据上训练了你的 DNNClassifier 模型。
    现在你可以在 Iris 测试数据上使用 evaluate() 方法来检测模型的准确性。
  '''
  # Define the test inputs
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x = {"x": np.array(test_set.data)},
      y = np.array(test_set.target),
      num_epochs = 1, # test_input_fn 将迭代数据一次，然后触发 OutOfRangeError。这个错误表示分类器停止评估，所以它将对输入只评估一次。
      shuffle = False
      )

  '''
    evaluate() 返回一个包含评估结果的 dict
  '''
  # Evaluate accuracy.
  accuracy_score = dnn_classifier.evaluate(input_fn = test_input_fn)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))



  '''
    使用 estimator 的 predict() 方法来分类新的样本。
    例如，假如你有这两个新的花的样本：
    | Sepal Length | Sepal Width | Petal Length | Petal Width |
    | 6.4          | 3.2         | 4.5          | 1.5         |
    | 5.8          | 3.1         | 5.0          | 1.7         |    
    
    使用 predict() 方法预测它们的品种。 predict 返回一个dict，可以简单的将其转为 list 
  '''
  
  # Classify two new flower samples.
  new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], \
    dtype=np.float32)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x             = {"x": new_samples},
      num_epochs    = 1,
      shuffle       = False
      )
  
  predictions = list(dnn_classifier.predict(input_fn = predict_input_fn))
  print('predictions: ', predictions)
  predicted_classes = [p["classes"] for p in predictions]
  print("New Samples, Class Predictions:    {}\n".format(predicted_classes))

if __name__ == "__main__":
    main()
