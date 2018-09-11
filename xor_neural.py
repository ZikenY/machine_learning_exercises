# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:47:51 2017
http://www.cnblogs.com/Belter/p/6711160.html
@author: xin
"""
# Neural Network for XOR
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)

HIDDEN_LAYER_SIZE = 2
INPUT_LAYER = 2  # input feature
NUM_LABELS = 1  # output class number

# input data 4*2
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# labels 4*1
y = np.array([[0], [1], [1], [0]])

# Randomly initialize the weights of a layer
#      return: a matrix[L_out: L_in+1]
#      the first column of the matrix handles the "bias" terms
def rand_initialize_weights(L_in, L_out, epsilon):
    """
    Randomly initialize the weights of a layer with L_in
    incoming connections and L_out outgoing connections;

    Note that W should be set to a matrix of size(L_out, 1 + L_in) as
    the first column of W handles the "bias" terms
    """
    epsilon_init = epsilon
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_gradient(z):
    g = np.multiply(sigmoid(z), (1 - sigmoid(z)))
    return g


def nn_cost_function(theta1, theta2, X, y):
    m = X.shape[0]  # m=4
    # 计算所有参数的偏导数（梯度）
    D_1 = np.zeros(theta1.shape)  # Δ_1
    D_2 = np.zeros(theta2.shape)  # Δ_2
    h_total = np.zeros((m, 1))  # 所有样本的预测值, m*1, probability
    for t in range(m):
        a_1 = np.vstack((np.array([[1]]), X[t:t + 1, :].T))  # 列向量, 3*1
        z_2 = np.dot(theta1, a_1)  # 2*1
        a_2 = np.vstack((np.array([[1]]), sigmoid(z_2)))  # 3*1
        z_3 = np.dot(theta2, a_2)  # 1*1
        a_3 = sigmoid(z_3)
        h = a_3  # 预测值h就等于a_3, 1*1
        h_total[t,0] = h
        delta_3 = h - y[t:t + 1, :].T  # 最后一层每一个单元的误差, δ_3, 1*1
        delta_2 = np.multiply(np.dot(theta2[:, 1:].T, delta_3), sigmoid_gradient(z_2))  # 第二层每一个单元的误差（不包括偏置单元）, δ_2, 2*1
        D_2 = D_2 + np.dot(delta_3, a_2.T)  # 第二层所有参数的误差, 1*3
        D_1 = D_1 + np.dot(delta_2, a_1.T)  # 第一层所有参数的误差, 2*3
    theta1_grad = (1.0 / m) * D_1  # 第一层参数的偏导数，取所有样本中参数的均值，没有加正则项
    theta2_grad = (1.0 / m) * D_2
    J = (1.0 / m) * np.sum(-y * np.log(h_total) - (np.array([[1]]) - y) * np.log(1 - h_total))
    return {'theta1_grad': theta1_grad,
            'theta2_grad': theta2_grad,
            'J': J, 'h': h_total}


theta1 = rand_initialize_weights(INPUT_LAYER, HIDDEN_LAYER_SIZE, epsilon=1)  # 之前的问题之一，epsilon的值设置的太小
theta2 = rand_initialize_weights(HIDDEN_LAYER_SIZE, NUM_LABELS, epsilon=1)

iter_times = 10000  # 之前的问题之二，迭代次数太少
alpha = 0.5  # 之前的问题之三，学习率太小
result = {'J': [], 'h': []}
theta_s = {}
for i in range(iter_times):
    cost_fun_result = nn_cost_function(theta1=theta1, theta2=theta2, X=X, y=y)
    theta1_g = cost_fun_result.get('theta1_grad')
    theta2_g = cost_fun_result.get('theta2_grad')
    J = cost_fun_result.get('J')
    h_current = cost_fun_result.get('h')
    theta1 -= alpha * theta1_g
    theta2 -= alpha * theta2_g
    result['J'].append(J)
    result['h'].append(h_current)
    # print(i, J, h_current)
    if i==0 or i==(iter_times-1):
        print('theta1', theta1)
        print('theta2', theta2)
        theta_s['theta1_'+str(i)] = theta1.copy()
        theta_s['theta2_'+str(i)] = theta2.copy()

plt.plot(result.get('J'))
plt.show()
print(theta_s)
print(result.get('h')[0], result.get('h')[-1])