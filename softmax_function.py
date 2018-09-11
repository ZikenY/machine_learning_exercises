# -*- coding: utf-8 -*-

import math
z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
z_exp = [math.exp(i) for i in z]  
print(z_exp)  # Result: [2.72, 7.39, 20.09, 54.6, 2.72, 7.39, 20.09] 
sum_z_exp = sum(z_exp)  
print(sum_z_exp)  # Result: 114.98 
softmax = [round(i / sum_z_exp, 3) for i in z_exp]

# 输出向量中拥有最大权重的项对应着输入向量中的最大值“4”。这也显示了这个函数通常的意义：对向量进行归一化，凸显其中最大的值并抑制远低于最大值的其他分量。

print(softmax)  # Result: [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]