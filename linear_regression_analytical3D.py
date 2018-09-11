# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
'''
        sudo apt-get remove python3-matplotlib
        sudo pip install -U matplotlib
'''
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline

string_data = """1    1    39    39    144
2     1    47    147    220
3     1    45    155    138
4     1    47    127    145
5     1    65    155    162
6     1    46    140    142
7     1    67    171    170
8     1    42    148    124
9     1    67    157    158
10    1    56    156    154
11    1    64    134    162
12    1    56    152    150
13    1    59    159    140
14    1    34    134    110
15    1    42    142    128
16    1    48    148    130
17    1    45    125    135
18    1    17    167    114
19    1    20    123    116
20    1    19    159    124
21    1    36    136    136
22    1    50    150    142
23    1    39    139    120
24    1    21    121    120
25    1    44    144    160
26    1    53    153    158
27    1    63    163    144
28    1    29    129    130
29    1    25    125    125
30    1    69    169    175"""

# we are going to transfer "rawlines" to "rawdata"
rawlines = string_data.split('\n')

# rawdata -   no    bias  x1   x2    Y
#           [[1.    1.   39.   33.  144.]
#            [2.    1.   47.   33.  220.]
#            [3.    1.   45.   33.  138.] ... 
rawdata = np.zeros([len(rawlines), 5])

# scatter points
scatter_x = []     # [[x1, x2]]`
scatter_y = []     # [y]

min_x1 = sys.maxsize
max_x1 = -sys.maxsize -1
min_x2 = sys.maxsize
max_x2 = -sys.maxsize -1

for row in range(0, len(rawlines)):
    line = rawlines[row]
    line_elements = line.split()    # line_elements - ['4', '1', '47', '58', '145']

    # determind the scales of x axis for the line
    if (int(line_elements[2]) < min_x1):
        min_x1 = int(line_elements[2])        
    if (int(line_elements[2]) > max_x1):
        max_x1 = int(line_elements[2])        
    if (int(line_elements[3]) < min_x2):
        min_x2 = int(line_elements[3])        
    if (int(line_elements[3]) > max_x2):
        max_x2 = int(line_elements[3])        

    # for scatter points
    scatter_x.append([line_elements[2], line_elements[3]])   # x1, x2
    scatter_y.append(line_elements[4])   # y

    # transfer data from rawlines to the 3D-array rawdata
    for col in range(0, 5):
        rawdata[row, col] = line_elements[col]

    row = row + 1



#         b   x1   x2
# X -   [[1.  39.  69.]            n x 3 array
#        [1.  47.  69.]
#        [1.  45.  69.] ...          
X = rawdata[:, 1:4:1]       # all rows, get col[1] to col[3] (not include col[4]) with stride 1          

# all rows, get col[3] for Y    n x 1 array
# Y = np.array(rawdata[:, 4]).reshape(len(rawlines), 1)
Y = rawdata[:, 4]


#----------------- do the linear regression (analytical method) ------------------
# W[0] - b
# W[1] - slope1
# W[2] - slope2                                                 W.shape: 3x1
#          3xn                  3x3    3xn nx3   3xn   nx1
W = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
#----------------- ------------------------ ------------------ ------------------

# prepare the x, y labels
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')

scatter_x = np.array(scatter_x, dtype = float)
scatter_y = np.array(scatter_y, dtype = float)
x1 = scatter_x[:, 0]
x2 = scatter_x[:, 1]
ax.scatter(x1, x2, scatter_y, c = 'g', marker = 'o', label = 'samples')

# how to draw a plane ?
X1 = np.arange(min_x1, max_x1)
X2 = np.arange(min_x2, max_x2)
X1, X2 = np.meshgrid(X1, X2)
Y = W[0] + W[1]*X1 + W[2]*X2
ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap='rainbow')

plt.legend()
plt.show
