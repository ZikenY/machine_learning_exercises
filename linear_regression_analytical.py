# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 22:36:20 2017

@author: ziken
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

'''
rawlines = []
raw = open("raw_bloodpressure_data.txt")
while 1:
    lines = raw.readlines(100000)
    if not lines:
        break
    for line in lines:
        rawlines.append(line.strip())
raw.close()
'''

raw_bloodpressure_data = """1  1  39  144
 2  1  47  220
 3  1  45  138
 4  1  47  145
 5  1  65  162
 6  1  46  142
 7  1  67  170
 8  1  42  124
 9  1  67  158
10  1  56  154
11  1  64  162
12  1  56  150
13  1  59  140
14  1  34  110
15  1  42  128
16  1  48  130
17  1  45  135
18  1  17  114
19  1  20  116
20  1  19  124
21  1  36  136
22  1  50  142
23  1  39  120
24  1  21  120
25  1  44  160
26  1  53  158
27  1  63  144
28  1  29  130
29  1  25  125
30  1  69  175"""

# we are going to transfer "rawlines" to "rawdata"
rawlines = raw_bloodpressure_data.split('\n')

# rawdata -   no    bias  x    Y
#           [[1.    1.   39.  144.]
#            [2.    1.   47.  220.]
#            [3.    1.   45.  138.] ... 
rawdata = np.zeros([len(rawlines), 4])

# scatter points
scatter_x = []     # age
scatter_y = []     # systolic blood pressure

for row in range(0, len(rawlines)):
    line = rawlines[row]
    line_elements = line.split()    # line_elements - ['4', '1', '47', '145']

    # for scatter points
    scatter_x.append(line_elements[2])   # age
    scatter_y.append(line_elements[3])   # systolic blood pressure

    # transfer data from rawlines to the 2D-array rawdata
    for col in range(0, 4):
        rawdata[row, col] = line_elements[col]

    row = row + 1

# X -   [[1.  39.]            n x 2 array
#        [1.  47.]
#        [1.  45.] ...          
X = rawdata[:, 1:3:1]       # all rows, get col[1] to col[2] (not include col[3]) with stride 1          

# all rows, get col[3] for Y    n x 1 array
Y = rawdata[:, 3]


#----------------- do the linear regression (analytical method) ------------------
# W[0] - intercept          W.shape: 2x1
# W[1] - slope
W = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
#----------------- ------------------------ ------------------ ------------------


# determind the y-axis of the line
# the line is from(min_x, min_y) to(max_x, max_y)
min_x = float(min(np.array(scatter_x)))
max_x = float(max(np.array(scatter_x)))
min_y = W[1]*min_x + W[0]
max_y = W[1]*max_x + W[0]

# prepare the x, y labels
plt.figure()

# draw all samples  
# plt.scatter(scatter_x, scatter_y, c = 'g', marker = 'o', label = 'samples')
for i in range(len(scatter_x)):
    plt.plot(X[i][1], Y[i], 'ob')

# draw line
line_label = '$f(x) = ' + '{0:.2f}'.format(W[1]) + 'x + ' + '{0:.2f}'.format(W[0]) + '$'
plt.plot([min_x, max_x], [min_y, max_y], '-g', label = line_label)

# plt.title('linear regression')
plt.xlabel('X (age)')
plt.ylabel('Y (systolic blood pressure)')
plt.legend()
plt.show
