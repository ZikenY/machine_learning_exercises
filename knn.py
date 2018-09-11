# -*- coding: utf-8 -*-
#########################################
# kNN: k Nearest Neighbors

# Input:      newInput: vector to compare to existing dataset (1xN)
#             dataSet:  size m data set of known vectors (NxM)
#             labels:     data set labels (1xM vector)
#             k:         number of neighbors to use for comparison 

# Output:     the most popular class label
#########################################

# http://blog.csdn.net/zouxy09/article/details/16955347

from numpy import *
import operator


# !!! 注意numpy.array的运算都是element-wise !!!

# classify using kNN
#       dataSet[sample#, feature#]
#       labels - corresponding to row# of dataSet
#       k - k
#       newInput[x, y]
#
#   return: class of newInput from labels
def kNNClassify(dataSet, labels, k, newInput):

    # the num of row
    numSamples = dataSet.shape[0]

    # step 1: calculate Euclidean distance （目的是计算input与所有点的距离）
    # dist = sqrt((x1-x0)^2 + (y1-y0)^2)

    # repeat the input by # of dataset row （第二个元素1代表每行只repeat一次）
    reps = (numSamples, 1)
    
    # tile(A, reps): Construct an array by repeating A reps times
    tiling_input = tile(newInput, reps)

    # Subtract square element-wise         （分别计算input与dataset中每个元素的差平方）
    diff = tiling_input - dataSet          #(x1-x0), (y1-y0)
    squaredDiff = diff ** 2                #(x1-x0)^2, (y1-y0)^2

    # 现在我们有sample# (row)个difference square，每行为[x, y]
    
    # sum is performed by row              （如果不指定axis，就是统计所有元素的sum了）
    squaredDist = sum(squaredDiff, axis = 1)
    # 变成了一列 （数组 shape(4,)）    
    # square root to get distance
    distance = squaredDist ** 0.5

    # step 2: sort the distance -> it means rank the labels for this input

    # argsort() returns "the indices" that would sort an array in a ascending order
    sorted_indices = argsort(distance)

    # store the top k distances' {labels -> vote# for this input}
    classCount = dict()

    for i in range(k):
        
        # step 3: choose the label for the min k distance
        label_index = sorted_indices[i]
        voteLabel = labels[label_index]

        # step 4: count the times labels occur

        # dict.get(key, default = None) 
        #               default -- 如果指定键的值不存在时，返回该默认值值。
        # when the key voteLabel is not in dictionary classCount, get() will return 0
        vote = classCount.get(voteLabel, 0) + 1
        
        # 这个input是voteLabel的投票数+1
        
        classCount[voteLabel] = vote

    ## step 5: the max voted class will return
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex

# create a dataset which contains 4 samples with 2 classes
def createDataSet():
    
     # create a matrix: each row as a sample
    #               [x ,  y]
    group = array([[1.0, 0.9],
                   [1.0, 1.0],
                   [0.1, 0.2],
                   [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B'] # four samples and two classes
    return group, labels

if __name__ == '__main__':
    
    dataSet, labels = createDataSet()
    k = 3  
    
    #              [x ,  y]
    testX = array([1.2, 1.0])
    
#    print('dataSet.shape: ', dataSet.shape)
    print('dataSet: \n', dataSet, ' k =', k)
    outputLabel = kNNClassify(dataSet, labels, k, testX)  
    print( "\nYour input is:\n", testX, "\n classified to class: ", outputLabel)
      
#    testX = array([0.1, 0.3])  
#    outputLabel = kNNClassify(testX, dataSet, labels, k)  
#    print("Your input is:", testX, "and classified to class: ", outputLabel)

    import platform  
    print('python ver:', platform.python_version())
