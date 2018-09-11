# -*- coding: utf-8 -*-
#########################################
# kNN: k Nearest Neighbors

# Input:      inX: vector to compare to existing dataset (1xN)
#             dataSet: size m data set of known vectors (NxM)
#             labels: data set labels (1xM vector)
#             k: number of neighbors to use for comparison 
            
# Output:     the most popular class label
#########################################

# http://blog.csdn.net/zouxy09/article/details/16955347

from numpy import *
import operator
import os


# classify using kNN
def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0] # shape[0] stands for the num of row

    ## step 1: calculate Euclidean distance
    # tile(A, reps): Construct an array by repeating A reps times
    # the following copy numSamples rows for dataSet
    diff = tile(newInput, (numSamples, 1)) - dataSet # Subtract element-wise
    
    # compute the distances between newInput and each row
    squaredDiff = diff ** 2 # squared for the subtract
    squaredDist = sum(squaredDiff, axis = 1) # sum is performed by row
    distances = squaredDist ** 0.5

    ## step 2: sort the distances
    # argsort() returns the indices that would sort an array in a ascending order
    sortedDistIndices = argsort(distances)

    classCount = {} # define a dictionary (can be append element)
    for i in range(k):
        ## step 3: choose the min k distances
        voteLabel = labels[sortedDistIndices[i]]

        ## step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    ## step 5: the max voted class will return
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex    

# convert image to a 1-D vector
def  img2vector(filename):
     rows = 32
     cols = 32
     imgVector = zeros((1, rows * cols)) 
     fileIn = open(filename)
     for row in range(rows):
         lineStr = fileIn.readline()
         for col in range(cols):
             imgVector[0, row * 32 + col] = int(lineStr[col])

     return imgVector

# load trainning dataSet and testing dataset from files
#       return:
#               train_x.shape:  (1934, 1024)
#               train_y.shape:  (1934,)
#               test_x.shape:  (946, 1024)
#               test_y.shape:  (946,)
#     
def loadDataSet():
    ## step 1: Getting training set
    print("---Getting training set...")
    dataSetDir = './'
    trainingFileList = os.listdir(dataSetDir + 'digits_train') # load the training set
    numSamples = len(trainingFileList)

    train_x = zeros((numSamples, 1024)) #有1024个features
    train_y = []
    for i in range(numSamples):
        filename = trainingFileList[i]

        # get train_x
        # 每个图像为32*32，有1024个features，
        # 展开为1D array, 写入第i个train_x数组
        train_x[i, :] = img2vector(dataSetDir + 'digits_train/%s' % filename)

        # get label from file name such as "1_18.txt"
        label = int(filename.split('_')[0]) # return 1
        train_y.append(label)

    ## step 2: Getting testing set
    print("---Getting testing set...")
    testingFileList = os.listdir(dataSetDir + 'digits_test') # load the testing set
    numSamples = len(testingFileList)
    test_x = zeros((numSamples, 1024)) #有1024个features
    test_y = []
    for i in range(numSamples):
        filename = testingFileList[i]

        # get test_x
        # 每个图像为32*32，有1024个features，
        # 展开为1D array, 写入第i个test_x数组
        test_x[i, :] = img2vector(dataSetDir + 'digits_test/%s' % filename) 

        # get label from file name such as "1_18.txt"
        label = int(filename.split('_')[0]) # return 1
        test_y.append(label)

    return train_x, train_y, test_x, test_y

# test hand writing class
def testHandWritingClass():
    ## step 1: load data
    print("step 1: load data...")
    train_x, train_y, test_x, test_y = loadDataSet()
    print("train_x.shape: ", array(train_x).shape)
    print("train_y.shape: ", array(train_y).shape)
    print("test_x.shape: ", array(test_x).shape)
    print("test_y.shape: ", array(test_y).shape)

    ## step 2: training...
    print("step 2: \nnot need training because it is a lazy algorithm...")
    pass

    ## step 3: testing
    print("step 3: testing... （这里有946个test data）")
    numTestSamples = test_x.shape[0]
    matchCount = 0
    for i in range(numTestSamples):
        #                    test input, dataset, label, k=3
        predict = kNNClassify(test_x[i], train_x, train_y, 3)
        
        # judge its correctness
        if predict == test_y[i]:
            matchCount += 1

    accuracy = float(matchCount) / numTestSamples

    ## step 4: show the result
    print("step 4: show the result...")
    print('The classify accuracy is: %.2f%%' % (accuracy * 100))
    
if __name__ == '__main__':
    testHandWritingClass()