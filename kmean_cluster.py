# -*- coding: utf-8 -*-
#################################################
# kmeans: k-means cluster
# Author : zouxy
# Date   : 2013-12-25
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

'''
http://blog.csdn.net/zouxy09/article/details/17589329
'''

from numpy import *
import sys
import time
import matplotlib.pyplot as plt


# calculate Euclidean distance
# e.g. sqrt(sum(power(matrix([2, 3]) - matrix([4, 5]), 2))) == 2.8
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))


# init centroids with random samples
# return centroids - k row, 2 column
def initCentroids(dataSet, k):          # dataset有80行2列，每列[x, y]
    sample_count, dim = dataSet.shape
    centroids = zeros((k, dim))         # zeros (初始化为k个[0, 0])

    for i in range(k):
        # 随机返回0~80中的一行
        index = int(random.uniform(0, sample_count))
        centroids[i, :] = dataSet[index, :]

    # 一共返回k行 随机的centroids
    return centroids


# k-means cluster (k clusters)
#   dataSet - 80行, 2列
def kmeans(dataSet, k):
    print('k = ', k)

    sample_count = dataSet.shape[0]

    # return clusters
    # first column stores which cluster this sample belongs to,             
    # second column stores the error between this sample and its centroid
    # 
    # 每行对应于每个sample
    # col0 - 属于第几个cluster
    # col1 - 这个sample与对应的centroid的距离平方，也就是所谓square error
    clusters = matrix(zeros((sample_count, 2)))    

    # 只有在step3 有cluster被更新后才需要重新计算
    # 如果没人跳槽说明已经收敛到稳定状态
    changed = True

    ## step 1: init centroids
    centroids = initCentroids(dataSet, k)

    while changed:
        changed = False

        # 根据当前的每个centroids调整所有sample的分类和err
        '''
        循环每个sample：
            1. in each iteration, 循环k个centroid：
                    找到离这个sample最近的centroid，计算修改距离(err); 
            2. 如果跳槽，修改clusters[i, [cluster#, err]]
        '''
        for i in range(sample_count):
            
            minIndex = "计算本轮距离这个sample最近的centroid的index"
            minDist = sys.maxsize # distance between this point to its centroid (err)

            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                # 第j个centroid与第i个sample的距离
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist  = distance
                    minIndex = j

            ## step 3: update its cluster
            if clusters[i, 0] != minIndex:              # 第0列是这个sample的cluster号
                clusters[i, :] = minIndex, minDist**2   # 注意是距离平方，non-negative
                changed = True                          # 有人跳槽了，得调整新的

        '''
        根据当前所有sample的位置，调整每个centroid
        '''
        ## step 4: update centroids
        for j in range(k):
            # clusters[:, 0].A表示取出每一行的第0列并转换为array  (col0 - 这个sample属于第几个cluster)
            # (clusters[:, 0].A == j) : 如果col0==第j个cluster就是[True], 否则是[False]， 返回一个array
            # nonzero(...)[0] : 返回所有True的行号，一个1D array
            acluster = nonzero(clusters[:, 0].A == j)[0]
            
            # 从dataset中取出行数in acluster的那些samples (？行2列)
            samples_in_cluster = dataSet[acluster]

            #                       按列求mean，变成一行2列
            # 更新第j个centroid
            centroids[j, :] = mean(samples_in_cluster, axis = 0)

    print( 'Congratulations, cluster complete! ', 'centroids.shape: ', centroids.shape, ', clusters.shape:', clusters.shape)
    return centroids, clusters


# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusters):
    sample_count, dim = dataSet.shape
    if dim != 2:
        print( "Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print( "Sorry! Your k is too large! please contact Zouxy")
        return 1

    # draw all samples
    for i in range(sample_count):
        markIndex = int(clusters[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)

    plt.show()

if __name__ == '__main__':
    ## step 1: load data  
    print( "step 1: load data..."  )
    dataSet = []  
    fileIn = open('testset_kmean.txt')  
    for line in fileIn.readlines():  
        lineArr = line.strip().split()  
        dataSet.append([float(lineArr[0]), float(lineArr[1])])  

    ## step 2: clustering...  
    print( "step 2: clustering..."  )
    dataSet = matrix(dataSet)  # make the 2-D list a numpy matrix
    
    k = 5
    centroids, clusters = kmeans(dataSet, k)  

    ## step 3: show the result  
    print( "step 3: show the result..."  )
    showCluster(dataSet, k, centroids, clusters)

    fileIn.close()


'''

a = matrix([[0, 1, 2],
            [4, 5, 6],
            [7, 8, 9]])

in:     a[:, :].A == (0, 5, 6)
out:    array([[ True, False, False],
               [False,  True,  True],
               [False, False, False]], dtype=bool)
    
in:     a[:, 0].A == 4
out:    array([[False],
               [ True],
               [False]], dtype=bool)

in:     np.nonzero(a[:, 0].A)[0]
Out:    array([1, 2])

in:     np.nonzero(a[:, 0].A == 4)[0]
Out:    array([1])

in:     (a[:, 0].A == 4).reshape(1,3)[0]
Out:    array([False,  True, False], dtype=bool)
'''