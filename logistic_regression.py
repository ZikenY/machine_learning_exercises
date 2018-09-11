#################################################  
# logRegression: Logistic Regression  
#################################################  

# http://blog.csdn.net/zouxy09/article/details/20319673

from numpy import *
import matplotlib.pyplot as plt
import time


# calculate the sigmoid function  work on scalars or vectors
def sigmoid(x):  
    return 1.0 / (1 + exp(-x))  


# train a logistic regression model using some optional optimize algorithm  
# input: train_x is a mat datatype, each row stands for one sample  
#        label is mat datatype too, each row is the corresponding label  
def train(train_x, label, alpha, max_iteration, optimize_type):  
    print(optimize_type)
    startTime = time.time()

    #   100            3
    sample_count, feature_count = shape(train_x)  

    # weights = ones((feature_count, 1)) # 3x1
    weights = random.uniform(0, 1, size=feature_count).reshape(feature_count, 1) # 贱

    # optimize through gradient descent algorilthm  
    for k in range(max_iteration):
        
        error = 'vector or scalar, 用来记录每轮迭代计算gradient后label - _y之间的偏差'
        
        if optimize_type == 'gradDescent':

            # ----- gradient descent algorilthm  -----
            # 一轮迭代中，对整个trainning data计算梯度 (error.transpose() * train_x)
            
            _y = sigmoid(train_x * weights)         # sigmoid(100x3 * 3x1)      每一行x计算一个_y和一个error，共100个error，
            error = label - _y                      # 100x1                     等会这100个error乘以各自的x后变成gradient，
                                                    #                           所以每轮每次在调整weights之前都遍历了整个training set

            #       注意这里的+号，实际上是gradient ascent. 因为我们在maximize the likelihood estimation
            weights = weights + alpha * train_x.transpose() * error
#            weights = weights + alpha * (error.transpose() * train_x).transpose()
            
            # make error a scalar for display
            error = error.A
            error = sqrt(error**2)
            error = sum(error) / (error.shape[0] * error.shape[1])


        elif optimize_type == 'stocGradDescent':

            # ----- stochastic gradient descent  -----

            for i in range(sample_count):
                # 在这一轮迭代中，每个x计算出一个y和error，然后马上更新weights, 所以gradient下降（上升）得很快
                x = train_x[i, :]                   # 1x3
                _y = sigmoid(x * weights)           # sigmoid(1x3 * 3*1)
                error = label[i, 0] - _y            # 注意这里error是个scalar
                weights = weights + alpha * train_x[i, :].transpose() * error       # 取出x中对应的第i行乘以error来计算gradient
                
        elif optimize_type == 'smoothStocGradDescent':
            
            # ----- smooth stochastic gradient descent -----
            # randomly select samples to optimize for reducing cycle fluctuations   
                        
            # 建立一个index表，每次随机从中选出一条记录index（training data）去gd，并从此表中删除这条index
            index_list = list(range(sample_count))   # shape (100,) [0, 1, 2,,,, 99]
            
            for i in range(sample_count):

                # 反正是越来越小
                alpha = 4.0 / (1.0 + k + i) + 0.01  

                # 每次随机从中选出一条index 
                # 注意index_rand是index_list的index，不是里面存放的train_x的index
                index_rand = int(random.uniform(0, len(index_list)))
                
                # 这才是用来从train_x中取sample的index
                x_index = index_list[index_rand]
                
                # 用这条x计算error和gradient, 并更新weight
                _y = sigmoid(train_x[x_index, :] * weights)
                error = label[x_index, 0] - _y
                weights = weights + alpha * train_x[x_index, :].transpose() * error  

                # 用完了就干掉, 所以每次取出的x都不同
                del(index_list[index_rand]) # during one interation, delete the optimized sample

        else:  
            raise NameError('Not support optimize method type!')

        print('error: %0.14f' % fabs(error), '   on iteration ', k)
    print('Congratulations, training complete! Took %fs!' % (time.time() - startTime))
  
    # 如果feature# = 2，分隔线方程为: W0 + W1*x + W2*y = 0,     
    # x, y对应两个train_x分量，对应sample point平面坐标
    return weights  


# test your trained Logistic Regression model given test set  
def test(weights, test_x, test_y):  
    sample_count, feature_count = shape(test_x)  
    matchCount = 0  
    for i in range(sample_count):  
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5  
        if predict == bool(test_y[i, 0]):  
            matchCount += 1  
    accuracy = float(matchCount) / sample_count  
    return accuracy  


# show your trained logistic regression model
# only available with 2-D data  
def show_result(weights, train_x, label):  
    # notice: train_x and label is mat datatype  
    sample_count, feature_count = shape(train_x)  
    if feature_count != 3:  
        print("Sorry! I can not draw because the dimension of your data is not 2!"  )
        return 1  

    # draw all samples  
    for i in range(sample_count):  
        if int(label[i, 0]) == 0:  
            plt.plot(train_x[i, 1], train_x[i, 2], 'or')  
        elif int(label[i, 0]) == 1:  
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')  

    # draw the classify line  
    min_x = min(train_x[:, 1])[0, 0]  
    max_x = max(train_x[:, 1])[0, 0]  
    weights = weights.getA()
    # 注意分隔线方程为: W0 + W1*x + W2*y = 0,     x, y对应两个train_x分量，对应sample point平面坐标
    y_min_x = -float(weights[0] + weights[1] * min_x) / weights[2]
    y_max_x = -float(weights[0] + weights[1] * max_x) / weights[2]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')  
    plt.xlabel('X1'); plt.ylabel('X2')  
    plt.show()
    print('weights: \n', weights)
    
def load_data():  
    train_x = []  
    label = []  
    fileIn = open('testset_logistic.txt')  # [x1, x2, label]
    for line in fileIn.readlines():  
        lineArr = line.strip().split()

        #               x0=1         x1                  x2
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
        label.append([float(lineArr[2])])
    
    train_x = matrix(train_x)
    label = matrix(label)
    print('train_x.shape: ', train_x.shape, ', label.shape: ', label.shape)
    return train_x, label


if __name__ == '__main__':
    ## step 1: load data  
    print("step 1: load data..."  )
    train_x, label = load_data()
    # 假
    test_x = train_x    # 100 x 3
    test_y = label    # 100 x 1

    ## step 2: training...  
    print("step 2: training..."  )
    #                                      alpha, maxIter, optimizeType = [gradDescent, stocGradDescent, smoothStocGradDescent]
    optimalWeights = train(train_x, label, 0.0069, 69, 'stocGradDescent')  
      
    ## step 3: testing  
    print("step 3: testing..."  )
    accuracy = test(optimalWeights, test_x, test_y)  
      
    ## step 4: show the result  
    print("step 4: show the result..." )
    print('The classify accuracy is: %.3f%%' % (accuracy * 100))  
    show_result(optimalWeights, train_x, label)
    