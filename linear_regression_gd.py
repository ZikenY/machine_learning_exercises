# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
linear regression - gradient descent solutions

http://www.cnblogs.com/Sinte-Beuve/p/6164689.html
------------------------------------------------------------------------------
'''

'''
    batch gradient descent
    x:          mxd  (m samples, d features each)
    y:          mx1  label
    
    return -
    theta:      1xd
'''
def batch_gradient_descent(x, y, alpha = 0.001, max_iter = 50000):
    m = len(x)      # sample count
    d = len(x[0])   # feature count
    theta = [0 for _ in range(d)] # zeros(d)
    iter = 0
    while iter < max_iter:
        err = 0
        gradient = [0 for _ in range(d)]
        
        for i in range(m):
            # predicted h := theta[0] * x[j][0] + theta[1] * x[j][1] *...
            h = 0
            for j in range(d):
                h += theta[j] * x[i][j]

            # compute gradent' for sample i and accumulate it on each components of gradient (gradient[j])
            for j in range(d):
                gradient[j] += (h - y[i])*x[i][j]
            
        # update weights from gradients
        for j in range(d):
#            theta[j] -= alpha * gradient[j] / m
            theta[j] -= alpha * gradient[j]
        
        #计算误差
        # err = err + (y[i] - (theta[0] * x[i][0] + theta[1] * x[i][1]) +...) ** 2
        for i in range(m):
            e = y[i]
            for j in range(d):
                e -= theta[j] * x[i][j]                
            err += e**2
                
        iter += 1        
        if err < 0.0001:
            break
        
    print('\n----- batch gradient descent ------')
    print ('iteration count: ', iter)
    print ('error: ', "{0:.10f}%".format(err))
    return theta



'''
    stocastic gradient descent，每一次梯度下降只使用一个样本。
    x:          mxd  (m samples, d features each)
    y:          mx1  label
    
    return -
    theta:      1xd
'''
def stochastic_gradient_descent(x, y, alpha = 0.001, max_iter = 50000):
    m = len(x)      # sample count
    d = len(x[0])   # feature count
    theta = [0 for _ in range(d)] # zeros(d)
    
    epoc = 0
    iter = 0

    while iter < max_iter:
        # 循环取训练集中的一个
        for i in range(m):
            # predicted h := theta[0] * x[j][0] + theta[1] * x[j][1] *...
            h = 0
            for j in range(d):
                h += theta[j] * x[i][j]

            # update weights from gradients
            for j in range(d):
                gradient = (h - y[i]) * x[i][j]
                theta[j] -= alpha * gradient

            iter += 1
        epoc += 1
        
        #计算误差
        err = 0
        for ii in range(m):
            # err += (y[ii] - (theta[0] * x[ii][0] + theta[1] * x[ii][1] + ...)) ** 2
            e = y[ii]
            for j in range(d):
                e -= theta[j] * x[ii][j]
            err += e**2
            
        if err < 0.0001:
            break
        
    print('\n----- stocastic gradient descent ------')
    print ('iteration count: ', iter)
    print ('epoc count: ', epoc)
    print ('error: ', "{0:.10f}%".format(err))
    return theta


if __name__ == '__main__':
    #     b     x1      x2      x3
    x = [[1,    2.21,   1.5,    2.1],
         [1,    2.5,    2.3,    2.5],
         [1,    3.369,  3.9,    3.3],
         [1,    3.9,    5.1,    3.9],
         [1,    2.71,   2.7,    2.70169]]

    labels = [2.5,
              3.9,
              6.701,
              8.8,
              4.6]
    
    print('\samples: ', x)
    print('\labels: ', labels)
    
    # batch gradient descent
    resultTheta = batch_gradient_descent(x, labels)
    print('theta = ', resultTheta)
    
    # stocastic gradient descent
    resultTheta = stochastic_gradient_descent(x, labels)
    print('theta = ', resultTheta)
