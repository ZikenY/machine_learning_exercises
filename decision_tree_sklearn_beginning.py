import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

'''
create 2D grid sample:
        a = [1 2 3]
        b = [2 3 4]
        x, y = meshgrid(a,b)
        [out:]  x = 1 2 3
                    1 2 3
                    1 2 3

                y = 2 2 2
                    3 3 3
                    4 4 4
[X,Y]=meshgrid(x)   <====>  [X,Y]=meshgrid(x,x)
'''

'''
input X:      data of 2 columns to be classified to 2 groups
input model:  a classifier
e.g. 
    X: 500 rows, 2 cols:
    [[ 2.8315865   2.21527897]
     [-0.04540029  1.49161615]
     [ 2.12133597  0.77991444]
     ...]
'''
def plot_decision_boundary(X, model):
    h = .02
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    print('x_min, x_max: ', x_min, x_max)  # -5.31600121715101 5.179910307979315

    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    print('y_min, y_max: ', y_min, y_max)  # -5.011469099057462 4.967651056434603
    
    a_ = np.arange(x_min, x_max, h)
    print('a_.shape', a_.shape)            # (525,)

    b_ = np.arange(y_min, y_max, h)
    print('b_.shape', b_.shape)            # (499,)

    xx, yy = np.meshgrid(a_, b_)
    print('xx.shape: ', xx.shape, 'yy.shape', yy.shape) # (499, 525)
    print('xx.ravel().shape: ', xx.ravel().shape)   # (261975,)
    
    # xx.ravel(), yy.ravel() flatten the twos
    # np.r_   =====> pandas中的concat()。
    # np.c_   =====> pandas中的merge()。
    to_be_predicted = np.c_[xx.ravel(), yy.ravel()]
    print('to_be_predicted.shape: ', to_be_predicted.shape) # (261975, 2)
    
    z = model.predict(to_be_predicted)
    print('z.shape before reshape: ', z.shape)           # (261975,)
    z = z.reshape(xx.shape)
    print('z.shape after reshape(xx.shape): ', z.shape)  # (499, 525)
    
    plt.contour(xx, yy, z, cmap=plt.cm.Paired)


'''
    training set of linear separatable samples
'''
np.random.seed(11)
N = 500
D = 2
# create N samples with D-dimension each
X = np.random.randn(N, D) # 500 * (x,y)

# Y is a array of 250 0s and 250 1s
Y = np.array([0] * (N//2) + [1] * (N//2))

# shift the samples into 2 groups
delta = 1.5
X[:N//2] += np.array([delta, delta]) 
X[N//2:] += np.array([-delta, -delta])

plt.scatter(X[:,0], X[:,1], s=130, c=Y, alpha=0.5)
plt.show()


'''
    baseline decision tree
'''
from sklearn.tree import DecisionTreeClassifier
# model: decision tree classifier
model = DecisionTreeClassifier(criterion='gini')
# train this model
model.fit(X, Y)
# score on training_set. 
score = model.score(X, Y)
print("score for basic tree:", score) # score = 1.0 (overfitting)

# draw the scatter points as background
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
# draw the boundary
plot_decision_boundary(X, model)
plt.show()


'''
    limit the max tree depth
'''
model_depth_3 = DecisionTreeClassifier(criterion='entropy', max_depth = 3)
model_depth_3.fit(X,Y)
print("score for basic tree:", model_depth_3.score(X, Y))   # score = 0.986

plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_3)
plt.show()


'''
    non-linear separatable samples - XOR
'''
np.random.seed(10)

N = 500
D = 2
X = np.random.randn(N, D)

delta = 1.75
X[:125] += np.array([delta, delta])
X[125:250] += np.array([delta, -delta])
X[250:375] += np.array([-delta, delta])
X[375:] += np.array([-delta, -delta])
Y = np.array([0] * 125 + [1]*125 + [1]*125 + [0] * 125)

plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()

model.fit(X, Y)
model_depth_3.fit(X, Y)


print("score for basic tree:", model.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model)
plt.show()


print("score for basic tree:", model_depth_3.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_3)
plt.show()



'''
    non-linear separatable samples - Donut
'''
np.random.seed(10)

N = 500
D = 2
X = np.random.randn(N, D)

R_smaller = 5
R_larger = 10

R1 = np.random.randn(N//2) + R_smaller
theta = 2 * np.pi * np.random.random(N//2)
X[:250] = np.concatenate([[R1 * np.cos(theta)], [R1*np.sin(theta)]]).T


R2 = np.random.randn(N//2) + R_larger
theta = 2 * np.pi * np.random.random(N//2)
X[250:] = np.concatenate([[R2 * np.cos(theta)], [R2*np.sin(theta)]]).T

Y = np.array([0] * (N//2) + [1] * (N//2))

plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()


model.fit(X, Y)
model_depth_3.fit(X, Y)


print("score for basic tree:", model.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model)
plt.show()


print("score for basic tree:", model_depth_3.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model_depth_3)
plt.show()


