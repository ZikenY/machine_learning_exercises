# Logistic Regression Hands on exercise

import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# use pandas to read .csv file
data = pd.read_csv("./HR_comma_sep.csv")

# field "left" is the Y label: 0 - no_left; 1 - left
data.left = data.left.astype(int)

# use dmatrices to make all categorical variables to dummy variables,
# and choose features "satisfaction_level, last_evaluation, ... " to predict the Y (left),
# then reassign the column names:
y, X = dmatrices('left~satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+promotion_last_5years+C(sales)+C(salary)', data, return_type='dataframe')

# just change feature name for display right now. those names are not useful later.
X = X.rename(columns = {
    'C(sales)[T.RandD]': 'Department: Random',
    'C(sales)[T.accounting]': 'Department: Accounting',
    'C(sales)[T.hr]': 'Department: HR',
    'C(sales)[T.management]': 'Department: Management',
    'C(sales)[T.marketing]': 'Department: Marketing',
    'C(sales)[T.product_mng]': 'Department: Product_Management',
    'C(sales)[T.sales]': 'Department: Sales',
    'C(sales)[T.support]': 'Department: Support',
    'C(sales)[T.technical]': 'Department: Technical',
    'C(salary)[T.low]': 'Salary: Low',
    'C(salary)[T.medium]': 'Salary: Medium'}) 
print(X)
'''
            Intercept  Department: Random  Department: Accounting  Department: HR  \
        0            1.0                 0.0                     0.0             0.0   
        1            1.0                 0.0                     0.0             0.0   
        2            1.0                 0.0                     0.0             0.0   
'''

X = np.asmatrix(X)
# make y to numpy 1-D array
y = np.ravel(y)

print('X:', X)
print('y:', y)
'''
                    print('X:', X)

                    print('y:', y)

                    X: [[1. 0. 0. ... 3. 0. 0.]
                    [1. 0. 0. ... 6. 0. 0.]
                    [1. 0. 0. ... 4. 0. 0.]
                    ...
                    [1. 0. 0. ... 3. 0. 0.]
                    [1. 0. 0. ... 4. 0. 0.]
                    [1. 0. 0. ... 3. 0. 0.]]
                    y: [1. 1. 1. ... 1. 1. 1.]
'''

# nomalization Xs to range[0,1]
for i in range(1, X.shape[1]):
    xmin = X[:,i].min()
    xmax = X[:,i].max()
    X[:, i] = (X[:, i] - xmin) / (xmax - xmin)    # xmax -> 1; xmin -> 0

np.random.seed(1)
alpha = 1  # learning rate
beta = np.random.randn(X.shape[1])  # initial β's all components to normal distribution

iteration_count = 200
losses = list()  # collect loss for each iteration
errors = list()  # collect error rate for each iteration

# use gradient descent to minimize the cross-entropy loss
for T in range(iteration_count):
    # prob = 1 / (1 + exp(-β0*X0 - β1*X1 - β2*X2...... - βn*Xn))   
    #                             matrix_X：n*m； vector β：size=m
    # the following "1." will be expanded to m-Dimension automatically
    prob = np.array(1. / (1 + np.exp(-np.matmul(X, beta)))) # prob is the "scoring function"
#    print('prob.shape: ', prob.shape) # (1, 14999)
    prob = prob.ravel()
#    print('prob.shape(after .ravel()): ', prob.shape) # (14999,)
    
    # zip each "sample -> score" pair
    prob_y = list(zip(prob, y))   
#    print('prob_y: ', prob_y)  # [(0.8563635682985654, 1.0), (0.11839265197520915, 1.0), .......]
    
    # compute cross_entropy loss. (along each dimension)
    loss = -sum([np.log(p) if y == 1 else np.log(1 - p) for p, y in prob_y]) / len(y)
    losses.append(loss)
    
    # computer error_rate(only for observation)
    # divide_line = 0.5
    error_rate = 0
    for i in range(len(y)):
        if ((prob[i] > 0.5 and y[i] == 0) or (prob[i] <= 0.5 and y[i] == 1)):
            error_rate += 1;
    error_rate /= len(y) # wrong_y : all y
    errors.append(error_rate)
    
    if T % 5 ==0 :
        print('T=' + str(T) + ' loss=' + str(loss) + ' error=' + str(error_rate))
        
    # compute the partial derivative in terms of each component of beta.
    # X.shape[1] is feature count. we should compute such number of partial derivatives parallelly to each sample
    deriv = np.zeros(X.shape[1])
#    print('deriv.shape: ', deriv.shape)   # (19,) such number of β's derivatives

    # computer each component of β's partial derivative of all samples
    for i in range(len(y)):
        # extract the whole line (features) of sample_i
        Xi = np.asarray(X[i,:]).ravel()
        
        # Xi is a vector
        deriv += Xi * (prob[i] - y[i]) 
        
    # get average of derivatives of sample/labels
    deriv /= len(y)
    
    # descent the gradient of β (in all dimensions)
    beta -= alpha * deriv

print('Gradient Descent done.')
'''
                    T=0 loss=1.1203823278066718 error=0.5037002466831122
                    T=5 loss=0.6492666637968594 error=0.2910194012934196
                    T=10 loss=0.6095807663133693 error=0.26668444562970867
                    T=15 loss=0.5816449211566243 error=0.25888392559503964
                    T=20 loss=0.5607552377630786 error=0.2526835122341489
                        ......
'''

# observate the error curve and the loss curve
import matplotlib.pyplot as plt
plt.plot(losses)
plt.plot(errors)
plt.show()

# check the gradient
np.random.seed(1)
alpha = 1  # learning rate
beta = np.random.randn(X.shape[1]) 

#dF/dbeta0
prob = np.array(1. / (1 + np.exp(-np.matmul(X, beta)))).ravel()
prob_y = list(zip(prob, y))
loss = -sum([np.log(p) if y == 1 else np.log(1. - p) for p, y in prob_y]) / len(y) # compute the loss function
deriv = np.zeros(X.shape[1])
for i in range(len(y)):
    deriv += np.asarray(X[i,:]).ravel() * (prob[i] - y[i])
deriv /= len(y)
print('We calculated ' + str(deriv[0]))

delta = 0.0001
beta[0] += delta
prob = np.array(1. / (1 + np.exp(-np.matmul(X, beta)))).ravel()  # expect the probability of left based on beta 
prob_y = list(zip(prob, y))
loss2 = -sum([np.log(p) if y == 1 else np.log(1. - p) for p, y in prob_y]) / len(y) # compute the loss function
shouldbe = (loss2 - loss) / delta # (F(b0+delta,b1,...,bn) - F(b0,...bn)) / delta
print('According to definition of gradient, it is ' + str(shouldbe))
'''
        We calculated 0.3306924011046167
        According to definition of gradient, it is 0.330699032340398
'''