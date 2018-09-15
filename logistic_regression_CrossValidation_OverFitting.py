''' --------------------------------------------
    Logistic Regression Cross_validation, overfitting test
'''

import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# read data using pandas
data = pd.read_csv("./HR_comma_sep.csv")

# all samples, lables in X, y
y, X = dmatrices('left~satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+promotion_last_5years+C(sales)+C(salary)', data, return_type='dataframe')
X = np.asmatrix(X)
y = np.ravel(y)

''' ---------------------------------------
    normalize all feature values into range[0,1]
'''
# normalize all feature values into [0, 1]
# X.shape[1]: feature count
for i in range(1, X.shape[1]):
    xmin = X[:,i].min() # get minimal value from all row for column i(feature i)
    xmax = X[:,i].max()
    X[:, i] = (X[:, i] - xmin) / (xmax - xmin)


''' ---------------------------------------
    logistic regression，gradient descent by hand
'''
np.random.seed(1)
alpha = 1  # learning rate
beta = np.random.randn(X.shape[1]) # initial β's all components to normal distribution
iteration_count = 200
losses = list()  # collect loss for each iteration
errors = list()  # collect error rate for each iteration

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
T=5 loss=0.6492666637968592 error=0.2910194012934196
T=10 loss=0.6095807663133694 error=0.26668444562970867
T=15 loss=0.5816449211566243 error=0.25888392559503964
            ......
'''



''' -------------------------------------------------------
    Split X, y into training_set and validation_set as 7:3
'''
# sk-learn dataset split, cross validation
from sklearn.model_selection import train_test_split, cross_val_score 

# training_set/testing_set : 7:3
Xtrain, Xvali, ytrain, yvali = train_test_split(X, y, test_size=0.3, random_state=0)
model_LR = LogisticRegression()
model_LR.fit(Xtrain, ytrain)

# get prediction of validation_set
from sklearn import metrics
pred = model_LR.predict(Xvali)
print('accuracy score: ', metrics.accuracy_score(yvali, pred))
print('confusion matrix: \n', metrics.confusion_matrix(yvali, pred))
'''
accuracy score:  0.7926666666666666
confusion matrix: 
 [[3208  254]
 [ 679  359]]
'''


''' -------------------------------------------------------
Cross Validation for cv = 10
'''
# cross_validation (mean of 10 times):
C = 1e1
accuracy_score = cross_val_score(LogisticRegression(C = C, penalty = 'l2'), X, y, scoring = 'accuracy', cv = 10).mean()
print('accuracy_score: ', accuracy_score, ' for C = ', C)
# accuracy_score:  0.7861785463015761  for C =  10.0

C = 1e3
accuracy_score = cross_val_score(LogisticRegression(C = C), X, y, scoring = 'accuracy', cv = 10).mean()
print('accuracy_score: ', accuracy_score, ' for C = ', C)
# accuracy_score:  0.7861118796349095  for C =  1000.0

C = 1e0
accuracy_score = cross_val_score(LogisticRegression(C = C), X, y, scoring = 'accuracy', cv = 10).mean()
print('accuracy_score: ', accuracy_score, ' for C = ', C)
# accuracy_score:  0.7849117017385341  for C =  1.0

C = 0.001
accuracy_score = cross_val_score(LogisticRegression(C = C), X, y, scoring = 'accuracy', cv = 10).mean()
print('accuracy_score: ', accuracy_score, ' for C = ', C)
# accuracy_score:  0.7619174793411018  for C =  0.001

# ----  Finally, C = 1e3 has choosen.   ----
# ----  use this hyper-parameter to fit all data to build model !!! ----
model_LR = LogisticRegression(C=1e3)
model_LR.fit(X, y)



''' -------------------------------------------------------
overfitting test
     for error_rate comparsion between training_set and validation_set:
'''
# training_set/testing_set : 8:2
Xtrain, Xvali, ytrain, yvali = train_test_split(X, y, test_size=0.2, random_state=0)

np.random.seed(1)
alpha = 5 # learning rate
beta = np.random.randn(Xtrain.shape[1])
error_rates_train=[]
error_rates_vali=[]
for T in range(200):
    prob = np.array(1. / (1 + np.exp(-np.matmul(Xtrain, beta)))).ravel()  # predict probability of left using current BETA
    prob_y = list(zip(prob, ytrain))
    loss = -sum([np.log(p) if y == 1 else np.log(1 - p) for p, y in prob_y]) / len(ytrain) # compute the loss function
    error_rate = 0
    for i in range(len(ytrain)):
        if ((prob[i] > 0.5 and ytrain[i] == 0) or (prob[i] <= 0.5 and ytrain[i] == 1)):
            error_rate += 1;
    error_rate /= len(ytrain)
    error_rates_train.append(error_rate)
    
    prob_vali = np.array(1. / (1 + np.exp(-np.matmul(Xvali, beta)))).ravel()  # predict the probability of left on validation_set
    prob_y_vali = list(zip(prob_vali, yvali))
    loss_vali = -sum([np.log(p) if y == 1 else np.log(1 - p) for p, y in prob_y_vali]) / len(yvali) # compute the loss function
    error_rate_vali = 0
    for i in range(len(yvali)):
        if ((prob_vali[i] > 0.5 and yvali[i] == 0) or (prob_vali[i] <= 0.5 and yvali[i] == 1)):
            error_rate_vali += 1
    error_rate_vali /= len(yvali)
    error_rates_vali.append(error_rate_vali)
    
    if T % 5 ==0 :
        print('T=' + str(T) + ' loss=' + str(loss) + ' error=' + str(error_rate)+ ' error_vali=' + str(error_rate_vali))
    
    # compute dLoss/dBetas
    deriv = np.zeros(Xtrain.shape[1])
    for i in range(len(ytrain)):
        deriv += np.asarray(Xtrain[i,:]).ravel() * (prob[i] - ytrain[i])
    deriv /= len(ytrain)
    
    # gradient descent on vector_beta
    beta -= alpha * deriv
'''
T=0 loss=1.135137140973581 error=0.5020478140775312
T=5 loss=0.6444185582182453 error=0.29631393466044387
T=10 loss=0.6030253387203952 error=0.27107343556529195
        ......
'''

# draw the two curves of error_rate from training_set and validation_set simultaneously.
plt.plot(range(50,200), error_rates_train[50:], 'r^', range(50, 200), error_rates_vali[50:], 'bs')
plt.show()  # overfitting show


''' -------------------------------------------------------
    check the gradient value:
'''
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
