'''
    use MLP to predict the classifications of otto commodities.
'''
import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv('./otto_train.csv')

print(data.head())
'''
       id  feat_1  feat_2  feat_3  feat_4  feat_5  feat_6  feat_7  feat_8  feat_9 
    0   1       1       0       0       0       0       0       0       0       0   
    1   2       0       0       0       0       0       0       0       1       0   
    2   3       0       0       0       0       0       0       0       1       0   

        ...
    [5 rows x 95 columns]
'''
# print(data.dtypes)

# get sample features from training_set
columns = data.columns[1:-1]
X = data[columns]
print('feature count: ', X.shape[1])

# the labels of training set:
y = np.ravel(data['target'])


# commodity class distributions
groupby_target = data.groupby('target')
print('target count: ', len(groupby_target.size())) # 9 classes in total
# print('data.groupby(''target''): ', groupby_target.head())
print('sample count of each', groupby_target.size()) # show sample count of each target
print('sample count in total: ', data.shape[0])      # 61878

target_distribution = groupby_target.size() / data.shape[0] * 100
target_distribution.plot(kind = 'bar')
plt.show()


# show the distribution of feature_20 in terms of different target_class
for class_id in range(0, 8):
    plt.subplot(3, 3, class_id+1) 
#    plt.axis('off') 
    feat_20 = data[data.target == 'Class_' + str(class_id + 1)].feat_20
    # 在canvas中画这个sub-figure
    feat_20.hist()
plt.show()


# these twe features is negative correlated
plt.scatter(data.feat_19, data.feat_20)
plt.show()


# print(X.corr) # too many numbers
# use heatmap to display the correlations among all feature pairs
fig = plt.figure()
# 1 row, 1 col, select the 1st subplot
ax = fig.add_subplot(111)
# show the correlation heatmap of each pair of features in X
cax = ax.matshow(X.corr(), interpolation = 'nearest')


''' 
    ------  initialize the MLP model，two hidden_layers，neurons: 93x30x10x9 --------
'''
num_fea = X.shape[1]
num_target = len(groupby_target.size())
print('feature count: ', num_fea)       # 93
print('target count: ', num_target)     # 9

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(solver='lbfgs',                 # an optimizer in the family of quasi-Newton methods.
                      activation = 'logistic',        # sigmoid activation
                      alpha=1e-5,                     # L2 penalty (regularization term) parameter
                      hidden_layer_sizes=(30,10),
                      random_state = 1, verbose = True)

# train the model on training_set(X, y):
model.fit(X, y)
'''
    MLPClassifier(activation='logistic', alpha=1e-05, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(30, 10), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=True,
           warm_start=False)
'''


# check the shape of coefficients (matrix of 93x30) between the first two layer:
print('shape of coefficients (matrix of 93x30) between the first two layer: ', model.coefs_[0].shape)

# bias parameters are stored in .intercepts_:
# the first hidden layer has 30 neurons, so the first group of biases has size of 30
print(model.intercepts_)

print(model.coefs_[0].shape)
print(model.coefs_[1].shape)
print(model.coefs_[2].shape)
print('total number of trainable variables: 93*30+30 + 30*10+10 + 10*9+9')
'''
    shape of coefficients (matrix of 93x30) between the first two layer:  (93, 30)
    [array([ 1.75840637,  1.1861953 ,  0.82721995,  1.47077122, -0.25715867,
           -1.81896469,  0.05835565,  0.16012998,  0.1403171 , -1.28304708,
            2.03083143,  0.30136014,  0.87467449, -0.33687858,  1.76013227,
            0.27830667, -0.13992491, -1.43643067,  0.41312864, -0.66130014,
           -0.29195014, -0.40472893, -1.03878953, -0.37846876, -2.1940681 ,
           -0.42545847, -2.1588057 ,  1.38087307, -2.35314261,  0.06821877]), array([-0.59899976, -0.25412398, -1.10871783, -0.28252679, -0.04739503,
           -1.31390536, -0.1204834 , -0.41982622,  0.25399996,  0.00742948]), array([ 0.68013743, -2.06294166,  0.2871028 , -0.50630787,  0.20878166,
           -1.31262218,  3.93079569, -1.33410495,  0.39696964])]
    (93, 30)
    (30, 10)
    (10, 9)
    total number of trainable variables: 93*30+30 + 30*10+10 + 10*9+9
'''

# just playing: predict X using this model
accuracy_X = model.score(X, y)
print('accuracy on training_set X: ', accuracy_X)
# or
pred = model.predict(X)
print('accuracy on training_set X: ', sum(pred == y) / len(y))



'''
    -----------  predict the solution on test_set.  -------------
'''
test_data = pd.read_csv('./otto_test.csv')
X_test = test_data[test_data.columns[1:]]

# the goal is to get the prediction probabilitiy for each sample in test_set
test_proba = model.predict_proba(X_test)

# print(np.sum(test_proba, axis = 1))  the sum of each row should be 1 !!!!

# output the prediction solution:
# step 1. using the test_proba to build a dataframe with the total 9 columns of classes but without the id column
solution = pd.DataFrame(test_proba, columns=['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])
print('shape of submit_dataframe without id column:', solution.shape)

# step 2. append a id_column on the tail of the dataframe.
#         each row in test_proba is corresponding to the same row in test_data
solution['id'] = test_data['id']

# step 3. get all columns, and move the last column to the head of columns
cols = solution.columns.tolist()
cols = cols[-1:] + cols[:-1]

# step 4. use the new columns to rebuild the dataframe
solution = solution[cols]
print('shape of final submit_dataframe:', solution.shape)
print(solution.head())

# save the dataframe to a .csv
solution.to_csv('./otto_prediction.tsv', index = False)

