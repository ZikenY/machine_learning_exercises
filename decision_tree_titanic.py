import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(os.path.join("./", "titanic.csv"), sep=",")
data.info()
data.head(13)

# the basic distribution of labels
data['survived'].value_counts(normalize = True)
sns.countplot(data['survived'])
plt.show()

# analysis the relations of fields and the label
sns.countplot(data['pclass'], hue = data['survived'])
plt.show()

# manipulate the field 'name'
data['name'].head()
'''
0                       Allen, Miss Elisabeth Walton
1                        Allison, Miss Helen Loraine
2                Allison, Mr Hudson Joshua Creighton
3    Allison, Mrs Hudson J.C. (Bessie Waldo Daniels)
4                      Allison, Master Hudson Trevor
'''

# get title
# create a new feature: 'name_title' 
data['name_title'] = data['name'].apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0]).apply(lambda x: x.split()[0])
data['name_title'].value_counts()
# relations of 'name_title' and 'survived'
data['survived'].groupby(data['name_title']).mean()

# get length of the name
# create a new numerical feature
data['name_len'] = data['name'].apply(lambda x: len(x))
# pd.qcut(['field'], 5) split the range of 'field' value into 5 sections
data['survived'].groupby(pd.qcut(data['name_len'], 5)).mean()
'''
name_len
(10.999, 17.0]    0.183746
(17.0, 20.0]      0.288321
(20.0, 24.0]      0.285171
(24.0, 28.0]      0.369295
(28.0, 62.0]      0.611111
'''

data['sex'].value_counts(normalize=True)
data['survived'].groupby(data['sex']).mean()

data['survived'].groupby(pd.qcut(data['age'], 5)).mean()

data['embarked'].value_counts()
data['survived'].groupby(data['embarked']).mean()

# destination: extract the last part of the destination field by splitting by ','
data['survived'].groupby(data['home.dest'].apply(lambda x: str(x).split(',')[-1])).mean()

# drop room, ticket, boat

'''
    -------- Feature Transform --------
'''
def name(data):
    data['name_len'] = data['name'].apply(lambda x: len(x))
    data['name_title'] = data['name'].apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0]).apply(lambda x: x.split()[0])
    del data['name']
    return data

def age(data):
    data['age_flag'] = data['age'].apply(lambda x: 1 if pd.isnull(x) else 0)
    grouped_age = data.groupby(['name_title', 'pclass'])['age']
    data['age'] = grouped_age.transform(lambda x: x.fillna(data['age'].mean()) if pd.isnull(x.mean()) else x.fillna(x.mean()))
    return data

def embark(data):
    data['embarked'] = data['embarked'].fillna('Southampton')
    return data


def dummies(data, columns=['pclass','name_title','embarked', 'sex']):
    for col in columns:
        data[col] = data[col].apply(lambda x: str(x))
        new_cols = [col + '_' + i for i in data[col].unique()]
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)[new_cols]], axis=1)
        del data[col]
    return data

# preprocess the raw data
drop_columns = ['row.names', 'home.dest', 'room', 'ticket', 'boat'] #+ ['ticket_len', 'ticket_title']
data = data.drop(drop_columns, axis=1)
data.head()

# Feature Transform
data = name(data)
data = age(data)
data = embark(data)
data = dummies(data)
data.head()

# use model to train the data
from sklearn.model_selection import train_test_split
from sklearn import tree
trainX, testX, trainY, testY = train_test_split(data.iloc[:,1:], data.iloc[:,0], test_size=0.2, random_state=33)

model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
model.fit(trainX, trainY)

# estimate the model:
from sklearn import metrics
def measure_performance(X, y, clf, show_accuracy=True, show_classification_report=True, show_confussion_matrix=True):
    y_pred = clf.predict(X)
    if show_accuracy:
        print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y, y_pred)),"\n")
    
    if show_classification_report:
        print("Classification report")
        print(metrics.classification_report(y, y_pred), "\n")
    
    if show_confussion_matrix:
        print("Confusion matrix")
        print(metrics.confusion_matrix(y, y_pred), "\n")

# apply the trained model on test_set
measure_performance(testX, testY, model)

# visualization
import graphviz
dot_data = tree.export_graphviz(model, out_file=None, feature_names=trainX.columns) 
graph = graphviz.Source(dot_data) 
graph.render("titanic") 
graph.view()