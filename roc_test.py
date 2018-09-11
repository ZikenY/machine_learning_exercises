    # -*- coding: utf-8 -*-  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import svm, datasets  
from sklearn.metrics import roc_curve, auc  ###计算roc和auc  
from sklearn import cross_validation  
  
# Import some data to play with  
iris = datasets.load_iris()  
X = iris.data  
y = iris.target  
  
##变为2分类  
X, y = X[y != 2], y[y != 2]
  
# Add noisy features to make the problem harder  
# 给随机生成器设置seed的目的是每次运行程序得到的随机数的值相同，这样方便测试。
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]  
  
# shuffle and split training and test sets  
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.3, random_state=0)

# Learn to predict each class against the other
svm = svm.SVC(kernel='linear', probability=True, random_state=random_state)

'''
SVC中的decision_function方法对每个样本都会给出在各个类别上的分数（在二元分类问题中，是对每个样本给出一个分数）。
如果构造函数的probability被设为True，则可以得到属于每个类别的概率估计（通过predict_proba和predict_log_proba方法）
'''
###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
y_score = svm.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
# 从高到低，依次将“y_score”值作为阈值threshold
fpr,tpr,threshold = roc_curve(y_test, y_score, pos_label=1, drop_intermediate=False) ###计算真正率和假正率  
roc_auc = auc(fpr,tpr) ###计算auc的值
  
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线  
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
