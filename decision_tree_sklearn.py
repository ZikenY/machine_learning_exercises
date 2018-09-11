#-*- coding:utf-8 -*-  

'''
https://nlpcs.com/article/dicision-tree-skl
pip install graphviz
pip install pydotplus
sudo apt-get install python-pydot
'''
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus
from PIL import Image
import csv

# 读取数据
data = open("play_golf.csv", "r")
reader = csv.reader(data, skipinitialspace = True, delimiter = ',', quoting = csv.QUOTE_NONE)
headers = next(reader)
print(headers)      # ['Day', 'Outlook', 'Temperature', 'Humidity', 'Wind', 'Play Golf']

# 构造 Feature 字典
featureList = [] # each element is a dictionary that represents a sample
labelList = []
for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1,len(row) - 1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)
print(featureList)  # [{'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Strong'}, {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak'}, {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak'}, {'Outlook': 'Rain', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Weak'}, {'Outlook': 'Rain', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Strong'}, {'Outlook': 'Overcast', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Strong'}, {'Outlook': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak'}, {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Weak'}, {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'Normal', 'Wind': 'Weak'}, {'Outlook': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'Normal', 'Wind': 'Strong'}, {'Outlook': 'Overcast', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Strong'}, {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'Normal', 'Wind': 'Weak'}, {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Strong'}]
print(str(labelList))

# Feature 标准化 (convert each feature into one-hot encoding)
vec = DictVectorizer()
X = vec.fit_transform(featureList).toarray()
print(vec.get_feature_names())
print(str(X))
'''
   Humidity(High) Humidity(Normal) Outlook(Overcast)  Outlook(Rain) Outlook(Sunny) Temperature(Cool) Temperature(Hot) Temperature(Mild) Wind(Strong)    Wind(Weak)
[[      1.              0.              0.                  0.          1.              0.                  1.              0.              0.              1.]
 [      1.              0.              0.                  0.          1.              0.                  1.              0.              1.              0.]...]
'''

# Label 标准化 (nx1 binary #)
lb = preprocessing.LabelBinarizer()
Y = lb.fit_transform(labelList)
print(Y) # [[0] [0] [1] [1] [1] [0] [1] [0] [1] [1] [1] [1] [1]]

# 训练模型
clf = tree.DecisionTreeClassifier(criterion = "entropy")
clf = clf.fit(X, Y)
print(str(clf))

# 模型写入文件
# with open("play_golf.dot", "w") as f:
#     f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file = f)

# draw the decision tree
dot_data = tree.export_graphviz(
    clf,
    feature_names   = vec.get_feature_names(),
#    class_names     = target_names, 
    out_file        = None,
    filled          = True,
    rounded         = True,
    special_characters = True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_jpg('play_golf.jpg')
img = Image.open('play_golf.jpg')
img.show()

# 将原始数据第一行稍作修改, 创建新的数据，用刚才生成的模型做预测
newRowX = X[0, :]
newRowX[0] = 0
newRowX[1] = 1
print('new row: ', newRowX)
predictedY = clf.predict(newRowX.reshape(1, -1))
print('prediction: ', predictedY)
