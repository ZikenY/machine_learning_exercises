import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

from scipy.stats import skew
from scipy.stats import norm
from scipy.special import boxcox1p, inv_boxcox

from datetime import date, datetime
import time

'''
    check for incomplete data
'''
train.isnull().values.any()
print(train.isnull().sum().sum())
print(test.isnull().sum().sum())
print(train.isna().sum().sum())
print(test.isna().sum().sum())

'''
    check data
'''
train.head()
'''
        datetime        season  holiday     workingday  weather     temp    atemp   humidity    windspeed   casual  registered  count
0   2011-01-01 00:00:00     1       0           0           1       9.84    14.395      81          0.0         3       13        16
...
...
'''

train.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8600 entries, 0 to 8599
Data columns (total 12 columns):
datetime      8600 non-null object
season        8600 non-null int64
holiday       8600 non-null int64
workingday    8600 non-null int64
weather       8600 non-null int64
temp          8600 non-null float64
atemp         8600 non-null float64
humidity      8600 non-null int64
windspeed     8600 non-null float64
casual        8600 non-null int64
registered    8600 non-null int64
count         8600 non-null int64
dtypes: float64(3), int64(8), object(1)
memory usage: 806.3+ KB
'''

train.shape # (8600, 12)

train.describe()
'''
season  holiday     workingday  weather     temp    atemp   humidity    windspeed   casual  registered  count
count   8600.000000     8600.000000     8600.000000     8600.00000  8600.000000     8600.000000     8600.000000     8600.000000     8600.000000     8600.000000     8600.000000
mean    2.505581    0.027791    0.682558    1.41000     20.119653   23.560989   61.590581   12.756693   35.689419   154.840814  190.530233
std     1.116628    0.164382    0.465508    0.63234     8.000975    8.690173    19.468443   8.209822    49.571896   150.760096  180.631042
min     1.000000    0.000000    0.000000    1.00000     0.820000    0.760000    0.000000    0.000000    0.000000    0.000000    1.000000
25%     2.000000    0.000000    0.000000    1.00000     13.940000   16.665000   46.000000   7.001500    4.000000    36.000000   41.000000
50%     3.000000    0.000000    1.000000    1.00000     20.500000   24.240000   61.000000   11.001400   16.000000   118.000000  144.000000
75%     4.000000    0.000000    1.000000    2.00000     27.060000   31.060000   78.000000   16.997900   48.000000   221.000000  282.000000
max     4.000000    1.000000    1.000000    4.00000     41.000000   45.455000   100.000000  56.996900   362.000000  886.000000  977.000000
'''


'''
    construct new features
'''
# datatime  ---> time, data, month, day, year, weekday
def transform_time(d):
    # datetime: '2011-01-01 03:00:00'
    d['time'] = d['datetime'].apply(lambda x: x[11:13])   #  (hour)
    d['date'] = d['datetime'].apply(lambda x: x[:10]) 
    d['month'] = d['date'].apply(lambda s: s[5:7])
    d['day'] = d['date'].apply(lambda s: s[8:10])
    d['year'] = d['date'].apply(lambda s: s[:4])
    
    # from datetime import date, datetime
    d['weekday'] = d['date'].apply(lambda s: date(*(int(i) for i in s.split('-'))).weekday() + 1)
    
transform_time(train)
transform_time(test)


'''
    full data:
'''
# step1. 
train_ = train[test.columns]
# step2. 
data = pd.concat([train_, test]).reset_index(drop = True)
print(train.shape, test.shape, data.shape)  # (8600, 18) (2286, 15) (10886, 15)

train.corr()
'''
season  holiday     workingday  weather     temp    atemp   humidity    windspeed   casual  registered  count   weekday
season  1.000000    0.091338    -0.030821   0.011746    0.291688    0.302915    0.214066    -0.157129   0.128356    0.171616    0.178462    -0.005030
holiday     0.091338    1.000000    -0.247918   -0.011177   0.041898    0.037992    0.025504    -0.007389   0.072915    -0.011759   0.010196    -0.170668
workingday  -0.030821   -0.247918   1.000000    0.017897    0.003680    -0.000643   -0.052684   0.032801    -0.322267   0.117110    0.009302    -0.712280
weather     0.011746    -0.011177   0.017897    1.000000    -0.042833   -0.042787   0.402360    0.017854    -0.125429   -0.109038   -0.125429   -0.036019
temp    0.291688    0.041898    0.003680    -0.042833   1.000000    0.992421    -0.058802   -0.012702   0.475617    0.320176    0.397756    -0.026336
atemp   0.302915    0.037992    -0.000643   -0.042787   0.992421    1.000000    -0.038796   -0.052349   0.475912    0.320004    0.397693    -0.024141
'''
# check relation of count and other features
plt.subplots(figsize = (16, 16))
corr = train.corr()
sns.heatmap(corr, square=True, annot=True, fmt='.2f')

'''
   non-linear dependency
'''
from sklearn.feature_selection import mutual_info_regression
temp_train = train.drop(['datetime', 'date'], axis=1) 
mutual_res = mutual_info_regression(temp_train, train['count'])

pd.Series(mutual_res, index=temp_train.columns + "~count").sort_values(ascending=False)
'''
count~count         6.110255
registered~count    2.261215
casual~count        0.747783
time~count          0.640007
temp~count          0.154315
atemp~count         0.142147
humidity~count      0.097074
month~count         0.084706
season~count        0.064218
year~count          0.048421
workingday~count    0.026363
windspeed~count     0.019257
weather~count       0.018287
weekday~count       0.011865
day~count           0.000000
holiday~count       0.000000
dtype: float64
'''
# check count's distribution:
sns.distplot(train['count'], fit = norm) 


'''
    try Box-Cox, lambda=0.37 
'''
from scipy.stats import skew
from scipy.stats import norm

print(skew(train['count'])) # 1.239，

# lower skew:   np.log1p(train['count'])
lg1p = np.log(train['count'] + 1)   
sns.distplot(lg1p) 
print(skew(lg1p))  # 0.84

# use boxcox to lower skew
from scipy.special import boxcox1p, inv_boxcox
lambda_ = 0.37
unskew_count = boxcox1p(train.loc[:, 'count'], lambda_)
sns.distplot(unskew_count, fit = norm)


print(skew(train['count']), skew(np.log2(train['count'] + 1)), skew(boxcox1p(train.loc[:, 'count'], 0.37)), sep='\t')
#       1.239241376322606     -0.8494341585491002                0.003099442136182699

# replace labels
train['count'] = np.log2(train['count'] + 1)
# train['count'] = unskew_count  # unskew_count = boxcox1p(train.loc[:, 'count'], lambda_)

sns.distplot(train['count'], fit=norm)

'''
   how to computer the skew?
'''
#  mean(((each_value - mean) / std)^3)
d = train['count']
print((((d - d.mean()) / d.std()) ** 3).mean())
# output: -0.8492860057353784

d = train['windspeed']
print((((d - d.mean()) / d.std()) ** 3).mean())
# output: 0.5802734147871008

'''
    analysis numerical features
'''
sns.heatmap(train[['temp', 'atemp', 'humidity', 'windspeed', 'count']].corr(), fmt='.4f', annot=True, square=True)

# check skew for this features:
value_feature = ['temp', 'atemp', 'humidity', 'windspeed']
print(skew(train[value_feature]))
# array([ 0.01835481, -0.09116029, -0.08359535,  0.58037464])

sns.distplot(train['windspeed'])

plt.scatter(data['atemp'], data['temp'])
plt.xlabel('atemp')
plt.ylabel('temp')

'''
    remove the noise points
'''
difference = (data['temp'] - data['atemp']).sort_values(ascending = False)[:30]
print(difference)
'''
        10444    23.140
        10445    23.140
        10446    22.320
        10443    22.320
        10442    21.500
        10448    21.500
        10447    21.500
        10441    19.040
        10449    18.220
        10440    18.220
        10450    17.400
        10439    16.580
        10438    15.760
        10430    15.760
        10451    15.760
        10432    14.940
        10431    14.940
        10452    14.940
        10433    14.120
        10434    14.120
        10435    14.120
        10453    14.120
        10437    14.120
        10436    13.300
        4920      1.950
        4905      1.950
        4912      1.825
'''
#  loc index=10420:10460，drow them
plt.subplot(1, 2, 1) 
data.loc[10420:10460, 'atemp'].plot()
plt.title('atemp')
plt.subplot(1, 2, 2)
data.loc[10420:10460, 'temp'].plot()
plt.title('temp')
plt.show()

'''
clean atemp 
'''
# difference = (data['temp'] - data['atemp']).sort_values(ascending=False)[:30]
sel = difference[:24].index
data.loc[sel, 'atemp'] = data.loc[sel, 'temp']
plt.scatter(data['atemp'], data['temp'])


'''
    use boxplot to show categorical features
'''
# do not do this：plt.scatter('season', 'count', data=train)

sns.boxplot('season', 'count', data = train)
'''
check Quartiles mainly
'''
sns.boxplot(train['time'], train['count'])
sns.boxplot(train['weekday'], train['count'])
sns.boxplot(train['holiday'], train['count'])

'''
check the relations between weekday, wokingday, holiday
pd.crosstab() 
'''
A = data.loc[:, 'holiday'] 
B = data.loc[:, 'weekday'] 
pd.crosstab(A, B)          
'''
        weekday     1       2       3       4       5       6       7
        holiday                             
                0   1312    1539    1527    1553    1481    1584    1579
                1   239     0       24      0       48      0       0
'''

A = data.loc[data['workingday'] == 0, 'holiday'] #  holiday | workingday==0
B = data.loc[data['workingday'] == 0, 'weekday'] #  weekday | workingday==0
# print(A.shape, B.shape)   :(3474,) (3474,)
pd.crosstab(A, B)
'''
        weekday     1   3   5   6       7
        holiday                     
                0   0   0   0   1584    1579
                1   239 24  48  0        0
'''

pd.crosstab(data.loc[data['workingday'] == 1, 'holiday'], data.loc[data['workingday'] == 1, 'weekday'])
'''
        weekday     1       2       3       4       5
        holiday                     
                0   1312    1539    1527    1553    1481
'''
'''
    conclusion: 
    workingday=(holidayis0)∩(weekdayin{1,2,3,4,5})workingday=(holidayis0)∩(weekdayin{1,2,3,4,5})
'''


plt.subplots(figsize=(15,3))
plt.subplot(1, 3, 1)   
sns.boxplot(train['season'], np.log(train['count']))
plt.subplot(1, 3, 2)
sns.boxplot(train['month'], train['count'])
plt.subplot(1, 3, 3)
sns.boxplot(train['day'], train['count'])
# 年影响大
sns.boxplot(train['year'], train['count'])


'''
    remove this 4 features, 
'''
# data.drop(['day', 'date', 'datetime', 'atemp'], axis=1, inplace=True) 
data = data.drop(['day', 'date', 'datetime', 'atemp'], axis = 1) 
data.info()
'''
RangeIndex: 10886 entries, 0 to 10885
Data columns (total 11 columns):
season        10886 non-null int64
holiday       10886 non-null int64
workingday    10886 non-null int64
weather       10886 non-null int64
temp          10886 non-null float64
humidity      10886 non-null int64
windspeed     10886 non-null float64
time          10886 non-null object
month         10886 non-null object
year          10886 non-null object
weekday       10886 non-null int64
dtypes: float64(2), int64(6), object(3)
'''

'''
    one-hot
'''
class_feature = ['weather', 'time', 'weekday', 'month', 'year', 'season']

# 做one-hot encoding
# data = pd.get_dummies(data, columns = class_feature)
data = pd.get_dummies(data, columns = class_feature, drop_first = True)
data.head()
'''
holiday     workingday  temp    humidity    windspeed   weather_2   weather_3   weather_4   time_01     time_02     ...     month_07    month_08    month_09    month_10    month_11    month_12    year_2012   season_2    season_3    season_4
0   0   0   9.84    81  0.0     0   0   0   0   0   ...     0   0   0   0   0   0   0   0   0   0
1   0   0   9.02    80  0.0     0   0   0   1   0   ...     0   0   0   0   0   0   0   0   0   0
2   0   0   9.02    80  0.0     0   0   0   0   1   ...     0   0   0   0   0   0   0   0   0   0
3   0   0   9.84    75  0.0     0   0   0   0   0   ...     0   0   0   0   0   0   0   0   0   0
4   0   0   9.84    75  0.0     0   0   0   0   0   ...     0   0   0   0   0   0   0   0   0   0
5 rows × 52 columns
'''
'''
    --- featrue_engineering end. ---
    begin to train
'''

'''
    get train_set, test_set from 'data'
'''
print('data.shape: ', data.shape)  # (10886, 52)
train_X = data[:train.shape[0]]
test_X = data[train.shape[0]:]
# label
train_y = train['count']  



from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
# standardization
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler

'''
    cross_validation，return avg root_mean_square_estimation
'''
def rmse_cv(model, n_splits):
    # cross_validation generator 'kf' shuffle=True，
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    mse = -cross_val_score(model, train_X, train_y, scoring="neg_mean_squared_error", cv = kf)

    print('n_splits=', n_splits, 'mse.shape: ', mse.shape)
    rmse = np.sqrt(mse)  
    return(rmse.mean())

c = train_X.columns

# from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

pd.DataFrame(train_X, columns=c).head()
'''
holiday     workingday  temp    humidity    windspeed   weather_2   weather_3   weather_4   time_01     time_02     ...     month_07    month_08    month_09    month_10    month_11    month_12    year_2012   season_2    season_3    season_4
0   -0.169071   -1.46635    -1.284875   0.997026    -1.553924   -0.58111    -0.29208    -0.010784   -0.208717   -0.207500   ...     -0.302276   -0.302276   -0.301588   -0.302276   -0.302046   -0.302276   -1.003028   -0.578603   -0.578603   -0.578961
1   -0.169071   -1.46635    -1.387368   0.945658    -1.553924   -0.58111    -0.29208    -0.010784   4.791183    -0.207500   ...     -0.302276   -0.302276   -0.301588   -0.302276   -0.302046   -0.302276   -1.003028   -0.578603   -0.578603   -0.578961
2   -0.169071   -1.46635    -1.387368   0.945658    -1.553924   -0.58111    -0.29208    -0.010784   -0.208717   4.819269    ...     -0.302276   -0.302276   -0.301588   -0.302276   -0.302046   -0.302276   -1.003028   -0.578603   -0.578603   -0.578961
3   -0.169071   -1.46635    -1.284875   0.688817    -1.553924   -0.58111    -0.29208    -0.010784   -0.208717   -0.207500   ...     -0.302276   -0.302276   -0.301588   -0.302276   -0.302046   -0.302276   -1.003028   -0.578603   -0.578603   -0.578961
4   -0.169071   -1.46635    -1.284875   0.688817    -1.553924   -0.58111    -0.29208    -0.010784   -0.208717   -0.207500   ...     -0.302276   -0.302276   -0.301588   -0.302276   -0.302046   -0.302276   -1.003028   -0.578603   -0.578603   -0.578961
5 rows × 52 columns
'''

'''
    try models
'''
from sklearn.tree import DecisionTreeRegressor 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.model_selection import GridSearchCV 

knn = KNeighborsRegressor()
print(rmse_cv(knn, n_splits=3)) # 0.7086534263011788

dt = DecisionTreeRegressor()
print(rmse_cv(dt, n_splits=3))

'''
    SKLearn 的model_selection GridSearchCV() 
'''

# from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(DecisionTreeRegressor(), scoring="neg_mean_squared_error", cv=3, verbose=3,
                  param_grid={"max_depth": [2, 5, 10, 20, 50, 100, None], 
                              "min_samples_split":[2, 5, 10, 20, 50, 100]}, ) 
gs.fit(train_X, train_y)
gs.best_params_  # out {'max_depth': 20, 'min_samples_split': 20}


gs = GridSearchCV(DecisionTreeRegressor(), scoring="neg_mean_squared_error", cv=3, verbose=1,
                  param_grid={"max_depth": [20, 50, 100, 150, 200, None], 
                              "min_samples_split":[i for i in range(15, 35)]}, )
gs.fit(train_X, train_y)
gs.best_params_       # {'max_depth': 50, 'min_samples_split': 29}

rmse = rmse_cv(gs.best_estimator_, n_splits=3)
print('rmse from best_estimator_: ', rmse)


gs.best_estimator_.fit(train_X, train_y)

# redict test_set
predict_y = gs.best_estimator_.predict(test_X)


'''
    count = 2 ** predict_y - 1
'''
count = 2 ** predict_y - 1
# get the final result, and save to .csv file
res = pd.DataFrame([test['datetime'].values, count], index=['datetime', 'count']).T
res.to_csv('result_bikesharing.csv', index=False, header=True)

