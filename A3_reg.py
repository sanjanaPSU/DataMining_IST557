# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 18:41:55 2019

@author: Sanjana
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 18:05:41 2019

@author: Sanjana
"""

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import preprocessing

print("------Logistic Regression------")

filepath = 'C:\\Users\\Sanjana\\Desktop\\train.txt'
data_train = pd.read_csv(filepath, header = None)

X= pd.DataFrame(data_train.iloc[:,0:9])
Y = data_train.iloc[:,9]

filepath1 = 'C:\\Users\\Sanjana\\Desktop\\test.txt'
X_final_test = pd.read_csv(filepath1, header = None)

X_final_test.iloc[:,0].replace(10.0,1.0,inplace=True)
X.iloc[:,0].replace(10.0,1.0,inplace=True)

min_max_scaler = preprocessing.MinMaxScaler()
X_final_test= min_max_scaler.fit_transform(X_final_test)

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# fit model on training data
model = LogisticRegression()
model.fit(X_train, Y_train)

clf = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial')
clf.fit(X_train,Y_train)
dict_param = clf.get_params()
#best parameter printing
print (clf.get_params())
scores = []

k_fold = KFold(n_splits=5)
#iter = 0
for train_index, test_index in k_fold.split(X,Y):
    #print("round ",iter)
    #iter+=1
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    # Random search of parameters
    lr = LogisticRegression(random_state=dict_param['random_state'], solver=dict_param['solver'], multi_class=dict_param['multi_class'])
    lr.fit(X_train,Y_train)
    y_pred = lr.predict(X_test)
    scores.append(metrics.accuracy_score(Y_test, y_pred))

lr_predict = lr.predict(X_final_test)
np.savetxt("C:\\Users\\Sanjana\\Desktop\\results.txt", lr_predict)
avg_score = np.mean(scores,axis=0)
print("Average scores: ",avg_score)
std_dev = np.std(scores)
print("Standard deviation: ",std_dev)
    