# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 15:53:59 2019

@author: Sanjana
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

news_data = pd.read_csv(r"C:\Users\Sanjana\Desktop\Matrix.txt")

news_data[:] = np.nan_to_num(news_data)

scaler = MinMaxScaler() 
scaled_values = scaler.fit_transform(news_data) 
news_data.loc[:,:] = scaled_values

X = news_data.drop(news_data.columns[[0,1]], axis=1)
y = news_data[news_data.columns[[1]]]

score = []
acc_score = []
#applying kfold
kf = KFold(n_splits=5,random_state=None,shuffle=False)

svclassifier = SVC(kernel='linear', C= 1.0)
#parameter c=1.0
#for linear
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    svclassifier.fit(X_train, y_train.values.ravel())
    y_pred = svclassifier.predict(X_test)
    score.append(svclassifier.score(X_test, y_test))
    acc_score.append(accuracy_score(y_test, y_pred))
    mean = np.mean(acc_score)
    std_dev = np.std(acc_score)
    print ("Mean : ",mean)
    print ("Standard deviation : ", std_dev)
    
print ("Mean Score (Linear) : ", np.mean(acc_score))

print ("**********************************")
#for rbf
svclassifier = SVC(kernel='rbf', C= 1.0)

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    svclassifier.fit(X_train, y_train.values.ravel())
    y_pred = svclassifier.predict(X_test)
    score.append(svclassifier.score(X_test, y_test))
    acc_score.append(accuracy_score(y_test, y_pred))
    mean = np.mean(acc_score)
    std_dev = np.std(acc_score)
    print ("Mean : ",mean)
    print ("Standard deviation : ", std_dev)
    
print ("Mean Score (RBF) : ", np.mean(acc_score))