# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 18:05:41 2019

@author: Sanjana
"""

from sklearn.model_selection import GridSearchCV,KFold
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np

print("------SVM------")

filepath = 'C:\\Users\\Sanjana\\Desktop\\train.txt'
data_train = pd.read_csv(filepath, header = None)

X= pd.DataFrame(data_train.iloc[:,0:9])
Y = data_train.iloc[:,9]

X.iloc[:,0].replace(10.0,1.0,inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# number of features at every split
kernel = ['linear', 'rbf']

# max depth
C = [0.01,0.1,1,10,100]
# create random grid
random_grid = {
 'kernel': kernel,
 'C': C
 }

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# fit model on training data
model = SVC()
model.fit(X_train, Y_train)

model = GridSearchCV(estimator = model,param_grid = random_grid)
model.fit(X_train,Y_train)
#best parameter printing
print (model.best_params_)
scores = []

k_fold = KFold(n_splits=5)
iter = 0
for train_index, test_index in k_fold.split(X,Y):
    #print("round ",iter)
    #iter+=1
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    svclassifier = SVC(kernel=model.best_params_['kernel'], C= model.best_params_['C'])
    svclassifier.fit(X_train,Y_train)
    svc_predict = svclassifier.predict(X_test)
    scores.append(metrics.accuracy_score(Y_test, svc_predict))

avg_score = np.mean(scores,axis=0)
print("Average scores: ",avg_score)
std_dev = np.std(scores)
print("Standard deviation: ",std_dev)
   