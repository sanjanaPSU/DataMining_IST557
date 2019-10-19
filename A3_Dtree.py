# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 12:25:50 2019

@author: Sanjana
"""

from sklearn.model_selection import GridSearchCV,cross_val_score,KFold
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

print("------Decision Tree------")

filepath = 'C:\\Users\\Sanjana\\Desktop\\train.txt'
data_train = pd.read_csv(filepath, header = None)

X=pd.DataFrame(data_train.iloc[:,0:9])
Y = data_train.iloc[:,9]

X.iloc[:,0].replace(10.0,1.0,inplace=True)

#scaling the data
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

num_leafs=[1,2,5,6,7,8,10,15,20]

param_grid = [{'min_samples_leaf':num_leafs}]

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# fit model on training data
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

model = GridSearchCV(estimator = model,param_grid = param_grid)
#best parameter printing
model.fit(X_train,Y_train)
print (model.best_params_)
scores = []

k_fold = KFold(n_splits=5)
iter = 0
for train_index, test_index in k_fold.split(X,Y):
    #print("round ",iter)
    #iter+=1
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    dtree = DecisionTreeClassifier(criterion='entropy',min_samples_leaf=model.best_params_['min_samples_leaf'])
    dtree = dtree.fit(X_train, Y_train)
    y_pred = dtree.predict(X_test)
    scores.append(metrics.accuracy_score(Y_test, y_pred))

avg_score = np.mean(scores,axis=0)
print("Average scores: ",avg_score)
std_dev = np.std(scores)
print("Standard deviation: ",std_dev)   




