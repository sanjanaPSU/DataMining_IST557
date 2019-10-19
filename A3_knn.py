# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 20:07:39 2019

@author: Sanjana
"""

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn import preprocessing

print("------KNN------")

filepath = 'C:\\Users\\Sanjana\\Desktop\\train.txt'
data_train = pd.read_csv(filepath, header = None)

X=pd.DataFrame(data_train.iloc[:,0:9])
Y = data_train.iloc[:,9]

X.iloc[:,0].replace(10.0,1.0,inplace=True)

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

#n_neighbours
n_neighbors = [3,4,5,6,7,8,9]
random_grid = {
 'n_neighbors': n_neighbors
 }

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# fit model on training data
model = KNeighborsClassifier()
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
    knn = KNeighborsClassifier(n_neighbors=model.best_params_['n_neighbors'])
    knn.fit(X_train,Y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(Y_test, y_pred))

avg_score = np.mean(scores,axis=0)
print("Average scores: ",avg_score)
std_dev = np.std(scores)
print("Standard deviation: ",std_dev)