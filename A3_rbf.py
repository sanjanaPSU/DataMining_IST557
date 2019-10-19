# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 13:23:25 2019

@author: Sanjana
"""

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
import numpy as np

print("------Random Forest------")

filepath = 'C:\\Users\\Sanjana\\Desktop\\train.txt'
data_train = pd.read_csv(filepath, header = None)

X=pd.DataFrame(data_train.iloc[:,0:9])
Y = data_train.iloc[:,9]

X.iloc[:,0].replace(10.0,1.0,inplace=True)

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

# number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# number of features at every split
max_features = ['auto', 'sqrt']

# max depth
max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
max_depth.append(None)
# create random grid
random_grid = {
 'n_estimators': n_estimators,
 'max_features': max_features,
 'max_depth': max_depth
 }

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# fit model on training data
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)

rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, cv = 5)
rfc_random.fit(X_train,Y_train)
best_param = rfc_random.best_params_
#best parameter printing
print(rfc_random.best_params_)
scores = []

k_fold = KFold(n_splits=5)
iter = 0
for train_index, test_index in k_fold.split(X,Y):
    #print("round ",iter)
    #iter+=1
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    rfc = RandomForestClassifier(n_estimators=rfc_random.best_params_['n_estimators'], max_depth=rfc_random.best_params_['max_depth'], max_features=rfc_random.best_params_['max_features'])
    rfc.fit(X_train,Y_train)
    rfc_predict = rfc.predict(X_test)
    scores.append(metrics.accuracy_score(Y_test, rfc_predict))
 
avg_score = np.mean(scores,axis=0)
print("Average scores: ",avg_score)
std_dev = np.std(scores)
print("Standard deviation: ",std_dev)