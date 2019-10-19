# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 20:47:01 2019

@author: Sanjana
"""

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
import numpy as np
import warnings

warnings.filterwarnings("ignore")

print("------Neural Networks------")

filepath = 'C:\\Users\\Sanjana\\Desktop\\train.txt'
data_train = pd.read_csv(filepath, header = None)

scaler = StandardScaler()


X=pd.DataFrame(data_train.iloc[:,0:9])
Y = data_train.iloc[:,9]

X.iloc[:,0].replace(10.0,1.0,inplace=True)

parameter_space = {
    'hidden_layer_sizes': [(120,),(50,100,50),(50,),(100,),(80,),(200,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# fit model on training data
model = MLPClassifier(hidden_layer_sizes=(30,30,30))
model.fit(X_train, Y_train)

model = GridSearchCV(estimator = model,param_grid = parameter_space,cv=3,scoring='accuracy')
model.fit(X_train,Y_train)
#best parameter printing
print (model.best_params_)
scores = []

k_fold = KFold(n_splits=5)
iter = 0
for train_index, test_index in k_fold.split(X,Y):
    #print("round ",iter)
    #iter+=1
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    # Fit only to the training data
    scaler.fit(X_train)
    mlp = MLPClassifier(hidden_layer_sizes=model.best_params_['hidden_layer_sizes'], activation=model.best_params_['activation'],solver=model.best_params_['solver'],alpha=model.best_params_['alpha'],learning_rate=model.best_params_['learning_rate'])
    mlp.fit(X_train,Y_train)
    mlp_predict = mlp.predict(X_test)
    scores.append(metrics.accuracy_score(Y_test, mlp_predict))

avg_score = np.mean(scores,axis=0)
print("Average scores: ",avg_score)
std_dev = np.std(scores)
print("Standard deviation: ",std_dev)
    