# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:52:25 2019

@author: Sanjana
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
cC = [0.01, 0.1, 1, 10, 100]
print ("C1")
#C1 tune parameter C and corresponding accuracy using 5-fold cross-validation 
for c in cC:
    svclassifier = SVC(kernel='linear', C= c)
    svclassifier.fit(X_train, y_train.values.ravel())
    # K-Fold CV
    scores = cross_val_score(svclassifier, X_train, y_train.values.ravel(), cv=5)
    print('Accuracy: ' , (c, scores.mean(), scores))

#C2 :
print ("**********************************")
print ("C2")
outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=1)

outer_scores = []
# outer folds
for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)): 
    print('[Outer fold %d/5]' % (i + 1))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    best_c, best_score = -1, -1
    svm_clfs = {}
    # hyperparameter tuning 
    for c in cC:
        inner_scores = []
        # inner folds
        for itrain_idx, val_idx in inner_cv.split(X_train, y_train):
            X_itrain, X_val = X_train.iloc[itrain_idx], X_train.iloc[val_idx]
            y_itrain, y_val = y_train.iloc[itrain_idx], y_train.iloc[val_idx]
            
            svclassifier = SVC(kernel='linear', C= c)
            svclassifier.fit(X_train, y_train.values.ravel())
            
            y_pred = svclassifier.predict(X_val)
            inner_scores.append(accuracy_score(y_val, y_pred))
        score_mean = np.mean(inner_scores)
        if best_score < score_mean:
            best_c, best_score = c, score_mean
        svm_clfs[c] = svclassifier
        
    # evaluate performance on test fold
    best = svm_clfs[best_c]
    best.fit(X_train, y_train.values.ravel())    
    y_pred = best.predict(X_test)
    outer_scores.append(accuracy_score(y_test, y_pred))
    print('Test accuracy: %.2f (Parameter=%d selected by inner 5-fold CV)' % 
                  (outer_scores[i], best_c))

print('\nTest accuracy: %.2f (5x5 nested CV)' % np.mean(outer_scores))

