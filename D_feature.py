# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:01:45 2019

@author: Sanjana
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

#dataframe to store the features and corresponding weights
data = [[0 for i in range(2)] for j in range(3000)]
news_data = pd.read_csv(r"C:\Users\Sanjana\Desktop\Matrix.txt")

news_data[:] = np.nan_to_num(news_data)
scaler = MinMaxScaler() 
scaled_values = scaler.fit_transform(news_data) 
news_data.loc[:,:] = scaled_values

X = news_data.drop(news_data.columns[[0,1]], axis=1)
y = news_data[news_data.columns[[1]]]

svclassifier = SVC(kernel='linear')
svclassifier.fit(X, y.values.ravel())

#extracting the dict features (unigrams)
f = open('C:\\Users\\Sanjana\\Desktop\\dict.txt', "r")
feature_name = f.readlines()
f.close
feature_name = [x.replace('\n','') for x in feature_name]

col_name = ['Feature', 'Weight']
df_features = pd.DataFrame(data, columns = list(col_name))
df_features.loc[:,'Feature'] = feature_name

df_features.loc[:,'Weight'] = svclassifier.coef_.ravel()

top_ten = df_features.nlargest(10,'Weight')
bottom_ten = df_features.nsmallest(10,'Weight')

print ("Top-10 unigrams")
print (top_ten)
print ("Bottom-10 unigrams")
print (bottom_ten)

