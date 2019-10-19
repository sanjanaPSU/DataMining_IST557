# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:25:02 2019

@author: Sanjana
"""
import pandas as pd
import os
import nltk
import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 

ps = PorterStemmer() 
arr1 = os.listdir('C:\\Users\\Sanjana\\Desktop\\20news\\rec.sport.hockey')
arr2 = os.listdir('C:\\Users\\Sanjana\\Desktop\\20news\\soc.religion.christian')
data = [[0 for i in range(3002)] for j in range(1997)]
# define punctuation
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
stop_words = set(stopwords.words('english')) 
f = open('C:\\Users\\Sanjana\\Desktop\\dict.txt', "r")
col_name = f.readlines()
f.close
col_name = [x.replace('\n','') for x in col_name]
col_name.insert(0,"docID")
col_name.insert(1,"class_label")
df = pd.DataFrame(data, columns = list(col_name))
df_idf = pd.DataFrame(data)
doc_len = []

for i in range(1997):
    if i<1000:
        df.loc[i,'docID'] = arr1[i]
        df.loc[i, 'class_label'] = 0
        filepath = 'C:\\Users\\Sanjana\\Desktop\\20news\\rec.sport.hockey\\'+arr1[i]
        f= open(filepath,"r")
        contents = f.read()
        contents = contents.lower()
        # remove punctuation from the string
        no_punct = ""
        for char in contents:
            if char not in punctuations:
                no_punct = no_punct + char
        # tokenize document
        tokens = nltk.word_tokenize(no_punct)
        # remove stop words
        filtered_content =[]
        for w in tokens: 
            if w not in stop_words: 
                filtered_content.append(w)
        # stemming process
        stemmed = []
        for w in filtered_content: 
            stemmed.append(ps.stem(w))
        doc_len.append(len(stemmed))
        for w in stemmed:
            if w in col_name:
                df.loc[i, w] = df.loc[i,w]+1
    else:
        df.loc[i,'docID'] = arr2[i-1000]
        df.loc[i, 'class_label'] = 1
        filepath = 'C:\\Users\\Sanjana\\Desktop\\20news\\soc.religion.christian\\'+arr2[i-1000]
        f= open(filepath,"r")
        contents = f.read()
        contents = contents.lower()
        # remove punctuation from the string
        no_punct = ""
        for char in contents:
            if char not in punctuations:
                no_punct = no_punct + char
        # tokenize document
        tokens = nltk.word_tokenize(no_punct)
        # remove stop words
        filtered_content =[]
        for w in tokens: 
            if w not in stop_words: 
                filtered_content.append(w)
        # stemming process
        stemmed = []
        for w in filtered_content: 
            stemmed.append(ps.stem(w))
        doc_len.append(len(stemmed))
        for w in stemmed:
            if w in col_name:
                df.loc[i, w] = df.loc[i,w]+1
                

tot_df = df.sum(axis = 0, skipna = True) 
for i in range(1997):
    for j in range(3,3002):
        df.iloc[i,j] = df.iloc[i,j]/doc_len[i]
        
for i in range(1997):
    for j in range(3,3002):
        df_idf.iloc[i,j] = - (math.log((tot_df.iloc[i]+1)/1997))

for i in range(1997):
    for j in range(3,3002):
        df.iloc[i,j] = df_idf.iloc[i,j]*df.iloc[i,j]

df.to_csv(r'C:\Users\Sanjana\Desktop\Matrix.txt', header=None, index=None, sep=',', mode='a')


#if-else is used to navigate to the right location
#df stores the term frequency initially nd then tf-idf
#df_idf stores the inverse document frequency
        
        

