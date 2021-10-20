#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data: http://help.sentiment140.com/for-students/

@author: shahab Sotudian
"""


import pandas as pd
import numpy as np
import math
import os
import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize 
import gensim 
from gensim.models import Word2Vec 
from nltk.corpus import stopwords 

import time
start_time = time.time()
# Data 
'''
Data file format has 6 fields:
0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
1 - the id of the tweet (2087)
2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
3 - the query (lyx). If there is no query, then this value is NO_QUERY.
4 - the user that tweeted (robotickilldozr)
5 - the text of the tweet (Lyx is cool)
'''
# Train Data
Tweets16M_Data = pd.read_csv('training.1600000.processed.noemoticon.csv', header=None, encoding = "ISO-8859-1")
Train_Data = pd.DataFrame(Tweets16M_Data[5][:])
Train_Data = Train_Data.rename(columns={5: "text"})
Train_Data['sentiment'] = Tweets16M_Data[0].values
Train_Data['sentiment'].replace(4, 1,inplace=True) # Change label of posoitve class from 4 to 1
Y_Train = Train_Data['sentiment']
# Test Data
Test_Data = pd.read_csv('sentiment-test.csv')
Y_Test = Test_Data['sentiment'] # 
del Tweets16M_Data



# CountVectorizer - Transform training data into a 'document-term matrix' 
My_CountVectorizer = CountVectorizer(stop_words= 'english', max_features=4000)
My_CountVectorizer.fit(list(Train_Data['text']))
X_Train_CountVectorizer = pd.DataFrame(My_CountVectorizer.transform(list(Train_Data['text'])).toarray(), columns=My_CountVectorizer.get_feature_names())
X_Test_CountVectorizer = pd.DataFrame(My_CountVectorizer.transform(list(Test_Data['text'])).toarray(), columns=My_CountVectorizer.get_feature_names())

MNB = MultinomialNB()
MNB.fit(X_Train_CountVectorizer, Y_Train)
Predicted_Y_Test = MNB.predict(X_Test_CountVectorizer)
print('Accuracy of Multinomial Naive Bayes classifier - CountVectorizer usimg 1.6M tweets:   ', metrics.accuracy_score(Y_Test, Predicted_Y_Test))




print("The running time is --- %s seconds ---" % (time.time() - start_time))

