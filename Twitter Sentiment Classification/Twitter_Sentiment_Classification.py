#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The file: sentiment-train.csv contains 60k tweets annotated by their sentiments (0: negative, 1: positive)




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
# Functions ###########================-------------------
    


def NLP_PreProcessing(text_main, Indicator_StopWords):
    # lower casing
    text = text_main.lower()    
    # sentence segmenting    
    sentences = nltk.sent_tokenize(text)       
    # Tokenization      
    Tokenized_sentences = [word_tokenize(S) for S in sentences] 
    # Stop Words
    All_Stop_words = stopwords.words('english')
    if Indicator_StopWords == 1:
        filtered_Stop_words_Tokenized_sentences = []
        for i in range(len(Tokenized_sentences)):
            filtered_Stop_words_Tokenized_sentences.append([word for word in Tokenized_sentences[i] if word not in All_Stop_words])
        return filtered_Stop_words_Tokenized_sentences
    else:
        return Tokenized_sentences

def Construct_Vec_representation(Train_Data, Y_Train, My_Word2Vec, dim_size):
    for i in range(len(Train_Data)):
        if (i%5000) == 0:
            print(i)
        Vec_rep = np.zeros(dim_size)
        text = Train_Data['text'][i].lower()
        Tweet_TOKENS = [w for w in (nltk.word_tokenize(text))] # split, tokenize
        LENGTH = len(Tweet_TOKENS)
        for Token in Tweet_TOKENS:
            try: 
                Vec_rep = Vec_rep + My_Word2Vec.wv[Token]
            except:
                LENGTH = LENGTH -1           
        if LENGTH > 0:    
            Vec_rep = Vec_rep/LENGTH
            if i == 0:
                X_Train_Vec_rep = Vec_rep
                Y_Train_Vec_rep = Y_Train[i]
            else:
                X_Train_Vec_rep = np.vstack([X_Train_Vec_rep,Vec_rep])
                Y_Train_Vec_rep = np.vstack([Y_Train_Vec_rep,Y_Train[i]])
    
    return X_Train_Vec_rep,Y_Train_Vec_rep







# Data 
Train_Data = pd.read_csv('sentiment-train.csv')
Y_Train = Train_Data['sentiment']# pd.DataFrame(Train_Data['sentiment'], columns=['sentiment'])
Test_Data = pd.read_csv('sentiment-test.csv')
Y_Test = Test_Data['sentiment'] # pd.DataFrame(Test_Data['sentiment'], columns=['sentiment'])


#  Q2.1   ###########================-------------------
# CountVectorizer - Transform training data into a 'document-term matrix' 
My_CountVectorizer = CountVectorizer(stop_words= 'english', max_features=1000)
My_CountVectorizer.fit(list(Train_Data['text']))
X_Train_CountVectorizer = pd.DataFrame(My_CountVectorizer.transform(list(Train_Data['text'])).toarray(), columns=My_CountVectorizer.get_feature_names())
X_Test_CountVectorizer = pd.DataFrame(My_CountVectorizer.transform(list(Test_Data['text'])).toarray(), columns=My_CountVectorizer.get_feature_names())

MNB = MultinomialNB()
MNB.fit(X_Train_CountVectorizer, Y_Train)
Predicted_Y_Test = MNB.predict(X_Test_CountVectorizer)
print('Accuracy of Multinomial Naive Bayes classifier - CountVectorizer:   ', metrics.accuracy_score(Y_Test, Predicted_Y_Test))

del My_CountVectorizer , Predicted_Y_Test, 




#  Q2.2   ###########================-------------------
# TFIDFVectorizer - Transform training data into a 'document-term matrix' 

My_TFIDFVectorizer = TfidfVectorizer(stop_words= 'english', max_features=1000)
My_TFIDFVectorizer.fit(list(Train_Data['text']))
X_Train_My_TFIDFVectorizer = pd.DataFrame(My_TFIDFVectorizer.transform(list(Train_Data['text'])).toarray(), columns=My_TFIDFVectorizer.get_feature_names())
X_Test_My_TFIDFVectorizer = pd.DataFrame(My_TFIDFVectorizer.transform(list(Test_Data['text'])).toarray(), columns=My_TFIDFVectorizer.get_feature_names())

MNB.fit(X_Train_My_TFIDFVectorizer, Y_Train)
Predicted_Y_Test = MNB.predict(X_Test_My_TFIDFVectorizer)
print('Accuracy of Multinomial Naive Bayes classifier - TFIDFVectorizer:   ', metrics.accuracy_score(Y_Test, Predicted_Y_Test))

del My_TFIDFVectorizer, MNB , Predicted_Y_Test,


#  Q2.3   ###########================-------------------

LR = LogisticRegression(random_state=0).fit(X_Train_CountVectorizer, Y_Train)
Predicted_Y_Test = LR.predict(X_Test_CountVectorizer)
print('Accuracy of logistic regression classifier - CountVectorizer:   ', metrics.accuracy_score(Y_Test, Predicted_Y_Test))

del LR , Predicted_Y_Test, X_Train_CountVectorizer,X_Test_CountVectorizer



#  Q2.4   ###########================-------------------

LR = LogisticRegression(random_state=0).fit(X_Train_My_TFIDFVectorizer, Y_Train)
Predicted_Y_Test = LR.predict(X_Test_My_TFIDFVectorizer)
print('Accuracy of logistic regression classifier - TFIDFVectorizer:   ', metrics.accuracy_score(Y_Test, Predicted_Y_Test))

del LR , Predicted_Y_Test, X_Train_My_TFIDFVectorizer,X_Test_My_TFIDFVectorizer

#  Q2.5   ###########================-------------------


#  Q2.5 - (a)
skf = StratifiedKFold(n_splits=5)
MNB = MultinomialNB()
Set_Max_F = [1000,2000,3000,4000]
List_Acc = []
for Max_F in Set_Max_F:
    print("OOOO  Result for max features =  ", Max_F, '   OOOO=====---------')
    itr =1
    Total_Acc = 0
    for train_index, test_index in skf.split(Train_Data['text'], Y_Train):
        X_tr, X_ts = Train_Data['text'][train_index], Train_Data['text'][test_index]
        y_tr, y_ts = Y_Train[train_index], Y_Train[test_index]
        # Training a Multinomial NB classifier
        My_TFIDFVectorizer = TfidfVectorizer(stop_words= 'english', max_features=Max_F)
        My_TFIDFVectorizer.fit(list(X_tr))
        FiveFold_X_Train = pd.DataFrame(My_TFIDFVectorizer.transform(list(X_tr)).toarray(), columns=My_TFIDFVectorizer.get_feature_names())
        FiveFold_X_Test = pd.DataFrame(My_TFIDFVectorizer.transform(list(X_ts)).toarray(), columns=My_TFIDFVectorizer.get_feature_names())
        MNB.fit(FiveFold_X_Train, y_tr)
        Predicted_Y_Test = MNB.predict(FiveFold_X_Test)
        print("Fold ", itr,' --------- ',"Accuracy: ", metrics.accuracy_score(y_ts, Predicted_Y_Test))
        Total_Acc = Total_Acc + metrics.accuracy_score(y_ts, Predicted_Y_Test)
        itr += 1    
    print('Average Accuracy across folds:',(Total_Acc/5),'\n')
    List_Acc.append((Total_Acc/5))

del Total_Acc,skf,MNB,itr,X_tr,X_ts,y_tr,y_ts,train_index,test_index
del My_TFIDFVectorizer,FiveFold_X_Train,FiveFold_X_Test,Predicted_Y_Test,Max_F

#  Q2.5 - (b)
Best_Max_F = Set_Max_F[List_Acc.index(max(List_Acc))]  # max features value that has the highest average accuracy
MNB = MultinomialNB()
My_TFIDFVectorizer = TfidfVectorizer(stop_words= 'english', max_features=Best_Max_F)
My_TFIDFVectorizer.fit(list(Train_Data['text']))
X_Train_My_TFIDFVectorizer = pd.DataFrame(My_TFIDFVectorizer.transform(list(Train_Data['text'])).toarray(), columns=My_TFIDFVectorizer.get_feature_names())
X_Test_My_TFIDFVectorizer = pd.DataFrame(My_TFIDFVectorizer.transform(list(Test_Data['text'])).toarray(), columns=My_TFIDFVectorizer.get_feature_names())

MNB.fit(X_Train_My_TFIDFVectorizer, Y_Train)
Predicted_Y_Test = MNB.predict(X_Test_My_TFIDFVectorizer)
print('Accuracy of Multinomial Naive Bayes classifier - Best max feature (',Best_Max_F,'):   ', metrics.accuracy_score(Y_Test, Predicted_Y_Test),'\n')

del My_TFIDFVectorizer, MNB , Predicted_Y_Test,Best_Max_F,Set_Max_F,X_Test_My_TFIDFVectorizer,X_Train_My_TFIDFVectorizer



#  Q2.6   ###########================-------------------


#  Q2.6 - (a)
# sentence segmenting + lower casing + Tokenization using function  NLP_PreProcessing
Train_Preprocessed_Tweets = []
for t in range(len(Train_Data)):
    Train_Preprocessed_Tweets = Train_Preprocessed_Tweets + NLP_PreProcessing(Train_Data['text'][t], Indicator_StopWords= 0 ) 

My_Word2Vec = Word2Vec(sentences=Train_Preprocessed_Tweets, size=300) # Creating Word2Vec Model


#  Q2.6 - (b)
# vector representation of each tweet as the average of all the word vectors in the tweet
X_Train_Vec_rep, Y_Train_Vec_rep = Construct_Vec_representation(Train_Data, Y_Train, My_Word2Vec, 300)



#  Q2.6 - (c)

# Convert test tweets to vector representation
X_Test_Vec_rep, Y_Test_Vec_rep = Construct_Vec_representation(Test_Data, Y_Test, My_Word2Vec, 300)
# Train LR
LR = LogisticRegression(random_state=0,max_iter = 1000).fit(X_Train_Vec_rep, Y_Train_Vec_rep.ravel())
Predicted_Y_Test = LR.predict(X_Test_Vec_rep)
print('Accuracy of logistic regression classifier - Vector representation:   ', metrics.accuracy_score(Y_Test_Vec_rep, Predicted_Y_Test))
del LR , Predicted_Y_Test


#  Q2.6 - (d)
# Remove stopwords
Train_Preprocessed_Tweets_No_Stop_words = []
for t in range(len(Train_Data)):
    Train_Preprocessed_Tweets_No_Stop_words = Train_Preprocessed_Tweets_No_Stop_words + NLP_PreProcessing(Train_Data['text'][t], Indicator_StopWords=1) 

My_Word2Vec_No_Stop_words = Word2Vec(sentences=Train_Preprocessed_Tweets_No_Stop_words, size=300) # Creating Word2Vec Model
# Convert Train tweets to vector representation
X_Train_Vec_rep_No_Stop_words, Y_Train_Vec_rep_No_Stop_words = Construct_Vec_representation(Train_Data, Y_Train, My_Word2Vec_No_Stop_words, 300)
# Convert test tweets to vector representation
X_Test_Vec_rep_No_Stop_words, Y_test_Vec_rep_No_Stop_words = Construct_Vec_representation(Test_Data, Y_Test, My_Word2Vec_No_Stop_words, 300)

# Train LR
LR = LogisticRegression(random_state=0,max_iter = 1000).fit(X_Train_Vec_rep_No_Stop_words, Y_Train_Vec_rep_No_Stop_words.ravel())
Predicted_Y_Test = LR.predict(X_Test_Vec_rep_No_Stop_words)
print('Accuracy of logistic regression classifier - Vector representation - No Stop Words:   ', metrics.accuracy_score(Y_test_Vec_rep_No_Stop_words, Predicted_Y_Test))
del LR , Predicted_Y_Test



































