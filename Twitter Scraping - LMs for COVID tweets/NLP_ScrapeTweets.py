#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Twitter Scraping. We used Twitter Developer API to scrape 10,000 most recent
tweets in the English language from Twitter with the keyword ’covid’. Out of
these 10,000 tweets, we used 9,000 to train a unigram, bigram, and trigram 
language models (LMs). we used NLTK library with KneserNeyInterpolated 
language model (currently possibly the best for smoothing) to build our
 LMs to deal with zero-count ngrams. 



@author: Shahab Sotudian
"""

import tweepy
from tweepy import OAuthHandler
import pandas as pd
import time
import pickle
import nltk
import sys
import string
from twython import Twython
import random
from sklearn.model_selection import train_test_split
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import KneserNeyInterpolated
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import ngrams
from nltk.tokenize.treebank import TreebankWordDetokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import re
import collections
import itertools
nltk.download('stopwords')
from nltk.lm import MLE

# Functions ###########================-------------------


def NLP_PreProcessing(text_main):
    # lower casing
    text = text_main.lower()    
    # Remove whitespaces
    text = text.strip()
    # sentence segmenting    
    sentences = nltk.sent_tokenize(text)       
    # Tokenization      
    Tokenized_sentences = [word_tokenize(S) for S in sentences]    
    return Tokenized_sentences


def Generate_Sentence(model, num_words, random_seed= random.randint(1,111)):
    SENT = []
    for token in model.generate(num_words, random_seed=random_seed):
        SENT.append(token)
    return detokenize(SENT)


def TopN_Words(Data_Tweets, N):
    All_Stop_words = stopwords.words('english')
    All_tweets = [Tweets.text for Tweets in Data_Tweets]
    Cleaned_All_tweets = [" ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", tweet).split()) for tweet in All_tweets] # Remove URL and redundant strings
    All_Words_All_Tweets = [tweet.lower().split() for tweet in Cleaned_All_tweets]
    All_Words = list(itertools.chain(*All_Words_All_Tweets))
    All_Words_filtered_Stop_words = [word for word in All_Words if word not in All_Stop_words]
    Words_counts = collections.Counter(All_Words_filtered_Stop_words)
    return Words_counts.most_common(N)




# Generate Or Load Data  ###########================-------------------
Do_You_Want_Generate_Data = 0 # if 1, it will use API to scrape 10000 tweets. If 0, it will load the saved data


if Do_You_Want_Generate_Data == 1: # I removed my Keys and tokens
    access_token = 'AAAAAAAA'
    access_token_secret = 'AAAAAAAA'
    consumer_key = 'AAAAAAAA'
    consumer_secret = 'AAAAAAAA'
    
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    tweets = []
    itr = 1
    tweets = []
    for tweet in tweepy.Cursor(api.search, q= "covid -filter:retweets" , count=10000, lang = 'en', result_type = 'recent').items(10000):# Adding to list that contains all tweets
        try:
            tweets.append((tweet))
            itr += 1
            print(itr)
        except BaseException as e:
            print('failed on_status,',str(e))
        #time.sleep(1.1)
        except StopIteration:
            break
        
        # Save data
    a_file = open("Data_10000_Twetes_RecentCOVID.pkl", "wb")
    pickle.dump(tweets, a_file)
    a_file.close()
else:
   # Load Data 
   Data = pickle.load( open("Data_10000_Twetes_RecentCOVID.pkl", "rb") ) 




# Train-Test split ###########================-------------------
X_train, X_test= train_test_split(Data, test_size=0.1, random_state=1)


#  Training ###########================-------------------

# sentence segmenting + lower casing + Tokenization + Padding using function  NLP_PreProcessing
Train_Preprocessed_Tweets = []
for t in range(len(X_train)):
    Train_Preprocessed_Tweets = Train_Preprocessed_Tweets + NLP_PreProcessing(X_train[t].text) 

# Train Unigram
train_data_Unigram, padded_sents_Unigram = padded_everygram_pipeline( 1 , Train_Preprocessed_Tweets)
Unigram_Model = KneserNeyInterpolated(1) 
Unigram_Model.fit(train_data_Unigram, padded_sents_Unigram)

# Train Bigram
train_data_Bigram, padded_sents_Bigram = padded_everygram_pipeline( 2 , Train_Preprocessed_Tweets)
Bigram_Model = KneserNeyInterpolated(2) 
Bigram_Model.fit(train_data_Bigram, padded_sents_Bigram)
 
# Train Trigram
train_data_Trigram, padded_sents_Trigram = padded_everygram_pipeline( 3 , Train_Preprocessed_Tweets)
Trigram_Model = KneserNeyInterpolated(3) 
Trigram_Model.fit(train_data_Trigram, padded_sents_Trigram)

print('Number of Ngram: ', Unigram_Model.counts)
print('Number of Ngram: ', Bigram_Model.counts)
print('Number of Ngram: ', Trigram_Model.counts)




#  Q1: Testing + Average perplexities ###########================-------------------

# sentence segmenting + lower casing + Tokenization + Padding using function  NLP_PreProcessing
Test_Preprocessed_Tweets = []
for t in range(len(X_test)):
    Test_Preprocessed_Tweets = Test_Preprocessed_Tweets + NLP_PreProcessing(X_test[t].text) 

Total_Uni_Perplex = 0
Total_Bi_Perplex = 0
Total_Tri_Perplex = 0

for tt in range(len(Test_Preprocessed_Tweets)):
    # Unigram
    test_data_Unigram = list(ngrams(Test_Preprocessed_Tweets[tt], n=1,pad_left=True, pad_right=True,left_pad_symbol='<s>',right_pad_symbol='</s>'))
    Uni_Preplex = Unigram_Model.perplexity(test_data_Unigram)
    Total_Uni_Perplex = Total_Uni_Perplex + Uni_Preplex
    # Bigram
    test_data_Bigram = list(ngrams(Test_Preprocessed_Tweets[tt], n=2,pad_left=True, pad_right=True,left_pad_symbol='<s>',right_pad_symbol='</s>'))
    Bi_Preplex = Bigram_Model.perplexity(test_data_Bigram)
    Total_Bi_Perplex = Total_Bi_Perplex + Bi_Preplex
    # Trigram
    test_data_Trigram = list(ngrams(Test_Preprocessed_Tweets[tt], n=3,pad_left=True, pad_right=True,left_pad_symbol='<s>',right_pad_symbol='</s>'))
    Tri_Preplex = Trigram_Model.perplexity(test_data_Trigram)
    Total_Tri_Perplex = Total_Tri_Perplex + Tri_Preplex

print("**********************************************\n")
print("Average perplexities :  ", (Total_Uni_Perplex/len(Test_Preprocessed_Tweets)),' -- ', (Total_Bi_Perplex/len(Test_Preprocessed_Tweets)),' -- ', (Total_Tri_Perplex/len(Test_Preprocessed_Tweets)))
print("**********************************************\n")



# Q2: Generate 10 tweet  ###########================-------------------

detokenize = TreebankWordDetokenizer().detokenize

for ii in range(10):
    print("\n Generated Tweet round:  ", (ii+1))
    print("**** Tweet from Unigram model ****")
    print(Generate_Sentence(Unigram_Model, random.randint(8,25)  ))
    print("**** Tweet from Bigram model ****") 
    print(Generate_Sentence(Bigram_Model, random.randint(8,25)   ))
    print("**** Tweet from Trigram model ****")
    print(Generate_Sentence(Trigram_Model, random.randint(8,25)  ))
    print("**********************************************")


# as you mnetioned in Piazza, I also used MLE to generate tweets
'''  
for N in range((3)):
   train_data_Ngram, padded_sents_Ngram = padded_everygram_pipeline( (N+1) , Train_Preprocessed_Tweets)
   N_Model = MLE((N+1)) 
   N_Model.fit(train_data_Ngram, padded_sents_Ngram)
   print("\n\n N in Ngram:  ", (N+1))
   for ii in range(30):
       print(Generate_Sentence(N_Model, random.randint(8,25) , random_seed= (ii+13)  ))
       print("-----------------------------------------------")

   print("**********************************************")
'''
    
    
# Q3-a: Average sentiment of tweets   ###########================-------------------

analyser = SentimentIntensityAnalyzer()
Total_Compound_sentiment = 0
Positive_Tweets = []
Negative_Tweets = []
for jj in range(len(Data)):
    SCORE = analyser.polarity_scores(Data[jj].text)
    print("Tweet ",jj, " : ",  str(SCORE))
    Total_Compound_sentiment = Total_Compound_sentiment + SCORE['compound']
    if SCORE['compound'] >0:
        Positive_Tweets.append(Data[jj]) 
    else:
        Negative_Tweets.append(Data[jj]) 

print("Average compound sentiment of the tweets:  ",(Total_Compound_sentiment/len(Data)))


# Q3-b: top 10 words mentione after removing stopword  ###########================-------------------

print("*********  Top 10 words mentione in Positive tweets  ******\n")
print(TopN_Words(Positive_Tweets, 10))
print("*********  Top 10 words mentione in Negative tweets  ******\n")
print(TopN_Words(Negative_Tweets, 10))


# Q3-c: sentiment compound scores for diffrent states  ###########================-------------------
Analyser = SentimentIntensityAnalyzer()
US_Tweets = []
Sates_Score = []
for jj in range(len(Data)):
    if  not Data[jj].place == None:
        if Data[jj].place.country_code == "US":
            US_Tweets.append(Data[jj])
            compound_SCORE_sentiment = Analyser.polarity_scores(Data[jj].text)['compound']
            Sates_Score.append([compound_SCORE_sentiment,Data[jj].place.full_name])      #average sentiment compound scores  + States name

Final_Sates_Score = []
States_Names = []
States_Scores = []
for i in range(len(Sates_Score)):
    # I remove some problematic samples
    if Sates_Score[i][1][-3:] != 'USA' and (Sates_Score[i][1][-3:] != 'ico') and (Sates_Score[i][1][-3:] != 'ter'):
        print('state: ', Sates_Score[i][1], '  ------    Score: ' , Sates_Score[i][0])
        Final_Sates_Score.append([Sates_Score[i][1][-2:],Sates_Score[i][0]])
        States_Names.append(Sates_Score[i][1][-2:])
        States_Scores.append(Sates_Score[i][0])
    
    
States_Scores_df=pd.DataFrame(States_Scores,columns=["Scores"],index=States_Names)
print("*********  Average sentiment compound scores from each of the states: ******\n")
print(States_Scores_df.groupby(States_Scores_df.index).mean().sort_values(by=['Scores']))













