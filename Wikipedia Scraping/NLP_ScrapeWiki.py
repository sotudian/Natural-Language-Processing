#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikipedia Scraping. We used "requests" to scrape HTML of this article in Wikipedia:
https://en.wikipedia.org/wiki/COVID-19
 and scrape also the HTML of articles within Wikipedia that are linked from only the
 content of this page 
Once we retrieved all the articles, we used  BeautifulSoup to extract only the text
 of each article’s content. We also trained a trigram KneserNeyInterpolated language
model.




@author: shahab Sotudian
"""

import requests
from bs4 import BeautifulSoup
import random
import time
import pickle
import spacy
import en_core_web_sm
SpaCy_Model = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS
import string   
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import KneserNeyInterpolated
from nltk.util import ngrams
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




Indicator_Data_Generator = 0    # 1 if you want to generate the text data, 0 if you want to load it


Main_Article = "https://en.wikipedia.org/wiki/COVID-19_pandemic"

response = requests.get(  url= Main_Article, )
soup = BeautifulSoup(response.content, 'html.parser')

title = soup.find(id="firstHeading")
print(title.text)

## Get all the wikipedia articles in content of URL    ######==========-------------------------------
# Get all the wikipedia links in content of URL
allLinks = soup.find(id="content").find_all("a", href=True)
List_All_Wiki_Links =[]
for link in allLinks:
    if link['href'].find("/wiki/") != -1: # We are only interested in wiki articles
        List_All_Wiki_Links.append(link)
        
## Remove non-articles links   ######==========------------------------------- 
Non_Articles = ["Category:", "Help:", "User:", "Template:", "User talk:",
                "Talk:", "Wikipedia talk:", "Wikipedia:" , "disambiguation",
                "Wikipedia:", "WP:", "File:", "MediaWiki:", "WP:Portal",
                "Special:","wikidata","Portal:","Template_talk"]
All_Wiki_Articles = []
for i in range(len(List_All_Wiki_Links)):
    keep = 1	
    link = List_All_Wiki_Links[i]
    for NA in Non_Articles:
         if link['href'].find(NA) != -1:
            keep = 0
            break
        
    if keep == 1:
        All_Wiki_Articles.append(link)
            
# Remove duplicate links        
URLs_all_Articles =  []   
for link in All_Wiki_Articles:
    URL =  link['href']
    if URL.find("wikipedia.org") == -1:
        URL = "https://en.wikipedia.org" + URL
    URLs_all_Articles.append(URL)
       
 
Final_URLs_all_Articles = list(set(URLs_all_Articles))   # Remove duplicate URLs
Final_URLs_all_Articles = [Main_Article] + Final_URLs_all_Articles
del  link,URL,NA,keep,Non_Articles, title,soup ,response, Main_Article,i    

# ********************************************************************************************************
    #########    Get text of all the wikipedia articles    ######==========-------------------------------
# ********************************************************************************************************
if Indicator_Data_Generator:    
    Final_URLs_all_Articles_TEXT =[]
    Problematic_URLs =[]
    for u in range(len(Final_URLs_all_Articles)):   
        One_URL = Final_URLs_all_Articles[u]
        try:
            RESPOND = requests.get( url= One_URL)
            B_Soup = BeautifulSoup(RESPOND.content, 'html.parser') 
            Text_URL = ''
            for p in B_Soup.find_all('p'):
                Text_URL = Text_URL + p.text    
            Final_URLs_all_Articles_TEXT.append(Text_URL) 
        except:                                                                             # Piazza In general, for things you cannot scrape, you can ignore. 
            print("Error reading the URL: ", u, "     ------     URL: ",One_URL)
            Problematic_URLs.append(u)
    # Save data
    a_file = open("Final_URLs_all_Articles_TEXT.pkl", "wb")
    pickle.dump(Final_URLs_all_Articles_TEXT, a_file)
    a_file.close()
else:
    # Load Data 
    Final_URLs_all_Articles_TEXT = pickle.load( open("Final_URLs_all_Articles_TEXT.pkl", "rb") )






#########   Question 1    ######==========-------------------------------
Vocab_All_words_Wiki = []
for i in range(len(Final_URLs_all_Articles_TEXT)):
    Wiki_Text = SpaCy_Model(Final_URLs_all_Articles_TEXT[i]) 
    # Sentence split + tokenize and Lemmatization + lower casing + remove stop words(also removed punctuation or white space)
    Cleaned_Tokens_text_link_i = [token.lemma_.lower() for token in Wiki_Text if not (token.is_stop or token.is_punct or token.is_space)]   
    Vocab_All_words_Wiki = Vocab_All_words_Wiki + Cleaned_Tokens_text_link_i

#   Question 1 - a
Vocab_All_words_Wiki_freq = Counter(Vocab_All_words_Wiki)
print("Top 20 words in the vocabulary:\n", Vocab_All_words_Wiki_freq.most_common(20))
del i, Wiki_Text, Cleaned_Tokens_text_link_i

#   Question 1 - b 
Wordcloud_Text = ''
for i in range(len(Final_URLs_all_Articles_TEXT)):
     Wordcloud_Text += Final_URLs_all_Articles_TEXT[i]
    
wordcloud = WordCloud(width = 2500, height = 2500, random_state=0, background_color='white', colormap='seismic', collocations=False, stopwords = STOPWORDS).generate(Wordcloud_Text)
plt.figure(figsize=(50, 50))
plt.imshow(wordcloud) 
plt.axis("off")




#########   Question 2    ######==========-------------------------------

# Load Tweets Data 
Data_Tweets = pickle.load( open("Data_10000_Twetes_RecentCOVID.pkl", "rb") ) 
X_train_Tweets, X_test_Tweets= train_test_split(Data_Tweets, test_size=0.1, random_state=1)

# Vocab Train Tweets
Vocab_All_words_Train_Tweets = []
for i in range(len(X_train_Tweets)):
    Tweet_Text = SpaCy_Model(X_train_Tweets[i].text) 
    # Sentence split + tokenize and Lemmatization + lower casing + remove stop words(also removed punctuation or white space)
    Cleaned_Tokens_text_tweet_i = [token.lemma_.lower() for token in Tweet_Text if not (token.is_stop or token.is_punct or token.is_space)]   
    Vocab_All_words_Train_Tweets = Vocab_All_words_Train_Tweets + Cleaned_Tokens_text_tweet_i
del i, Tweet_Text, Cleaned_Tokens_text_tweet_i,X_train_Tweets
# Vocab Test Tweets
Vocab_All_words_Test_Tweets = []
for i in range(len(X_test_Tweets)):
    Tweet_Text = SpaCy_Model(X_test_Tweets[i].text) 
    # Sentence split + tokenize and Lemmatization + lower casing + remove stop words(also removed punctuation or white space)
    Cleaned_Tokens_text_tweet_i = [token.lemma_.lower() for token in Tweet_Text if not (token.is_stop or token.is_punct or token.is_space)]   
    Vocab_All_words_Test_Tweets = Vocab_All_words_Test_Tweets + Cleaned_Tokens_text_tweet_i
del i, Tweet_Text, Cleaned_Tokens_text_tweet_i


#   Question 2 - a
Word_Types_Test_Tweets = set(Vocab_All_words_Test_Tweets)
Word_Types_Wiki = set(Vocab_All_words_Wiki)

print("\n\n*********  Question 2  **************************")
print("*********  Question 2-a:")
print('    Number of word types in your tweets that are out-of-vocabulary: ', len(Word_Types_Test_Tweets - Word_Types_Wiki))
print('    OOV-rate: ', len(Word_Types_Test_Tweets - Word_Types_Wiki)/len(Word_Types_Test_Tweets))

#   Question 2 - b
OOV_Types = list(Word_Types_Test_Tweets - Word_Types_Wiki)
Num_OOV_Tokens = 0
for i in range(len(Vocab_All_words_Test_Tweets)):
    if Vocab_All_words_Test_Tweets[i] in OOV_Types:
        Num_OOV_Tokens += 1
print("\n*********  Question 2-b:")
print('    Number of word token in your tweets that are out-of-vocabulary: ', Num_OOV_Tokens)
print('    OOV-rate: ', Num_OOV_Tokens/len(Vocab_All_words_Test_Tweets))

#   Question 2 - c
OOV_Type_Q2_c = list(Word_Types_Test_Tweets - set(Vocab_All_words_Train_Tweets))
Num_OOV_Tokens_Q2_c = 0
for i in range(len(Vocab_All_words_Test_Tweets)):
    if Vocab_All_words_Test_Tweets[i] in OOV_Type_Q2_c:
        Num_OOV_Tokens_Q2_c += 1
print("\n*********  Question 2-c:")
print('    Number of word token in your tweets that are out-of-vocabulary: ', Num_OOV_Tokens_Q2_c)
print('    OOV-rate: ', Num_OOV_Tokens_Q2_c/len(Vocab_All_words_Test_Tweets))



#########   Question 3    ######==========-------------------------------

# First_9000: sentence segmenting + lower casing + Tokenization + Padding using function  NLP_PreProcessing
Preprocessed_First_9000_wiki = []
for t in range(len(Final_URLs_all_Articles_TEXT)):
    Preprocessed_First_9000_wiki = Preprocessed_First_9000_wiki + NLP_PreProcessing(Final_URLs_all_Articles_TEXT[t]) 
    if len(Preprocessed_First_9000_wiki) >= 9000:
        break
Preprocessed_First_9000_wiki = Preprocessed_First_9000_wiki[:9000]

 # Train Trigram
train_data_Trigram, padded_sents_Trigram = padded_everygram_pipeline( 3 , Preprocessed_First_9000_wiki)
Trigram_Model = KneserNeyInterpolated(3) 
Trigram_Model.fit(train_data_Trigram, padded_sents_Trigram)

print('Number of Ngram: ', Trigram_Model.counts)


#   Question 3 - a
# 1000 tweets: sentence segmenting + lower casing + Tokenization + Padding using function  NLP_PreProcessing
Test_Preprocessed_1000_Tweets = []
for t in range(len(X_test_Tweets)):
    Test_Preprocessed_1000_Tweets = Test_Preprocessed_1000_Tweets + NLP_PreProcessing(X_test_Tweets[t].text) 

# Average perplexity
Total_Tri_Perplex = 0
for tt in range(len(Test_Preprocessed_1000_Tweets)):
    # Trigram
    test_data_Trigram = list(ngrams(Test_Preprocessed_1000_Tweets[tt], n=3,pad_left=True, pad_right=True,left_pad_symbol='<s>',right_pad_symbol='</s>'))
    Tri_Preplex = Trigram_Model.perplexity(test_data_Trigram)
    Total_Tri_Perplex = Total_Tri_Perplex + Tri_Preplex

print("**********************************************\n")
print("Average perplexities :  ",  (Total_Tri_Perplex/len(Test_Preprocessed_1000_Tweets)))
print("**********************************************\n")



























