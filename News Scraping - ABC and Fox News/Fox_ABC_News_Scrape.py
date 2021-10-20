#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scrape ABC and Fox News articles from their sitemaps. Extract the text of the articles, then sentence split,
tokenize, and remove stop words using spacy.

@author: shahab Sotudian
"""



import re
from bs4 import BeautifulSoup
import requests
from newspaper import Article
import pickle
import spacy
import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS
import string   
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import string


# Functions ###########================-------------------

def Text_Of_Articles(List_Of_URLs):
    Text_Of_Articles = []
    for url in List_Of_URLs:
        News_Article = Article(url, language='en')
        News_Article.download()
        News_Article.parse()
        Text_Of_Articles.append(News_Article.text)
    return Text_Of_Articles
        

# Generate Or Load Data  ###########================-------------------

Do_You_Want_Generate_Data = 0 # Indicator to generate data or read the saved data


if Do_You_Want_Generate_Data == 1: # Generate data or read it
    ### Scrape Fox and ABC News sitemaps for URLs  @@@=====---------------
    # Fox News
    Fox_News = "https://www.foxnews.com/sitemap.xml?type=news"
    response = requests.get(  url= Fox_News )
    soup = BeautifulSoup(response.content,"xml") 
    urls = soup.findAll('url')
    Fox_News_Urls = []
    for url in urls:
        Ui = re.findall("<loc>.*?</loc>", str(url))[0]
        Ui = Ui.replace('<loc>', '') 
        Ui = Ui.replace('</loc>', '') 
        Fox_News_Urls.append(Ui)
    
    del url, urls,response, soup,Ui, Fox_News
    # ABC News - I run this part several times
    ABC_News =  "https://abcnews.go.com/xmlLatestStories"
    response = requests.get(  url= ABC_News )
    soup = BeautifulSoup(response.content,'html.parser') 
    urls = soup.findAll('url')
    ABC_News_Urls = []
    for url in urls:
        Ui = re.findall("<loc>.*?</loc>", str(url))[0]
        Ui = Ui.replace('<loc>', '') 
        Ui = Ui.replace('</loc>', '') 
        ABC_News_Urls.append(Ui)
    del url, urls,response, soup,Ui, ABC_News
    
    ### Scrape texts of Fox and ABC News articles 
    # Fox News
    Fox_News_Texts = Text_Of_Articles(Fox_News_Urls)
    # ABC News
    ABC_News_Texts = Text_Of_Articles(ABC_News_Urls)
    
    ### Save data  
    # Fox News
    a_file = open("Final_Fox_News_Texts.pkl", "wb")
    pickle.dump(Fox_News_Texts, a_file)
    a_file.close()
    
    a1_file = open("Final_Fox_News_URLs.pkl", "wb")
    pickle.dump(Fox_News_Urls, a1_file)
    a1_file.close()
    
    # ABC News
    b_file = open("Final_ABC_News_Texts.pkl", "wb")
    pickle.dump(ABC_News_Texts, b_file)
    b_file.close()
    
    b1_file = open("Final_ABC_News_URLs.pkl", "wb")
    pickle.dump(ABC_News_Urls, b1_file)
    b1_file.close()
    
    del a_file,a1_file,b_file,b1_file

else:
    # Load Data 
    Fox_News_Texts = pickle.load( open("Final_Fox_News_Texts.pkl", "rb") )
    Fox_News_Urls = pickle.load( open("Final_Fox_News_URLs.pkl", "rb") )
    ABC_News_Texts = pickle.load( open("Final_ABC_News_Texts.pkl", "rb") )
    ABC_News_Urls = pickle.load( open("Final_ABC_News_URLs.pkl", "rb") )





#########   Question 1    ######==========-------------------------------

# Ssentence split, tokenize, and remove stop words
stop_words = set(stopwords.words('english'))

 ### Concat all texts and extract Token-Type @@@=====---------------
# Fox News
All_Fox_News_Texts = ''
for text in Fox_News_Texts:
    All_Fox_News_Texts += text
     # split, tokenize, and remove stop words
Fox_News_TOKENS = [w for w in (nltk.word_tokenize(All_Fox_News_Texts)) if not w in stop_words]
Fox_News_Types = nltk.Counter(Fox_News_TOKENS)


# ABC News
All_ABC_News_Texts = ''
for text in ABC_News_Texts:
    All_ABC_News_Texts += text
 # split, tokenize, and remove stop words
ABC_News_TOKENS = [w for w in (nltk.word_tokenize(All_ABC_News_Texts)) if not w in stop_words] 
ABC_News_Types = nltk.Counter(ABC_News_TOKENS)

del text


 ### Plotting Token-Type graph  @@@=====---------------
# Fox News
x_Fox_News = [] 
y_Fox_News = []
for i in range(len(Fox_News_TOKENS)): 
    print(i)
    Types = (nltk.Counter(Fox_News_TOKENS[:(i+1)]))
    x_Fox_News.append((i+1))
    y_Fox_News.append(len(Types))
    
plt.plot(x_Fox_News, y_Fox_News)
plt.xlabel('Number of Tokens') 
plt.ylabel('Number of Types') 
plt.title('Word Type-Token graph of Fox News') 
plt.savefig('Fig_Type_Token_Fox_News.png', dpi=1000)
plt.show()

del Types,i

# ABC News
x_ABC_News = [] 
y_ABC_News = []
for i in range(len(ABC_News_TOKENS)): 
    Types = (nltk.Counter(ABC_News_TOKENS[:(i+1)]))
    x_ABC_News.append((i+1))
    y_ABC_News.append(len(Types))
    
plt.plot(x_ABC_News, y_ABC_News)
plt.xlabel('Number of Tokens') 
plt.ylabel('Number of Types') 
plt.title('Word Type-Token graph of ABC News') 
plt.savefig('Fig_Type_Token_ABC_News.png', dpi=1000)
plt.show()

del Types,i





#########   Question 2    ######==========-------------------------------

# Construct the word clouds from the two texts.

# Fox News
wordcloud_Fox_News = WordCloud(width = 2500, height = 2500, random_state=0, background_color='black', colormap='twilight', collocations=False, stopwords = stop_words).generate(All_Fox_News_Texts)
plt.figure(figsize=(50, 50))
plt.title('Word cloud of Fox News') 
plt.imshow(wordcloud_Fox_News) 
plt.axis("off")


# ABC News
wordcloud_ABC_News = WordCloud(width = 2500, height = 2500, random_state=0, background_color='black', colormap='twilight', collocations=False, stopwords = stop_words).generate(All_ABC_News_Texts)
plt.figure(figsize=(50, 50))
plt.title('Word cloud of ABC News') 
plt.imshow(wordcloud_ABC_News) 
plt.axis("off")












































