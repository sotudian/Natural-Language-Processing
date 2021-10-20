#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

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
import csv
from matplotlib import pyplot
from sklearn.decomposition import PCA
import scipy
from sklearn.metrics.pairwise import cosine_similarity as CosineSimilarity
# Functions ###########================-------------------
    
def NLP_PreProcessing(text_main):
    # lower casing
    text = text_main.lower()    
    # sentence segmenting    
    sentences = nltk.sent_tokenize(text)       
    # Tokenization      
    Tokenized_sentences = [word_tokenize(S) for S in sentences] 
    return Tokenized_sentences

    
    
def My_create_cooccurrence_matrix(Lines,Vocabs_List):
    Vocabulary = { W:inx for inx,W in enumerate(Vocabs_List)}
    Value = []
    Row = []
    Col = []
    tokenizer = nltk.tokenize.word_tokenize
    for Line in Lines:
        Line = Line.lower() # lower casing
        tokens = [token for token in tokenizer(Line)] #  Tokenization  
        for pos1, token in enumerate(tokens):
            try:
                Px = Vocabulary[token]
                for pos2 in range(0, len(tokens)):
                    if pos2 == pos1:
                        continue
                    try:
                        Py = Vocabulary[tokens[pos2]]
                        Row.append(Px)
                        Col.append(Py)
                        Value.append(1.)
                    except:
                        continue
            except:
                continue

    CO_OCC_Matrix_Sparse = scipy.sparse.coo_matrix((Value, (Row, Col)))
    return Vocabulary, CO_OCC_Matrix_Sparse

def Play_Representation(Play_Text,DF_My_CO_OCC_Matrix):
    Vec_rep = np.zeros((len(DF_My_CO_OCC_Matrix)))
    Play_Text = Play_Text.lower() # lower casing
    Play_TOKENS = [w for w in (nltk.word_tokenize(Play_Text))] # split, tokenize
    LENGTH = len(Play_TOKENS)
    for Token in Play_TOKENS:    
        try:
            Rep_Token = np.array(DF_My_CO_OCC_Matrix.iloc[DF_My_CO_OCC_Matrix.index.isin([Token])][:].values).T.ravel()
            Vec_rep = np.add(Vec_rep,Rep_Token)
        except:
            LENGTH = LENGTH -1                 
    if LENGTH > 0:    
        Vec_rep = Vec_rep/LENGTH
    return Vec_rep    

def Generate_DF_Cos_Similarity(Play_Namess,DF_Play_Representation):
    Similarity_Matrix = np.zeros((len(Play_Namess),len(Play_Namess)))
    for i, Play1 in enumerate(Play_Namess): 
        for j, Play2 in enumerate(Play_Namess):
            V1 = DF_Play_Representation[Play1].values
            V2 = DF_Play_Representation[Play2].values
            Dist = CosineSimilarity(V1.reshape(1, -1), V2.reshape(1, -1))[0][0]
            Similarity_Matrix[i,j] = Dist
    DF_Similarity_Matrix = pd.DataFrame(Similarity_Matrix,index=Play_Namess,columns = Play_Namess)
    return DF_Similarity_Matrix

def Play_Rep_Word2Vec(Play_Text,My_Word2Vec_Plays, dim_size):
    Vec_rep = np.zeros((dim_size))
    Play_Text = Play_Text.lower() # lower casing
    Play_TOKENS = [w for w in (nltk.word_tokenize(Play_Text))] # split, tokenize
    LENGTH = len(Play_TOKENS)
    for Token in Play_TOKENS:    
        try:
            Rep_Token = My_Word2Vec_Plays.wv[Token]
            Vec_rep = np.add(Vec_rep,Rep_Token)
        except:
            LENGTH = LENGTH -1                 
    if LENGTH > 0:    
        Vec_rep = Vec_rep/LENGTH
    return Vec_rep    

def FindPosition(DF, value): 
    LIST = [] 
    P = DF.isin([value]) 
    seriesObj = P.any() 
    COL_Names = list(seriesObj[seriesObj == True].index) 
    for col in COL_Names: 
        rows = list(P[col][P[col] == True].index) 
  
        for row in rows: 
            LIST.append((row, col)) 
    return LIST 

# Data ###########================-------------------
will_play_text = pd.read_csv('will_play_text.csv', sep=';', header=None)

Vocabs_List =[]
with open('vocab.txt', mode='r') as csvfile1:
   spamreader1 = csv.reader(csvfile1, delimiter=';')
   for row in spamreader1:
       Vocabs_List.append(row[0])
       
Play_Names_List =[]
with open('play_names.txt', mode='r') as csvfile2:
   spamreader2 = csv.reader(csvfile2, delimiter=';')
   for row in spamreader2:
       Play_Names_List.append(row[0])       

del spamreader1,row,csvfile1,spamreader2,csvfile2




#   Q3.1 ###########================-------------------
Play_Text_DF = pd.DataFrame( {i:[' '.join(will_play_text[will_play_text[1] == i].iloc[:][5])] for i in Play_Names_List})
My_CountVectorizer = CountVectorizer(vocabulary=Vocabs_List)
Document_Vector = My_CountVectorizer.fit_transform(Play_Text_DF.iloc[0])
Term_Document_Matrix_Count = pd.DataFrame(Document_Vector.toarray().transpose(),index=My_CountVectorizer.get_feature_names())
Term_Document_Matrix_Count.columns = Play_Text_DF.columns


# Q3.2 ###########================-------------------
pca = PCA(n_components=2)
PC_Count = pca.fit_transform(Term_Document_Matrix_Count.transpose())
pyplot.figure(figsize=(30, 15), dpi=500)
pyplot.scatter(PC_Count[:, 0], PC_Count[:, 1], color = 'b')	
words = list(Term_Document_Matrix_Count.columns)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(PC_Count[i][0], PC_Count[i][1]), fontsize=10)
pyplot.savefig('Q_3_2.png')
    
del i, word, words, pca , Document_Vector ,My_CountVectorizer 


# Q3.3 ###########================-------------------    
My_TfidfVectorizer = TfidfVectorizer(vocabulary=Vocabs_List)
Document_Vector_Tfidf = My_TfidfVectorizer.fit_transform(Play_Text_DF.iloc[0])
Term_Document_Matrix_Tfidf = pd.DataFrame(Document_Vector_Tfidf.toarray().transpose(),index=My_TfidfVectorizer.get_feature_names())
Term_Document_Matrix_Tfidf.columns = Play_Text_DF.columns
   

# Q3.4   ###########================-------------------  
pca = PCA(n_components=2)
PC_Tfidf = pca.fit_transform(Term_Document_Matrix_Tfidf.transpose())
pyplot.figure(figsize=(12, 15), dpi=500)
pyplot.scatter(PC_Tfidf[:, 0], PC_Tfidf[:, 1], color = 'r')	
words = list(Term_Document_Matrix_Tfidf.columns)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(PC_Tfidf[i][0], PC_Tfidf[i][1]), fontsize=10)
pyplot.savefig('Q_3_4.png')
    
del i, word, words, pca , Document_Vector_Tfidf ,My_TfidfVectorizer 




# Q3.5 ###########================-------------------
List_All_Sentences = list(will_play_text[5])
MY_Vocabs,My_CO_OCC_Matrix = My_create_cooccurrence_matrix(List_All_Sentences,Vocabs_List)
DF_My_CO_OCC_Matrix  = pd.DataFrame(My_CO_OCC_Matrix.todense(),
                          index=MY_Vocabs.keys(),
                          columns = MY_Vocabs.keys())
DF_My_CO_OCC_Matrix = DF_My_CO_OCC_Matrix.sort_index()[sorted(MY_Vocabs.keys())]


# Q3.6 ###########================-------------------
# vector representation of each play
DF_Play_Representation = pd.DataFrame()
for Play in Play_Text_DF.columns:
    Vec_Play_Representation = Play_Representation(Play_Text_DF[Play][0],DF_My_CO_OCC_Matrix)
    DF_Play_Representation[Play] = Vec_Play_Representation

del Play,Vec_Play_Representation



Comedies = ['The Tempest','Two Gentlemen of Verona','Taming of the Shrew',
            'Alls well that ends well', 'Loves Labours Lost','A Winters Tale',
            'A Midsummer nights dream','Merry Wives of Windsor','As you like it',
            'A Comedy of Errors','Measure for measure','Much Ado about nothing',
            'Twelfth Night', 'Merchant of Venice','Pericles']




DF_Similarity_Matrix_Comedies = Generate_DF_Cos_Similarity(Comedies,DF_Play_Representation)
print("Average pairwise cosine-similarity between plays - Comedies: ",((np.triu(DF_Similarity_Matrix_Comedies, 1)).sum())/((len(Comedies)**2-len(Comedies))/2))

        
Histories = ['Henry IV','Henry VIII','Henry V','Richard III',
             'Henry VI Part 3','Henry VI Part 1', 'Richard II',
             'King John', 'Henry VI Part 2']

DF_Similarity_Matrix_Histories = Generate_DF_Cos_Similarity(Histories,DF_Play_Representation)
print("Average pairwise cosine-similarity between plays - Histories: ",((np.triu(DF_Similarity_Matrix_Histories, 1)).sum())/((len(Histories)**2-len(Histories))/2))



Tragedies = ['Antony and Cleopatra', 'Coriolanus','Hamlet','macbeth',
           'Romeo and Juliet', 'Julius Caesar','King Lear',
           'Timon of Athens', 'Troilus and Cressida',
           'Othello', 'Cymbeline', 'Titus Andronicus']
DF_Similarity_Matrix_Tragedies = Generate_DF_Cos_Similarity(Tragedies,DF_Play_Representation)
print("Average pairwise cosine-similarity between plays - Tragedies: ",((np.triu(DF_Similarity_Matrix_Tragedies, 1)).sum())/((len(Tragedies)**2-len(Tragedies))/2))


# Q3.7 ###########================-------------------

# sentence segmenting + lower casing + Tokenization using function  NLP_PreProcessing
Preprocessed_Text_Plays = []
for t in range(len(List_All_Sentences)):
    Preprocessed_Text_Plays = Preprocessed_Text_Plays + NLP_PreProcessing(List_All_Sentences[t]) 

My_Word2Vec_Plays = Word2Vec(sentences=Preprocessed_Text_Plays, size=100, min_count = 1) # Creating Word2Vec Model

# vector representation of Plays

DF_Play_Rep_Word2Vec = pd.DataFrame()
for Play in Play_Text_DF.columns:
    Vec_Play_Rep = Play_Rep_Word2Vec(Play_Text_DF[Play][0],My_Word2Vec_Plays,dim_size = 100)
    DF_Play_Rep_Word2Vec[Play] = Vec_Play_Rep

del Play,Vec_Play_Rep,t


DF_Similarity_Matrix_Comedies_Word2Vec = Generate_DF_Cos_Similarity(Comedies,DF_Play_Rep_Word2Vec)
print("Average pairwise cosine-similarity between plays using Word2Vec - Comedies: ",((np.triu(DF_Similarity_Matrix_Comedies_Word2Vec, 1)).sum())/((len(Comedies)**2-len(Comedies))/2))

DF_Similarity_Matrix_Histories_Word2Vec = Generate_DF_Cos_Similarity(Histories,DF_Play_Rep_Word2Vec)
print("Average pairwise cosine-similarity between plays using Word2Vec - Histories: ",((np.triu(DF_Similarity_Matrix_Histories_Word2Vec, 1)).sum())/((len(Histories)**2-len(Histories))/2))

DF_Similarity_Matrix_Tragedies_Word2Vec = Generate_DF_Cos_Similarity(Tragedies,DF_Play_Rep_Word2Vec)
print("Average pairwise cosine-similarity between plays using Word2Vec - Tragedies: ",((np.triu(DF_Similarity_Matrix_Tragedies_Word2Vec, 1)).sum())/((len(Tragedies)**2-len(Tragedies))/2))


# Q3.8 ###########================-------------------


Characters = [x for x in list(will_play_text[4].unique()) if str(x) != 'nan']

Characters_Text_DF = pd.DataFrame( {i:[' '.join(will_play_text[will_play_text[4] == i].iloc[:][5])] for i in Characters})

DF_Characters_Rep_Word2Vec = pd.DataFrame()
for character in Characters_Text_DF.columns:
    Vec_Characters_Rep = Play_Rep_Word2Vec(Characters_Text_DF[character][0],My_Word2Vec_Plays,dim_size = 100)
    DF_Characters_Rep_Word2Vec[character] = Vec_Characters_Rep

del character,Vec_Characters_Rep


pca = PCA(n_components=2)
PC_Characters = pca.fit_transform(DF_Characters_Rep_Word2Vec.transpose())
pyplot.figure(figsize=(35, 40), dpi=100)
pyplot.scatter(PC_Characters[:, 0], PC_Characters[:, 1], color = 'g')	
words = list(DF_Characters_Rep_Word2Vec.columns)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(PC_Characters[i][0], PC_Characters[i][1]), fontsize=10)
pyplot.savefig('Q_3_8.png')
    
del i, word, words, pca   





# Q3.9 ###########================-------------------
# what are characters that are most similar/dissimilar to each other?
DF_Similarity_Matrix_Characters = Generate_DF_Cos_Similarity(list(DF_Characters_Rep_Word2Vec.columns),DF_Characters_Rep_Word2Vec)

print("Most dissimilar characters:")
PP = np.reshape(DF_Similarity_Matrix_Characters.values, 934*934)
sorted_index_array = np.argsort(PP) 
sorted_PP = PP[sorted_index_array] 
rslt = sorted_PP[1:50 : ]    # we want 25 smallest value
for i, Val in enumerate(list(set(rslt))):
    print('Num ', (i+1), ': ', (FindPosition(DF_Similarity_Matrix_Characters, Val))[0] , ' ----  Similarity: ',Val)

del PP,sorted_index_array,sorted_PP,rslt,Val,i

# Find 25 most similar Characters
DF_Similarity_Matrix_Characters.values[[np.arange(DF_Similarity_Matrix_Characters.shape[0])]*2] = 0   # Set zero diagonal elements
PP = np.reshape(DF_Similarity_Matrix_Characters.values, 934*934)
sorted_index_array = np.argsort(PP) 
sorted_PP = PP[sorted_index_array] 
rslt = sorted_PP[-50 : ]    # we want 25 largest value
print("Most similar characters:")
for i, Val in enumerate(list(set(rslt))):
    print('Num ', (i+1), ': ', (FindPosition(DF_Similarity_Matrix_Characters, Val))[0] , ' ----  Similarity: ',Val)

del PP,sorted_index_array,sorted_PP,rslt,Val,i






# Compare MALE-FEMALE  - I chose 18 males and 18 females
DF_PC_Characters = pd.DataFrame(PC_Characters, columns=(['PC1','PC2']))
DF_PC_Characters['Characters_Names'] = list(DF_Characters_Rep_Word2Vec.columns)

Female_Characters = ['BEATRICE','BIANCA','CORDELIA','CRESSIDA','DESDEMONA',
                     'EMILIA','GONERIL','HERMIA','JESSICA','JULIET',
                     'VIOLA','ROSALINE','PORTIA','REGAN',
                     'MARIA','MIRANDA','OLIVIA','PERDITA'] 

Male_Characters = ['THOMAS MOWBRAY','WILLIAM PAGE', 'HENRY BOLINGBROKE','SIR TOBY BELCH',
                   'BRABANTIO','KING JOHN','KING RICHARD II','KING RICHARD III','of King Henry VI',
                   'COSTARD','DONALBAIN','of Prince Edward','FERDINAND',
                   'JOHN OF GAUNT','MACBETH','CAESAR','OTHELLO','ROMEO'] 
                   

fig, ax = pyplot.subplots(figsize=(10, 10), dpi=200)

for color in [['tab:blue','Males'], ['tab:orange','Females']]:
    if color[1] == 'Males':
        X = (DF_PC_Characters[:][DF_PC_Characters['Characters_Names'].isin(Male_Characters)])['PC1'].values
        Y = (DF_PC_Characters[:][DF_PC_Characters['Characters_Names'].isin(Male_Characters)])['PC2'].values
        words = (DF_PC_Characters[:][DF_PC_Characters['Characters_Names'].isin(Male_Characters)])['Characters_Names'].values
    else:
        X = (DF_PC_Characters[:][DF_PC_Characters['Characters_Names'].isin(Female_Characters)])['PC1'].values
        Y = (DF_PC_Characters[:][DF_PC_Characters['Characters_Names'].isin(Female_Characters)])['PC2'].values
        words = (DF_PC_Characters[:][DF_PC_Characters['Characters_Names'].isin(Female_Characters)])['Characters_Names'].values        
    
    ax.scatter(X, Y, c=color[0], label=color[1], edgecolors='none')   #,alpha=0.3, s=scale
    for i, word in enumerate(words):
        ax.annotate(word, xy=(X[i], Y[i]), fontsize=10)

ax.legend()
ax.grid(True)

pyplot.show()
fig.savefig('Q_3_9_1.png')

del color, X,Y,i,word,words,ax,fig


# Can you find plays that are central to each category (i.e., comedies, histories, tragedies)?
pca = PCA(n_components=2)
PC_Play_Word2Vec = pca.fit_transform(DF_Play_Rep_Word2Vec.transpose())

DF_PC_Plays = pd.DataFrame(PC_Play_Word2Vec, columns=(['PC1','PC2']))
DF_PC_Plays['Play_Names'] = list(DF_Play_Rep_Word2Vec.columns)

fig, ax = pyplot.subplots(figsize=(15, 15), dpi=200)

for color in [['tab:blue','Comedies'], ['tab:green','Histories'], ['tab:orange','Tragedies']]:
    if color[1] == 'Comedies':
        X = (DF_PC_Plays[:][DF_PC_Plays['Play_Names'].isin(Comedies)])['PC1'].values
        Y = (DF_PC_Plays[:][DF_PC_Plays['Play_Names'].isin(Comedies)])['PC2'].values
        words = (DF_PC_Plays[:][DF_PC_Plays['Play_Names'].isin(Comedies)])['Play_Names'].values
    elif color[1] == 'Histories':
        X = (DF_PC_Plays[:][DF_PC_Plays['Play_Names'].isin(Histories)])['PC1'].values
        Y = (DF_PC_Plays[:][DF_PC_Plays['Play_Names'].isin(Histories)])['PC2'].values
        words = (DF_PC_Plays[:][DF_PC_Plays['Play_Names'].isin(Histories)])['Play_Names'].values        
    else:
        X = (DF_PC_Plays[:][DF_PC_Plays['Play_Names'].isin(Tragedies)])['PC1'].values
        Y = (DF_PC_Plays[:][DF_PC_Plays['Play_Names'].isin(Tragedies)])['PC2'].values
        words = (DF_PC_Plays[:][DF_PC_Plays['Play_Names'].isin(Tragedies)])['Play_Names'].values        
    
    ax.scatter(X, Y, c=color[0], label=color[1], edgecolors='none')   #,alpha=0.3, s=scale
    for i, word in enumerate(words):
        ax.annotate(word, xy=(X[i], Y[i]), fontsize=10)

ax.legend()
ax.grid(True)

pyplot.show()
fig.savefig('Q_3_9_2.png')




fig, ax = pyplot.subplots(figsize=(15, 15), dpi=500)
itr = 3
for color in [['tab:blue','Comedies'], ['tab:green','Histories'], ['tab:orange','Tragedies']]:
    pyplot.figure(figsize=(10, 10), dpi=500)
    if color[1] == 'Comedies':
        X = (DF_PC_Plays[:][DF_PC_Plays['Play_Names'].isin(Comedies)])['PC1'].values
        Y = (DF_PC_Plays[:][DF_PC_Plays['Play_Names'].isin(Comedies)])['PC2'].values
        words = (DF_PC_Plays[:][DF_PC_Plays['Play_Names'].isin(Comedies)])['Play_Names'].values
    elif color[1] == 'Histories':
        X = (DF_PC_Plays[:][DF_PC_Plays['Play_Names'].isin(Histories)])['PC1'].values
        Y = (DF_PC_Plays[:][DF_PC_Plays['Play_Names'].isin(Histories)])['PC2'].values
        words = (DF_PC_Plays[:][DF_PC_Plays['Play_Names'].isin(Histories)])['Play_Names'].values        
    else:
        X = (DF_PC_Plays[:][DF_PC_Plays['Play_Names'].isin(Tragedies)])['PC1'].values
        Y = (DF_PC_Plays[:][DF_PC_Plays['Play_Names'].isin(Tragedies)])['PC2'].values
        words = (DF_PC_Plays[:][DF_PC_Plays['Play_Names'].isin(Tragedies)])['Play_Names'].values        
    
    pyplot.scatter(X, Y, c=color[0], label=color[1], edgecolors='none')   #,alpha=0.3, s=scale
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(X[i], Y[i]), fontsize=10)
    
    pyplot.show()
    Name_file = 'Q_3_9_' + str(itr) + '.png'
    fig.savefig(Name_file)
    itr +=1






del pca,color,PC_Play_Word2Vec, X,Y,i,word,words,ax,fig,itr



















