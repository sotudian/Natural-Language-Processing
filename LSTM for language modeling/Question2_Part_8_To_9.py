#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shahab Sotudian
"""

import re
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.lm.preprocessing import pad_both_ends
from collections import Counter
import math
# Functions ###########================-------------------
'''
############################################################
#### Piazza calculate Preplexity
net.cuda()
net.eval()
H = 0
TOTAL_PROBs = 1
with torch.no_grad():
    for Test_Sentence in Test_1_Preprocessed_Pride_Text_Perplexity:
        H += len(Test_Sentence)
        # Calculate for each sentence
        Total_prob_Sentence = 1
        for i,word in enumerate(Test_Sentence):
            if i == len(Test_Sentence)-1:
                continue
            else:
                if i==0:
                    h = net.init_hidden(1)
                    h = tuple([each.data for each in h])
                else:
                    h = h_new
                    
                x = np.array([[word2idx[word]]])
                inputs = torch.from_numpy(x)
                inputs = inputs.cuda()
                
                out, h_new = net(inputs, h)
                # get the token probabilities
                p = F.softmax(out, dim=1).data
                p = p.cpu()
                p = p.numpy()
                p = p.reshape(p.shape[1],)
                Prob_next_Word = p[word2idx[Test_Sentence[i+1]]]  # P(w4|w1,w2,w3)
                Total_prob_Sentence = Prob_next_Word * Total_prob_Sentence
            
    TOTAL_PROBs = TOTAL_PROBs * Total_prob_Sentence

Preplexity = (1/TOTAL_PROBs)**(1/float(H)) 
############################################################
'''
def NLP_PreProcessing(text_main):    
    # sentence segmenting    
    sentences = nltk.sent_tokenize(text_main)       
    # Tokenization + lower casing    
    Tokenized_sentences = [word_tokenize(S.lower()) for S in sentences] 
    # Padding 
    Pad_Tokenized_sentences = [list(pad_both_ends(TS, n=2)) for TS in Tokenized_sentences]
    
    return Pad_Tokenized_sentences

def NLP_PreProcessing_Test(text_main):       
    # Tokenization + lower casing    
    Tokenized_sentences = word_tokenize(text_main.lower())
    # Padding 
    Pad_Tokenized_sentences = [list(pad_both_ends(Tokenized_sentences, n=2))]
    
    return Pad_Tokenized_sentences  
    
def Equal_seq(text, seq_len):
    sequences = []
    if len(text) > seq_len:
      for i in range(seq_len, (len(text)+1)):
        seq = text[i-seq_len:i]
        sequences.append(seq)
    else:
        sequences = [['_PAD']*(seq_len-len(text)) + text ]
    
    return sequences    
    
#----
def get_batches(arr_x, arr_y, batch_size):
         
    # iterate through the arrays
    prv = 0
    for n in range(batch_size, arr_x.shape[0], batch_size):
      x = arr_x[prv:n,:]
      y = arr_y[prv:n,:]
      prv = n
      yield x, y
      
      
class WordLSTM(nn.Module):
    
    def __init__(self, n_hidden=256, n_layers=4, drop_prob=0.3, lr=0.001):
        super().__init__()

        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        self.emb_layer = nn.Embedding(vocab_size, 200)

        ## define the LSTM
        self.lstm = nn.LSTM(200, n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        ## define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        ## define the fully-connected layer
        self.fc = nn.Linear(n_hidden, vocab_size)      
    
    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''

        ## pass input through embedding layer
        embedded = self.emb_layer(x)     
        
        ## Get the outputs and the new hidden state from the lstm
        lstm_output, hidden = self.lstm(embedded, hidden)
        
        ## pass through a dropout layer
        out = self.dropout(lstm_output)
        
        #out = out.contiguous().view(-1, self.n_hidden) 
        out = out.reshape(-1, self.n_hidden) 

        ## put "out" through the fully-connected layer
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        # if GPU is available
        if (torch.cuda.is_available()):
          hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                    weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        
        # if GPU is not available
        else:
          hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                    weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden      
      
def train(net, epochs, batch_size, lr, clip, print_every,XX,YY):
    
    # optimizer
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    
    # loss
    criterion = nn.CrossEntropyLoss()
    
    # push model to GPU
    net.cuda()
    
    counter = 0

    net.train()

    for e in range(epochs):

        # initialize hidden state
        h = net.init_hidden(batch_size)
        
        for x, y in get_batches(XX, YY, batch_size):
            counter+= 1
            
            # convert numpy arrays to PyTorch arrays
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            # push tensors to GPU
            inputs, targets = inputs.cuda(), targets.cuda()

            # detach hidden states
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()
            
            # get the output from the model
            output, h = net(inputs, h)
            
            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(-1))

            # back-propagate error
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)

            # update weigths
            opt.step()            
            
            if counter % print_every == 0:
            
              print("Epoch: {}/{}...".format(e+1, epochs),
                    "Step: {}...".format(counter))      
      
def predict(net, tkn, h=None, word2idx_Inp = None, idx2word_Inp =None ):  
  # tensor inputs
  x = np.array([[word2idx_Inp[tkn]]])
  inputs = torch.from_numpy(x)
  
  # push to GPU
  inputs = inputs.cuda()

  # detach hidden state from history
  h = tuple([each.data for each in h])

  # get the output of the model
  out, h = net(inputs, h)

  # get the token probabilities
  p = F.softmax(out, dim=1).data

  p = p.cpu()

  p = p.numpy()
  p = p.reshape(p.shape[1],)

  # get indices of top 3 values
  top_n_idx = p.argsort()[-3:][::-1]

  # randomly select one of the three indices
  sampled_token_index = top_n_idx[random.sample([0,1,2],1)[0]]

  # return the encoded value of the predicted char and the hidden state
  return idx2word_Inp[sampled_token_index], h


# function to generate text
def sample(net, size, prime="<s>",word2idx_Inp = None, idx2word_Inp =None ):
        
    # push to GPU
    net.cuda()
    
    net.eval()

    # batch size is 1
    h = net.init_hidden(1)

    toks = prime.split()

    # predict next token
    for t in prime.split():
      token, h = predict(net, t, h,word2idx_Inp,idx2word_Inp)
    
    toks.append(token)

    # predict subsequent tokens
    if size == '</s>':
        while(token!='</s>'):
            token, h = predict(net, toks[-1], h,word2idx_Inp,idx2word_Inp)
            toks.append(token)
    else: 
        for i in range(size-1):
            token, h = predict(net, toks[-1], h,word2idx_Inp,idx2word_Inp)
            toks.append(token)

    return ' '.join(toks)    







def Testing(net, batch_size,Test_X,Test_Y):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    # initialize hidden state
    h = net.init_hidden(batch_size)
    test_loss = 0.
    
    with torch.no_grad():
        for x, y in get_batches(Test_X, Test_Y, batch_size):
            # convert numpy arrays to PyTorch arrays
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)            
            # push tensors to GPU
            inputs, targets = inputs.cuda(), targets.cuda()    
            # detach hidden states
            h = tuple([each.data for each in h])
            # get the output from the model
            output, h = net(inputs, h)            
            test_loss += criterion(output, targets.view(-1)).item()

    test_loss = test_loss / (len(Test_X) // batch_size)
    print('-' * 40)
    print('Test loss {:5.2f}  ------  Test perplexity {:8.2f}'.format(test_loss, math.exp(test_loss)))
    print('-' * 40)

# ---
#    def create_emb_layer(weights_matrix, non_trainable=False):
#     num_embeddings, embedding_dim = weights_matrix.size()
#     emb_layer = nn.Embedding(num_embeddings, embedding_dim)
#     emb_layer.load_state_dict({'weight': weights_matrix})
#     if non_trainable:
#         emb_layer.weight.requires_grad = False

#     return emb_layer, num_embeddings, embedding_dim

# class ToyNN(nn.Module):
#     def __init__(self, weights_matrix, hidden_size, num_layers):
#         super(self).__init__()
#         self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        
#     def forward(self, inp, hidden):
#         return self.gru(self.embedding(inp), hidden)
    
#     def init_hidden(self, batch_size):
#         return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

# ----
      
class WordLSTM_with_Glove(nn.Module): 
   
    def __init__(self, n_hidden=256, n_layers=4, drop_prob=0.3, lr=0.001):
        super().__init__()

        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        self.emb_layer = nn.Embedding(vocab_size_Q6,100, padding_idx=0) 
        self.emb_layer.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.emb_layer.weight.requires_grad = False ## freeze embeddings       
        '''
                self.emb_layer = nn.Embedding(vocab_size_Q6,100) 
                self.emb_layer.weight = nn.Parameter(torch.from_numpy(embedding_matrix).float())
        '''
        ## define the LSTM
        self.lstm = nn.LSTM(100, n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        ## define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        ## define the fully-connected layer
        self.fc = nn.Linear(n_hidden, vocab_size_Q6)      
    
    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''

        ## pass input through embedding layer
        embedded = self.emb_layer(x)     
        
        ## Get the outputs and the new hidden state from the lstm
        lstm_output, hidden = self.lstm(embedded, hidden)
        
        ## pass through a dropout layer
        out = self.dropout(lstm_output)
        
        #out = out.contiguous().view(-1, self.n_hidden) 
        out = out.reshape(-1, self.n_hidden) 

        ## put "out" through the fully-connected layer
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        # if GPU is available
        if (torch.cuda.is_available()):
          hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                    weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        
        # if GPU is not available
        else:
          hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                    weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden  
    
    
    









# Q2.8 ###########================-------------------


with open('tweet.txt') as f:
    tweet = [line.rstrip() for line in f]


# sentence segmenting + lower casing + Tokenization + Padding using function  NLP_PreProcessing_Test
tweet_Preprocessed_Text = []
for t in range(len(tweet)):
    tweet_Preprocessed_Text = tweet_Preprocessed_Text + NLP_PreProcessing_Test((tweet[t])[4:-5]) 

tweet_Preprocessed_Text_Equal_seqs_L5 = sum([Equal_seq(i,5) for i in tweet_Preprocessed_Text], [])

del t,f

# Create Vocab tweet
tweet_words = Counter() 
for i, sentence in enumerate(tweet_Preprocessed_Text):
    for word in sentence: 
        tweet_words.update([word]) 
tweet_words = {k:v for k,v in tweet_words.items() if v>1} # Removing the words that only appear once
del i,sentence,word
tweet_words = sorted(tweet_words, key=tweet_words.get, reverse=True) # Sorting the words
tweet_words = ['_PAD','_UNK'] + tweet_words
tweet_word2idx = {o:i for i,o in enumerate(tweet_words)}
tweet_idx2word = {i:o for i,o in enumerate(tweet_words)}


# Looking up the mapping dictionary and assigning the index to the respective words
tweet_Preprocessed_Text_Equal_seqs_INDICES_L5 =[]
for i, sentence in enumerate(tweet_Preprocessed_Text_Equal_seqs_L5):
    tweet_Preprocessed_Text_Equal_seqs_INDICES_L5.append([tweet_word2idx[word] if word in tweet_word2idx else tweet_word2idx['_UNK'] for word in sentence])
del i, sentence


X = []
Y = []
for S in tweet_Preprocessed_Text_Equal_seqs_INDICES_L5:
  X.append(S[:-1])
  Y.append(S[1:])

tweet_x_int_L5 = np.array(X)
tweet_y_int_L5 = np.array(Y)

del X,Y,S




# Train Or Load LSTM
vocab_size = len(tweet_word2idx)
Do_want_To_Train = 0
batch_size = 320
epochs=20
lr=0.001


if Do_want_To_Train == 1:
    net8 = WordLSTM() # instantiate the model
    net8.cuda() # push the model to GPU
    train(net8, epochs, batch_size, lr, 1, 256,tweet_x_int_L5,tweet_y_int_L5) # train the model
    torch.save(net8, 'Q2_Part_8_Network.pt')
else:
    net8 = torch.load('Q2_Part_8_Network.pt')
    net8.eval()
    



print(net8)

# Generate text
for i in range(10):
    print('=======================================')
    print("- Examples "+str(i)+" - LSTM LM - TWEETs: ")
    print(sample(net8, size='</s>' , prime="<s>", word2idx_Inp = tweet_word2idx, idx2word_Inp =tweet_idx2word),'\n')


del i,Do_want_To_Train






# Q2.9 ###########================-------------------


with open('test_2.txt') as f:
    test_2 = [line.rstrip() for line in f]

# sentence segmenting + lower casing + Tokenization + Padding using function  NLP_PreProcessing_Test
Test_2_Preprocessed_Pride_Text = []
for t in range(len(test_2)):
    Test_2_Preprocessed_Pride_Text = Test_2_Preprocessed_Pride_Text + NLP_PreProcessing_Test((test_2[t])[4:-5]) 

Test_2_Pride_Text_Equal_seqs_L5 = sum([Equal_seq(i,5) for i in Test_2_Preprocessed_Pride_Text], [])

del t,f
# Looking up the mapping dictionary and assigning the index to the respective words
Test_2_Pride_Text_Equal_seqs_INDICES_L5 =[]
for i, sentence in enumerate(Test_2_Pride_Text_Equal_seqs_L5):
    Test_2_Pride_Text_Equal_seqs_INDICES_L5.append([tweet_word2idx[word] if word in tweet_word2idx else tweet_word2idx['_UNK'] for word in sentence])
del i, sentence


Test_2_X = []
Test_2_Y = []
for S in Test_2_Pride_Text_Equal_seqs_INDICES_L5:
  Test_2_X.append(S[:-1])
  Test_2_Y.append(S[1:])

Test_2_x_int = np.array(Test_2_X)
Test_2_y_int = np.array(Test_2_Y)

del Test_2_X,Test_2_Y,S
# Calculate Perplexity
Testing(net8, batch_size ,Test_2_x_int,Test_2_y_int)     





















