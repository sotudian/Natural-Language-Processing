#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shahab Sotudian
"""


import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import nltk
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
# nltk.download('punkt')

# torch.cuda.empty_cache()
# Functions ###########================-------------------

def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):# shortens sentences or pads sentences with 0 to a fixed length
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features



def Network_Function(X_tr,X_ts,y_tr,y_ts, ES,HS,Want_To_Train_NN,itr):
    
    class SentimentNet_Bidirectional_LSTM(nn.Module):
        def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
            super(SentimentNet_Bidirectional_LSTM, self).__init__()
            self.output_size = output_size
            self.n_layers = n_layers
            self.hidden_dim = hidden_dim
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(hidden_dim*2, output_size)  # 2 for bidirection
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            batch_size = x.size(0)
            h0 = torch.zeros(self.n_layers*2, x.size(0), self.hidden_dim).to(device) # 2 for bidirection 
            c0 = torch.zeros(self.n_layers*2, x.size(0), self.hidden_dim).to(device)
            x = x.long()
            embeds = self.embedding(x)
            lstm_out, _ = self.lstm(embeds, (h0, c0))
            #lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) #shak
            out = self.dropout(lstm_out)
            out = self.fc(out[:, -1, :])
            out = self.sigmoid(out)
    
            out = out.view(batch_size, -1)
            out = out[:,-1]
            return out
        '''
        batch_size = x.size(0)
            x = x.long()
            embeds = self.embedding(x)
            lstm_out, hidden = self.lstm(embeds, hidden)
            lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
            
            out = self.dropout(lstm_out)
            out = self.fc(out)
            out = self.sigmoid(out)
            
            out = out.view(batch_size, -1)
            out = out[:,-1]
            return out, hidden
        '''
        def init_hidden(self, batch_size):
            weight = next(self.parameters()).data
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                          weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
            return hidden
    
    # Parameters
    output_size = 1
    embedding_dim = ES
    hidden_dim = HS
    n_layers = 2
    epochs = 2
    clip = 5
    lr=0.005

     # Inputs
    train_sentences = list(X_tr)
    test_sentences = list(X_ts)
    
    # Converting our labels into numpy arrays
    train_labels = np.array(y_tr)
    test_labels = np.array(y_ts)   

    # Modify URLs to <url>
    for i in range(len(train_sentences)):
        train_sentences[i]  = ' '.join(re.sub("(\w+:\/\/\S+)", "<url>", train_sentences[i]).split())
        
    for i in range(len(test_sentences)):
        test_sentences[i]  = ' '.join(re.sub("(\w+:\/\/\S+)", "<url>", test_sentences[i]).split())

    # Create Vocab
    words = Counter() 
    for i, sentence in enumerate(train_sentences):
        train_sentences[i] = []
        for word in nltk.word_tokenize(sentence): #Tokenizing the words
            words.update([word.lower()]) #Converting all the words to lower case
            train_sentences[i].append(word)

    # Removing the words that only appear once
    words = {k:v for k,v in words.items() if v>1}
    # Sorting the words according to the number of appearances, with the most common word being first
    words = sorted(words, key=words.get, reverse=True)
    # Adding padding and unknown to our vocabulary so that they will be assigned an index
    words = ['_PAD','_UNK'] + words
    # Dictionaries to store the word to index mappings and vice versa
    word2idx = {o:i for i,o in enumerate(words)}
    idx2word = {i:o for i,o in enumerate(words)}

    # Convert to equal Sequ
    train_sentences_Sequ = []
    for i, sentence in enumerate(train_sentences):
        train_sentences_Sequ.append([word2idx[word] if word in word2idx else word2idx['_UNK'] for word in sentence])
    
    test_sentences_Sequ = []
    for i, sentence in enumerate(test_sentences):
        test_sentences_Sequ.append([word2idx[word.lower()] if word.lower() in word2idx else word2idx['_UNK'] for word in nltk.word_tokenize(sentence)])
        
    train_sentences_Sequ = pad_input(train_sentences_Sequ, 200) #200 is the length that the sentences will be padded/shortened to
    test_sentences_Sequ = pad_input(test_sentences_Sequ, 200)
    
    del i,sentence,word

    # Prepare data for network
    batch_size = 51
    train_data = TensorDataset(torch.from_numpy(train_sentences_Sequ), torch.from_numpy(train_labels))
    test_data = TensorDataset(torch.from_numpy(test_sentences_Sequ), torch.from_numpy(test_labels))
    
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    # check GPU 
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()

    # TRAINING
    vocab_size = len(word2idx) + 1

    model = SentimentNet_Bidirectional_LSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    model.to(device)
    #print('*************** Network Architecture *************** ')
    #print(model)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    Name_Model = 'Network_Q3_P6_Embedding_'+str(ES)+'_Hidden_'+str(HS)+"_Fold_"+ str(itr)+'.pt'    # Define name for models
    
    if Want_To_Train_NN == 1:
        model.train()
        for i in range(epochs):
            #h = model.init_hidden(batch_size)    
            for inputs, labels in train_loader:
                #h = tuple([e.data for e in h])
                inputs, labels = inputs.to(device), labels.to(device)
                model.zero_grad()
                #output, h = model(inputs, h)
                output= model(inputs)
                loss = criterion(output.squeeze(), labels.float())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
        torch.save(model.state_dict(), Name_Model)
    else:
        model.load_state_dict(torch.load(Name_Model)) #Loading the model
    

    # TESTING
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
    test_losses = []
    num_correct = 0
    
    model.eval()
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output= model(inputs)
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        pred = torch.round(output.squeeze()) #rounds the output to 0/1
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)
    
    test_losses = np.mean(test_losses)       
    test_acc = num_correct/len(test_loader.dataset)
    return test_losses,test_acc









#  Q3.6   ###########================-------------------



# Data 
Train_Data = pd.read_csv('sentiment-train.csv')


skf = StratifiedKFold(n_splits=5)
Hidden_Size_Values = [128,512]
Embedding_Size_Values = [100,400]
Want_To_Train_NN = 0
List_Acc = []

for HS in Hidden_Size_Values:
    for ES in Embedding_Size_Values:
        print("\nOOOO  Result for Hidden size =  ", HS,'  --  Embedding size = ',ES, '   OOOO=====---------')
        itr =1    
        Total_Acc = 0
        for train_index, test_index in skf.split(Train_Data['text'], Train_Data['sentiment']):
            X_tr, X_ts = Train_Data['text'][train_index], Train_Data['text'][test_index]
            y_tr, y_ts = Train_Data['sentiment'][train_index], Train_Data['sentiment'][test_index]
            # NN Training
            test_losses,test_acc = Network_Function(X_tr,X_ts,y_tr,y_ts, ES,HS,Want_To_Train_NN,itr)
            print("Fold ", itr,' --------- '," Loss: {:.3f}".format(np.mean(test_losses)),"  ---  Accuracy: {:.3f}%".format(test_acc*100))
            Total_Acc = Total_Acc + test_acc
            itr += 1      
        print('Average Accuracy across folds:',(Total_Acc/5),'\n')
        List_Acc.append((Total_Acc/5))


del Total_Acc,skf,itr,X_tr,X_ts,y_tr,y_ts,train_index,test_index,Train_Data,test_acc
del Hidden_Size_Values,Embedding_Size_Values,Want_To_Train_NN, HS,ES,test_losses






















