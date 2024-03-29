# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

pip install -qq transformers

#import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import TreebankWordTokenizer
import re
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import RegexpTokenizer
from transformers import BertTokenizer
import torch
from torchtext.legacy import data
import random
from transformers import BertModel
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/smile-annotations-final.csv', names=['id', 'text', 'category'])
#preprocess dataset
df.set_index('id', inplace=True)
df = df[~df.category.str.contains('\|')]
df = df[df.category != 'nocode']
possible_labels = df.category.unique()
label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
df['label'] = df.category.replace(label_dict)
token = TreebankWordTokenizer()
negations = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
             "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
             "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
             "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
             "mustn't":"must not"
            }
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text = text.lower()
    for a, b in negations.items():
        if a in text:
            text = text.replace(a,b)
    text = re.sub("[^a-zA-Z]", " ", text)
    text=re.sub(pattern,'',text)
    #remove unwanted white spaces
    word_list = token.tokenize(text)
    text = " ".join(word_list).strip()
    return text
#Apply function on text column
df['text']=df['text'].apply(remove_special_characters)
df.to_csv('data.csv')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id
max_input_length=30

TEXT = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

LABEL = data.LabelField(dtype = torch.float)

fields = {'text':('text',TEXT), 'label':('label',LABEL)}
dataset = data.TabularDataset('/content/data.csv',format='csv', fields=fields)

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
train_data, test_data = dataset.split(split_ratio=0.5,random_state = random.seed(SEED))
train_data, val_data = train_data.split(split_ratio=0.7,random_state = random.seed(SEED))

LABEL.build_vocab(train_data)
print(LABEL.vocab.stoi)

BATCH_SIZE =128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, val_data, test_data), 
    batch_size = BATCH_SIZE, 
    sort_within_batch = True,
    sort_key = lambda x: len(x.text),
    device = device)

bert = BertModel.from_pretrained('bert-base-uncased')
class LSTM(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super(LSTM, self).__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.embedding_dim=embedding_dim
        self.lstm = nn.LSTM(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
                
        with torch.no_grad():
            embedded = self.bert(text)[0]
                
        #embedded = [batch size, sent len, emb dim]
        
        #_, hidden = self.lstm(embedded)
        out, (hidden, cell) = self.lstm(embedded)
        #print(hidden)
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)
        #output = [batch size, out dim]
        
        return output

HIDDEN_DIM = 256
OUTPUT_DIM = 6 
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = LSTM(bert,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

for name, param in model.named_parameters():                
    if name.startswith('bert'):
        param.requires_grad = False

print(f'The model has {count_parameters(model):,} trainable parameters')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.Adam(model.parameters())
criterion =nn.CrossEntropyLoss() 
model = model.to(device)
criterion = criterion.to(device)

def accuracy(preds, y):
    preds = torch.max(preds, 1)[1]
    correct=0
    total=0
    correct += (preds == y).float().sum()
    total += y.shape[0]
    acc = correct / total
    return acc
  
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        #batch.label = torch.LongTensor(batch.label).to(device)
        loss = criterion(predictions, batch.label.long())
        
        acc = accuracy(predictions, batch.label.long())
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label.long())
            
            acc = accuracy(predictions, batch.label.long())

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 9
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        
    end_time = time.time()
        
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        model_name='model.pt'
        torch.save(model.state_dict(), model_name)
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

model.load_state_dict(torch.load('model.pt'))
def testevaluate(model, iterator, criterion):
    preds=[]
    label=[]
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            predictions = torch.max(predictions, 1)[1]
            lab_list=batch.label.tolist()
            label=label+lab_list
            preds=preds+predictions.tolist()
            
        print('Precision: ', precision_score(label,preds, average='weighted'))
        print('Recall: ', recall_score(label,preds, average='weighted'))
        print('F1 score: ', f1_score(label, preds, average='weighted'))

testevaluate(model, test_iterator, criterion)
