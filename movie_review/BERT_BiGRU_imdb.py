# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

pip install -qq transformers

# import libraries
import pandas as pd
import numpy as np
import json 
import seaborn as sns
import matplotlib.pyplot as plt
import time
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string
from nltk.tokenize.toktok import ToktokTokenizer
from transformers import BertTokenizer
import torch
from torchtext.legacy import data
import random
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report

#read dataset
imdb_data=pd.read_csv('/content/drive/My Drive/Colab Notebooks/IMDB Dataset.csv')
print(imdb_data.shape)

#data cleaning
#Removing html tags
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(denoise_text)
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(remove_special_characters)
nltk.download('stopwords')
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')
stop=set(stopwords.words('english'))
print(stop)

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
    
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(remove_stopwords)

#converting nominal labels to binary form
imdb_data.sentiment = [1 if i == 'positive' else 0 for i in imdb_data.sentiment]
imdb_data.head()
imdb_data.to_csv('data.csv')

#function to tokenize and pad
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id
max_input_length=500

#define fields
TEXT = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

LABEL = data.LabelField(dtype = torch.float)

fields = {'review':('text',TEXT), 'sentiment':('label',LABEL)}
dataset = data.TabularDataset('/content/data.csv',format='csv', fields=fields)

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
train_data, test_data = dataset.split(split_ratio=0.8,random_state = random.seed(SEED))
train_data, val_data = train_data.split(split_ratio=0.7,random_state = random.seed(SEED))

LABEL.build_vocab(train_data)
print(LABEL.vocab.stoi)

BATCH_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, val_data, test_data), 
    batch_size = BATCH_SIZE, 
    sort_within_batch = True,
    sort_key = lambda x: len(x.text),
    device = device)

#load bert
bert = BertModel.from_pretrained('bert-base-uncased')

#create model
class BERT_GRU(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(embedding_dim,
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
        _, hidden = self.rnn(embedded)
        #hidden = [n layers * n directions, batch size, emb dim]
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        output = self.out(hidden)
        #output = [batch size, out dim]
        return output

HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERT_GRU(bert,HIDDEN_DIM,OUTPUT_DIM,N_LAYERS,BIDIRECTIONAL,DROPOUT)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

for name, param in model.named_parameters():                
    if name.startswith('bert'):
        param.requires_grad = False

print(f'The model has {count_parameters(model):,} trainable parameters')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    for batch in iterator: 
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
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
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def testevaluate(model, iterator, criterion):
    preds=[]
    label=[]
    model.eval()
    with torch.no_grad():
    
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            rounded_preds = torch.round(torch.sigmoid(predictions))
            rounded_preds=rounded_preds.tolist()
            lab_list=batch.label.tolist()
            preds=preds+rounded_preds
            label=label+lab_list
        cm=confusion_matrix(label,preds)
        print(cm)
        print(classification_report(label,preds,digits=4))
        acc=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
        prc=cm[1][1]/(cm[1][1]+cm[0][1])
        rcl=cm[1][1]/(cm[1][1]+cm[1][0])
        score=(2*prc*rcl)/(prc+rcl)
        fpr=cm[0][1]/(cm[0][1]+cm[0][0])
        print('accuracy:',acc,'Precision:',prc,'recall:',rcl,'f1-score:',score,'fpr:',fpr)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

#train bert-bigru
N_EPOCHS = 7
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

#load and evaluate performance
model.load_state_dict(torch.load('model.pt'))
testevaluate(model, test_iterator, criterion)
