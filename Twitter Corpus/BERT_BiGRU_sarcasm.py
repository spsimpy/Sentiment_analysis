# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

pip install -qq transformers

#import libraries
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer
import re
from transformers import BertTokenizer
import torch
from torchtext.legacy import data
import random
import numpy as np
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report

#read dataset
df_train = pd.read_json('/content/drive/MyDrive/Colab Notebooks/twitter/sarcasm_detection_shared_task_twitter_training.jsonl', lines=True)
df_test= pd.read_json('/content/drive/My Drive/Colab Notebooks/twitter/sarcasm_detection_shared_task_twitter_testing.jsonl', lines=True)
df_train['response']=df_train['response']+df_train['context'].astype(str)
df_test['response']=df_test['response']+df_test['context'].astype(str)

#expand nagation terms
token = TreebankWordTokenizer()
negations = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
             "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
             "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
             "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
             "mustn't":"must not"
            }

#text cleaning
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text = text.lower()
    for a, b in negations.items():
        if a in text:
            text = text.replace(a,b)
    text = re.sub("[^a-zA-Z]", " ", text)
    text=re.sub(pattern,'',text)
    text=text.replace('user','')
    text=text.replace('[','')
    text=text.replace(']','')
    #remove unwanted white spaces
    word_list = token.tokenize(text)
    text = " ".join(word_list).strip()
    return text
#Apply function on response column
df_train['response']=df_train['response'].apply(remove_special_characters)
df_test['response']=df_test['response'].apply(remove_special_characters)
df_train.label = [1 if i == 'NOT_SARCASM' else 0 for i in df_train.label]
df_test.label = [1 if i == 'NOT_SARCASM' else 0 for i in df_test.label]
#save cleaned files in json format
df_train.to_json('train_sarc.json', orient='records', lines=True)
df_test.to_json('test_sarc.json', orient='records', lines=True)

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
max_input_length=300

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
fields = {'response': ('text', TEXT), 'label': ('label', LABEL)}
#load dataset
train_data, test_data = data.TabularDataset.splits('/content', 
                                                   train='train_sarc.json',
                                                   test='test_sarc.json',
                                                   format='json',
                                                   fields=fields)

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# split data
train_data, valid_data = train_data.split(random_state = random.seed(SEED))
#build label vocabulary
LABEL.build_vocab(train_data)
print(LABEL.vocab.stoi)

BATCH_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#create iterators
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
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
model = BERT_GRU(bert, HIDDEN_DIM,OUTPUT_DIM,N_LAYERS,BIDIRECTIONAL,DROPOUT)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

#freeze  bert parameters
for name, param in model.named_parameters():                
    if name.startswith('bert'):
        param.requires_grad = False

print(f'The model has {count_parameters(model):,} trainable parameters')

import torch.optim as optim
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

import time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

#train the model
N_EPOCHS = 5
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

#load trained model
model.load_state_dict(torch.load('model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

#performance evaluation
testevaluate(model, test_iterator, criterion)
