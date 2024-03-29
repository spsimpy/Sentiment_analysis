# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

pip install -qq transformers

#import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec

from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, GRU, Flatten, LSTM, Dropout, GlobalMaxPooling1D,Bidirectional
from keras.regularizers import l1, l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from nltk.tokenize import TreebankWordTokenizer
import re
from transformers import BertTokenizer
import torch
from torchtext.legacy import data
import random
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import time

#read dataset
df_train = pd.read_json('/content/drive/MyDrive/Colab Notebooks/twitter/sarcasm_detection_shared_task_twitter_training.jsonl', lines=True)
df_test= pd.read_json('/content/drive/My Drive/Colab Notebooks/twitter/sarcasm_detection_shared_task_twitter_testing.jsonl', lines=True)
df_train['response']=df_train['response']+df_train['context'].astype(str)
df_test['response']=df_test['response']+df_test['context'].astype(str)

#expand negation terms
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
train=df_train
test=df_test[['label','response','context']]
train=pd.concat([train, test], axis=0)
train.reset_index(drop=True, inplace=True)

tweets_train = train['response']
labels_train = train['label']
#tokenize tweets
tkr = RegexpTokenizer('[a-zA-Z@]+')
tweets_split = []
for i, line in enumerate(tweets_train):
    tweet = str(line).lower().split()
    tweet = tkr.tokenize(str(tweet))
    tweets_split.append(tweet)

# set word vector dimension and train word2vec
n_dim = 200
model = Word2Vec(sentences=tweets_split,size=n_dim, min_count=10, sg=1)
words=list(model.wv.vocab)
model.wv.save_word2vec_format('word2.txt',binary=False)

#create a dictionary of word vectors for each word
embedding_index={}
f=open(os.path.join('','word2.txt'),encoding="utf-8")
for line in f:
  values=line.split()
  word=values[0]
  coefs=np.asarray(values[1:])
  embedding_index[word]=coefs
f.close()

#integer encoding of  each token in tweet
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(tweets_split)
sequences = tokenizer_obj.texts_to_sequences(tweets_split)
word_index=tokenizer_obj.word_index
print('unique tokens',len(word_index))

#pad integer encoded sequences
maxlen=300
response_pad=pad_sequences(sequences,maxlen)
sarcasm=train['label'].values
print(response_pad.shape)
print(sarcasm.shape)

#create embedding matrix gor Word2Vec
num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,n_dim))
for word,i in word_index.items():
  if i>num_words:
    continue
  embedding_vector=embedding_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i]=embedding_vector

#split dataset
split=0.5
indices=np.arange(response_pad.shape[0])
np.random.shuffle(indices)
response_pad=response_pad[indices]
sarcasm=sarcasm[indices]
num_validation_samples=int(split*response_pad.shape[0])
x_train_pad=response_pad[:-num_validation_samples]
y_train=sarcasm[:-num_validation_samples]
x_test_pad=response_pad[-num_validation_samples:]
y_test=sarcasm[-num_validation_samples:]
data_bert = train.iloc[indices, :]
train_bert=data_bert[0:-num_validation_samples]
test_bert=data_bert[-num_validation_samples:]
train_bert.to_csv('train_bert.csv') 
test_bert.to_csv('test_bert.csv')

#create Word2Vec embedding based model
es_callback = EarlyStopping(monitor='val_loss', patience=2)
embedding_layer = Embedding(num_words,n_dim,embeddings_initializer=Constant(embedding_matrix),input_length=maxlen,trainable=False)
model1 = Sequential()
model1.add(embedding_layer)
model1.add((LSTM(100)))
#model1.add(Bidirectional(LSTM(100)))
#model1.add(Bidirectional(GRU(100)))
model1.add(Dropout(0.2))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model1.summary())

#download glove word vectors
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove*.zip
print('Indexing word vectors.')

#read glove vector file of desired dimension 
embeddings_index2 = {}
f2= open('glove.6B.200d.txt', encoding='utf-8')
for line in f2:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index2[word] = coefs
f2.close()

#create embedding matrix for glove embeddings
n_dim=200
num_words=len(word_index)+1
embedding_matrix2=np.zeros((num_words,n_dim))
for word,i in word_index.items():
  if i>num_words:
    continue
  embedding_vector2=embeddings_index2.get(word)
  if embedding_vector2 is not None:
    embedding_matrix2[i]=embedding_vector2

#create glove embedding based model 
embedding_layer = Embedding(num_words,n_dim,embeddings_initializer=Constant(embedding_matrix2),input_length=maxlen,trainable=False)
model2 = Sequential()
model2.add(embedding_layer)
#model2.add(Bidirectional(LSTM(100)))
model2.add((GRU(100)))
#model2.add(Bidirectional(GRU(100)))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model2.summary())

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
                                                   train='train_bert.csv',
                                                   test='test_bert.csv',
                                                   format='csv',
                                                   fields=fields)


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
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

#bert model
bert = BertModel.from_pretrained('bert-base-uncased')
class BertGRU(nn.Module):
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
        #print(hidden)
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)
        
        #output = [batch size, out dim]
        
        return output

class BertLSTM(nn.Module):
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
        out, (hidden, cell) = self.lstm(embedded)
        
        if self.lstm.bidirectional:
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

model = BertGRU(bert,
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

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

#train base classifiers
trainstart=time.time()
classifiers = {"bert":model,"word2vec": model1,
               "glove": model2
               }
for key in classifiers:
    # Get classifier
    classifier = classifiers[key]
    # Fit classifier
    if(key=="bert"):
      N_EPOCHS = 6
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
      classifiers[key]=model.load_state_dict(torch.load('model.pt'))
    else:
      classifier.fit(x_train_pad, y_train, epochs=10, verbose=1, batch_size=128,validation_split=0.1, callbacks=[es_callback] )
      # Save fitted classifier
      classifiers[key] = classifier

trainend=time.time()
trainingtime = trainend - trainstart
print('training time',trainingtime)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def predict_sentiment(model, tokenizer, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

results = pd.DataFrame()
for key in classifiers: 
    # Make prediction on test set
    if key=='bert':
      df_test=pd.read_csv('/content/test_bert.csv')
      testx=df_test.response.values
      testy=df_test.label.values
      preds=[predict_sentiment(model,tokenizer,text) for text in testx]
      results[f"{key}"]=preds
    else:
      y_pred = classifiers[key].predict(x_test_pad)
      # Save results in pandas dataframe object
      results[f"{key}"]=y_pred.reshape(y_pred.shape[0])

# Add the test set to the results object
results["Target"] = y_test

results['target']=df_test['label']

x=results[['bert','word2vec', 'glove']] 
y=results['target']
samp_len=int(0.7*len(results))
trainx_ens=x[:samp_len].values
trainy_ens=y[:samp_len].values
testx_ens=x[samp_len:].values
testy_ens=y[samp_len:].values
trainx_ens = trainx_ens.reshape(-1, 1, 3)
testx_ens=testx_ens.reshape(-1, 1, 3)
trainy_ens = trainy_ens.reshape(-1, 1, 1)
testy_ens=testy_ens.reshape(-1, 1, 1)

#create meta classifier
model_meta = Sequential()
model_meta.add((LSTM(4, input_shape=(1,3)))) #LSTM
#model_meta.add(Bidirectional(LSTM(4),input_shape=(1,3))) #BiLSTM
#model_meta.add(Bidirectional(GRU(4 ),input_shape=(1,3)))
model_meta.add(Dense(1, activation='sigmoid'))
model_meta.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model_meta.summary())

history=model_meta.fit(trainx_ens, trainy_ens, epochs=30, verbose=0,validation_split=0.2,batch_size=32, callbacks=[es_callback] )

#performance evaluation of ensemble framework
predy_ens=model_meta.predict(testx_ens)
y_pred=[1 if x>0.5 else 0 for x in predy_ens]
y_test=testy_ens.reshape(len(testy_ens),)
print(classification_report(y_test,y_pred,digits=4))
cm=confusion_matrix(y_test,y_pred)
acc=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
prc=cm[0][0]/(cm[0][0]+cm[1][0])
rcl=cm[0][0]/(cm[0][0]+cm[0][1])
score=(2*prc*rcl)/(prc+rcl)
fpr=cm[1][0]/(cm[1][0]+cm[1][1])
print(acc,prc,rcl,score,fpr)
