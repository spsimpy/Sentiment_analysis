# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

pip install -qq transformers

#import libraries
import os
import numpy as np
import pandas as pd
import time
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
import torch
from torchtext.legacy import data
import random
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim

#read processed dataset
tweetsData = pd.read_csv('/content/drive/My Drive/Colab Notebooks/cleaned_sent.csv')
tweetsData = tweetsData.sample(frac=1).reset_index(drop=True)
tweetsData.dropna(subset = ["SentimentText"], inplace=True)
tweetsData=tweetsData[0:800000]

# tokenization of tweets
tweets = tweetsData['SentimentText']
tkr = RegexpTokenizer('[a-zA-Z@]+')
tweets_split = []
for i, line in enumerate(tweets):
    #print(line)
    tweet = str(line).lower().split()
    tweet = tkr.tokenize(str(tweet))
    tweets_split.append(tweet)
print(tweets_split[1])

# set word vector dimension and train word2vec
n_dim = 200
model = Word2Vec(sentences=tweets_split,size=n_dim, min_count=10)
words=list(model.wv.vocab)
#save trained word2vec
model.wv.save_word2vec_format('cleaned_sent_word2vec.txt',binary=False)

#create a dictionary of word vectors for each word
embedding_index={}
f=open(os.path.join('','cleaned_sent_word2vec.txt'),encoding="utf-8")
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
tweet_pad=pad_sequences(sequences,maxlen=30)

sentiment=tweetsData['Sentiment'].values
print(tweet_pad.shape)
print(sentiment.shape)

#create an  embedding matrix
num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,n_dim))
for word,i in word_index.items():
  if i>num_words:
    continue
  embedding_vector=embedding_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i]=embedding_vector

#split data into train and test
split=0.2
indices=np.arange(tweet_pad.shape[0])
np.random.shuffle(indices)
tweet_pad=tweet_pad[indices]
sentiment=sentiment[indices]
num_validation_samples=int(split*tweet_pad.shape[0])
x_train_pad=tweet_pad[:-num_validation_samples]
y_train=sentiment[:-num_validation_samples]
x_test_pad=tweet_pad[-num_validation_samples:]
y_test=sentiment[-num_validation_samples:]
data_bert = tweetsData.iloc[indices, :]
train_bert=data_bert[0:-num_validation_samples]
test_bert=data_bert[-num_validation_samples:]
train_bert.to_csv('train_bert.csv') 
test_bert.to_csv('test_bert.csv')

#model1(word2vec-lstm)
es_callback = EarlyStopping(monitor='val_loss', patience=2)
# create embedding layer
embedding_layer = Embedding(num_words,n_dim,embeddings_initializer=Constant(embedding_matrix),input_length=30,trainable=False)
#create model
model1= Sequential()
model1.add(embedding_layer)
model1.add(LSTM(100))
#model1.add(Bidirectional(GRU(100)))
model1.add(Dropout(0.2))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
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

n_dim=200
num_words=len(word_index)+1
embedding_matrix2=np.zeros((num_words,n_dim))
for word,i in word_index.items():
  if i>num_words:
    continue
  embedding_vector2=embeddings_index2.get(word)
  if embedding_vector2 is not None:
    embedding_matrix2[i]=embedding_vector2

#model2(glove-gru)
embedding_layer = Embedding(num_words,n_dim,embeddings_initializer=Constant(embedding_matrix2),input_length=30,trainable=False)
#create model
model2 = Sequential()
model2.add(embedding_layer)
model2.add((GRU(100)))
#model2.add(Bidirectional(GRU(100)))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model2.summary())

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
fields = {'SentimentText':('text',TEXT), 'Sentiment':('label',LABEL)}
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
train_data, val_data = train_data.split(split_ratio=0.7)#random_state = random.seed(SEED))
LABEL.build_vocab(train_data)
print(LABEL.vocab.itos)

BATCH_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, val_data, test_data), 
    batch_size = BATCH_SIZE, 
    sort_within_batch = True,
    sort_key = lambda x: len(x.text),
    device = device)

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
model = BertGRU(bert,HIDDEN_DIM,OUTPUT_DIM,N_LAYERS,BIDIRECTIONAL,DROPOUT)

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

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

#train base classifiers
classifiers = {"bert":model,"word2vec": model1,
               "glove": model2
               }
trainstart=time.time()
for key in classifiers:
    # Get classifier
    classifier = classifiers[key]
    # Fit classifier
    if(key=="bert"):
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
      classifiers[key]=model.load_state_dict(torch.load('model.pt'))
    else:
      classifier.fit(x_train_pad, y_train, epochs=10, verbose=1, batch_size=128,validation_split=0.2, callbacks=[es_callback] )
      # Save fitted classifier
      classifiers[key] = classifier

trainend=time.time()
traintime=trainend-trainstart
print('training time',traintime)

#function to get bert predictions
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

#create dataframe containing predictions from each base classifier
results = pd.DataFrame()
for key in classifiers:
    # Make prediction on test set
    if key=='bert':
      df_test=pd.read_csv('/content/test_bert.csv')
      testx=df_test.SentimentText.values
      testy=df_test.Sentiment.values
      preds=[predict_sentiment(model,tokenizer,text) for text in testx]
      results[f"{key}"]=preds
    else:
      y_pred = classifiers[key].predict(x_test_pad)
      # Save results in pandas dataframe object
      results[f"{key}"]=y_pred.reshape(y_pred.shape[0])

# Add the test set to the results object
results["Target"] = y_test

results['target']=df_test['Sentiment']

x=results[['bert','word2vec', 'glove']] 
y=results['target']
samp_len=int(0.8*len(results))
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
model_meta.add((LSTM(1, input_shape=(1,3))))
#model_meta.add(Bidirectional(GRU(4), input_shape=(1,3)))
#model_meta.add(Bidirectional(LSTM(4), input_shape=(1,3)))
model_meta.add(Dense(1, activation='sigmoid'))
model_meta.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model_meta.summary())

#train meta classifier
history=model_meta.fit(trainx_ens, trainy_ens, epochs=30, verbose=1,validation_split=0.3, batch_size=128, callbacks=[es_callback])

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs",fontsize=14)
  plt.ylabel(string,fontsize=14)
  plt.legend(['training '+string, 'validation '+string], fontsize=14)
  plt.savefig(string+'.eps')
  plt.savefig(string+'.png')
  plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

#performance evalustion
predy_ens=model_meta.predict(testx_ens)
y_pred=[1 if x>0.5 else 0 for x in predy_ens]
y_test=testy_ens.reshape(len(testy_ens),)
cm=confusion_matrix(y_test,y_pred)
print(cm)
acc=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
prc=cm[1][1]/(cm[1][1]+cm[0][1])
rcl=cm[1][1]/(cm[1][1]+cm[1][0])
score=(2*prc*rcl)/(prc+rcl)
fpr=cm[0][1]/(cm[0][1]+cm[0][0])
print(acc,prc,rcl,score,fpr)
print(classification_report(y_test,y_pred,digits=4))
