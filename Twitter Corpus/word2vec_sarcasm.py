# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

#import libraries
import pandas as pd
import numpy as np
import json 
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from bs4 import BeautifulSoup
import spacy
import re
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from gensim.models import Word2Vec
import os
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, GRU, Flatten, LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers.core import Dense, Dropout
from keras.initializers import Constant
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

# read dataset
df_train1 = pd.read_json('/content/drive/My Drive/Colab Notebooks/sentiment_analysis/sarcasm/twitter/sarcasm_detection_shared_task_twitter_training.jsonl', lines=True)
df_train2= pd.read_json('/content/drive/My Drive/Colab Notebooks/sentiment_analysis/sarcasm/twitter/sarcasm_detection_shared_task_twitter_testing.jsonl', lines=True)
df_train2=df_train2[['label','response','context']]
train=pd.concat([df_train1, df_train2], axis=0)
train.reset_index(drop=True, inplace=True)
train['response']=train['response']+train['context'].astype(str)
train.label = [1 if i == 'NOT_SARCASM' else 0 for i in train.label]
tweets_train = train['response']
labels_train = train['label']

#text cleaning
token = TreebankWordTokenizer()
#expand negation words 
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
    text=text.replace('user','')
    text=text.replace('[','')
    text=text.replace(']','')
    #remove unwanted white spaces
    word_list = token.tokenize(text)
    text = " ".join(word_list).strip()
    return text
#Apply function on review column
train['response']=train['response'].apply(remove_special_characters)

#tokenize tweets
tkr = RegexpTokenizer('[a-zA-Z@]+')
tweets_split = []
for i, line in enumerate(tweets_train):
    #print(line)
    tweet = str(line).lower().split()
    tweet = tkr.tokenize(str(tweet))
    tweets_split.append(tweet)

#turn words into their root form
ps = PorterStemmer() 
for index,line in enumerate(tweets_split):
    tweets_split[index]=[ps.stem(w) for w in line]

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
from keras.preprocessing.sequence import pad_sequences
response_pad=pad_sequences(sequences,maxlen)
sarcasm=train['label'].values
print(response_pad.shape)
print(sarcasm.shape)

#create embedding matrix
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
indices=np.arange(response_pad.shape[0])
np.random.shuffle(indices)
response_pad=response_pad[indices]
sarcasm=sarcasm[indices]
num_validation_samples=int(split*response_pad.shape[0])
x_train_pad=response_pad[:-num_validation_samples]
y_train=sarcasm[:-num_validation_samples]
x_test_pad=response_pad[-num_validation_samples:]
y_test=sarcasm[-num_validation_samples:]

#create model
es_callback = EarlyStopping(monitor='val_loss', patience=2)
embedding_layer = Embedding(num_words,n_dim,embeddings_initializer=Constant(embedding_matrix),input_length=maxlen,trainable=False)
model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#fit model
history=model.fit(x_train_pad, y_train, epochs=15, verbose=1, batch_size=128,callbacks=[es_callback], validation_split=0.2)

#performance evaluation
score = model.evaluate(x_test_pad, y_test, batch_size=128, verbose=2)
print(score[1])
res = model.predict(x_test_pad)
res=res.round()
k=confusion_matrix(y_test,res)
print(k)
print(classification_report(y_test,res))

print("accuracy: ",(k[0][0]+k[1][1])/(k[0][0]+k[0][1]+k[1][0]+k[1][1]))
precision=(k[0][0])/(k[0][0]+k[1][0])
print("precision: ",precision)
recall=(k[0][0])/(k[0][0]+k[0][1])
print("recall: ",recall) #0sarcasm,1=not_sarcasm
print("f1-score: ",(2*precision*recall)/(precision+recall))
print("False positive rate: ",(k[1][0])/(k[1][0]+k[1][1]))
