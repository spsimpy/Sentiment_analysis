# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

# import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import RegexpTokenizer

import time
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Embedding, Dense, GRU, Flatten, LSTM, Dropout, GlobalMaxPooling1D,Bidirectional
from keras.regularizers import l2, l1
from keras.initializers import Constant
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#read processed dataset
tweetsData = pd.read_csv('/content/drive/My Drive/Colab Notebooks/sentiment_analysis/cleaned_sent.csv')

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

#integer encoding of  each token in tweet
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(tweets_split)
sequences = tokenizer_obj.texts_to_sequences(tweets_split)
word_index=tokenizer_obj.word_index
print('unique tokens',len(word_index))

#count length of tweets
len_count=[]
for i in range(len(tweets_split)):
  len_count.append(len(tweets_split[i]))
print(len(len_count),min(len_count),max(len_count),len_count[0:5])

#plot length distribution of tweets
fig, ax = plt.subplots(figsize =(7, 5)) 
plt.xlabel("tweet_length", fontsize=15) 
plt.ylabel("count", fontsize=15) 
plt.title('Tweet Text Length Distribution', fontsize=15) 
ax.hist(len_count, bins =100)
plt.savefig('len_distribution.eps')

#download glove word vectors
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove*.zip
print('Indexing word vectors.')

#read glove vector file of desired dimension 
embeddings_index = {}
f = open('glove.6B.200d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

#pad tweet sequences to fixed length
maxlen=30
from keras.preprocessing.sequence import pad_sequences
tweet_pad=pad_sequences(sequences,maxlen)
sentiment=tweetsData['Sentiment'].values
print(tweet_pad.shape)
print(sentiment.shape)

#create embedding matrix containg each word and its corresponding embedding
n_dim=200
num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,n_dim))
for word,i in word_index.items():
  if i>num_words:
    continue
  embedding_vector=embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i]=embedding_vector

#split into train and test set
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

es_callback = EarlyStopping(monitor='val_loss', patience=2)
#create and initialize embedding layer
embedding_layer = Embedding(num_words,n_dim,embeddings_initializer=Constant(embedding_matrix),input_length=maxlen,trainable=False)
#create model
model = Sequential()
model.add(embedding_layer)
model.add((GRU(100)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

#fit model
start=time.time()
history=model.fit(x_train_pad, y_train, epochs=15, verbose=1, batch_size=128,validation_split=0.2, callbacks=[es_callback] )
stop=time.time()
print("training_time:",(stop-start)) #training time

#function to plot validation curves
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs",fontsize=14)
  plt.ylabel(string,fontsize=14)
  plt.legend([string, 'val_'+string], fontsize=14)
  plt.savefig(string+'.eps')
  plt.savefig(string+'.png')
  plt.show()
   
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# results
score = model.evaluate(x_test_pad, y_test, batch_size=128, verbose=2)
print(score[1])
res = model.predict(x_test_pad)
res=res.round()
k=confusion_matrix(y_test,res)
print(k)
print(classification_report(y_test,res))
#print("accuracy_score",accuracy_score(y_test,res))
print("accuracy: ",(k[0][0]+k[1][1])/(k[0][0]+k[0][1]+k[1][0]+k[1][1]))
precision=(k[1][1])/(k[1][1]+k[0][1])
print("precision: ",precision)
recall=(k[1][1])/(k[1][1]+k[1][0])
print("recall: ",recall)#0-negative,1=positive
print("f1-score: ",(2*precision*recall)/(precision+recall))
print("False positive rate: ",(k[0][1])/(k[0][1]+k[0][0]))
