# -*- coding: utf-8 -*-


from google.colab import drive
drive.mount('/content/drive')

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

es_callback = EarlyStopping(monitor='val_loss', patience=2)
# create embedding layer
embedding_layer = Embedding(num_words,n_dim,embeddings_initializer=Constant(embedding_matrix),input_length=30,trainable=False)
#create model
model = Sequential()
model.add(embedding_layer)
model.add(GRU(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
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
#print("accuracy_score",accuracy_score(y_test,res))
print("accuracy: ",(k[0][0]+k[1][1])/(k[0][0]+k[0][1]+k[1][0]+k[1][1]))
precision=(k[1][1])/(k[1][1]+k[0][1])
print("precision: ",precision)
recall=(k[1][1])/(k[1][1]+k[1][0])
print("recall: ",recall)#0-negative,1=positive
print("f1-score: ",(2*precision*recall)/(precision+recall))
print("False positive rate: ",(k[0][1])/(k[0][1]+k[0][0]))
