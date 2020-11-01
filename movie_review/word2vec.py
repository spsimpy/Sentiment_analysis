from google.colab import drive
drive.mount('/content/drive')

#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import RegexpTokenizer

from gensim.models import Word2Vec
import os
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from keras.layers import Dense, GRU, Flatten, LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers.core import Dense, Dropout
from keras.initializers import Constant
from keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


#read the dataset
imdb_data=pd.read_csv('/content/drive/My Drive/Colab Notebooks/sentiment_analysis/movie_review/IMDB Dataset.csv')

#removing html tags
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
imdb_data.head()

#download stopwords
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

#convert nominal labels to binary form
imdb_data.sentiment = [1 if i == 'positive' else 0 for i in imdb_data.sentiment]
#imdb_data.head()

#read reviews and labels seperately
reviews = imdb_data['review']
labels = imdb_data['sentiment']

#tokenization of reviews
from nltk.tokenize import RegexpTokenizer
tkr = RegexpTokenizer('[a-zA-Z@]+')
reviews_split = []

for i, line in enumerate(reviews):
    #print(line)
    review = str(line).lower().split()
    review = tkr.tokenize(str(review))
    reviews_split.append(review)

print(reviews_split[1])

#train word2vec model
n_dim = 200
model = Word2Vec(sentences=reviews_split,size=n_dim, min_count=10)
words=list(model.wv.vocab)
model.wv.most_similar('good')

#save trained word2vec model in text format
model.wv.save_word2vec_format('cleaned_sent_word2vec.txt',binary=False)

# read vector file and store in a dictionary
embedding_index={}
f=open(os.path.join('','cleaned_sent_word2vec.txt'),encoding="utf-8")
for line in f:
  values=line.split()
  word=values[0]
  coefs=np.asarray(values[1:])
  embedding_index[word]=coefs
f.close()


#convert tokens to corresponding integer index
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(reviews_split)
sequences = tokenizer_obj.texts_to_sequences(reviews_split)
word_index=tokenizer_obj.word_index
print('unique tokens',len(word_index))

#padding reviews to fixed length
review_pad=pad_sequences(sequences,maxlen=700)
sentiment=imdb_data['sentiment'].values
print(review_pad.shape)
print(sentiment.shape)

#create an embedding matrix containing embedding for each unique token 
num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,n_dim))
for word,i in word_index.items():
  if i>num_words:
    continue
  embedding_vector=embedding_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i]=embedding_vector


es_callback = EarlyStopping(monitor='accuracy', patience=2)
#embedding layer
embedding_layer = Embedding(num_words,n_dim,embeddings_initializer=Constant(embedding_matrix),input_length=700,trainable=False)
#create model
model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(GRU(100)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

#division into training and testing sets
split=0.2
indices=np.arange(review_pad.shape[0])
np.random.shuffle(indices)
review_pad=review_pad[indices]
sentiment=sentiment[indices]
num_validation_samples=int(split*review_pad.shape[0])
x_train_pad=review_pad[:-num_validation_samples]
y_train=sentiment[:-num_validation_samples]
x_test_pad=review_pad[-num_validation_samples:]
y_test=sentiment[-num_validation_samples:]

#fit model
model.fit(x_train_pad, y_train, epochs=10, verbose=1, batch_size=128,validation_split=0.1,callbacks=[es_callback])

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

