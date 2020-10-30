from google.colab import drive
drive.mount('/content/drive')

# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from nltk.tokenize.toktok import ToktokTokenizer
from keras.preprocessing.text import Tokenizer

from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping

#read dataset
imdb_data=pd.read_csv('/content/drive/My Drive/Colab Notebooks/sentiment_analysis/movie_review/IMDB Dataset.csv')
print(imdb_data.shape)
imdb_data.head(10)

sns.countplot(imdb_data.sentiment)

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

#removing the stopwords
nltk.download('stopwords')
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')
stop=set(stopwords.words('english'))
print(stop)
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

#find length distribution of reviews
len_count=[]
for i in range(len(reviews_split)):
  len_count.append(len(reviews_split[i]))
print('min_len:',min(len_count),'max_len:',max(len_count),len_count[0:5])
#plt.style.use('ggplot')
fig, ax = plt.subplots(figsize =(10,5)) 
plt.xlabel("Review_length") 
plt.ylabel("count") 
plt.title('Review Text Length Distribution') 
ax.hist(len_count, bins=list(range(0,400,4)),ec='white')

#convert tokens to corresponding integer index
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(reviews_split)
sequences = tokenizer_obj.texts_to_sequences(reviews_split)
word_index=tokenizer_obj.word_index
print('unique tokens',len(word_index))

#download pretrained glove vectors
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove*.zip
print('Indexing word vectors.')

# read desired word vector file and store in a dictionary
embeddings_index = {}
f = open('glove.6B.200d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

#padding reviews to fixed length
maxlen=700
from keras.preprocessing.sequence import pad_sequences
review_pad=pad_sequences(sequences,maxlen)
sentiment=imdb_data['sentiment'].values
print(review_pad.shape)
print(sentiment.shape)

#create an embedding matrix containing embedding for each unique token 
n_dim=200
num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,n_dim))
for word,i in word_index.items():
  if i>num_words:
    continue
  embedding_vector=embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i]=embedding_vector

#split into training and testing set
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

es_callback = EarlyStopping(monitor='val_loss', patience=2)
max_len=700
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
##embedding layer using embedding matrix
embedding_layer = Embedding(num_words,
                           n_dim,
                           weights = [embedding_matrix],
                           input_length = max_len,
                           trainable=False,
                           name = 'embeddings')
embedded_sequences = embedding_layer(sequence_input)
x = (GRU(100, return_sequences=True,name='lstm_layer'))(embedded_sequences)
x = GlobalMaxPooling1D()(x)
x = Dropout(0.2)(x)
preds = Dense(1, activation="sigmoid")(x)
model = Model(sequence_input, preds)
model.compile(loss = 'binary_crossentropy',
             optimizer='adam',
             metrics = ['accuracy'])
model.summary()

#fit the model
history=model.fit(x_train_pad, y_train, verbose=0, epochs=15, batch_size=128,callbacks=[es_callback], validation_split=0.2)

#test the model
score = model.evaluate(x_test_pad, y_test, batch_size=128, verbose=2)
print(score[1])
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
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

#accuracy and loss curves
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend(['training_'+string, 'validation_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

