#!/usr/bin/env python
# coding: utf-8




#import libraries
import pandas as pd
import re 
import html
from nltk.tokenize import TreebankWordTokenizer

#read unprocessed dataset
data=pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1')
data.head()

#function to select desired columns and change label to binary form
def prepare(data):
    data.columns=["Sentiment","ItemID","Date","Blank","SentimentSource","SentimentText"]
    data.drop(['ItemID','Date','Blank', 'SentimentSource'], axis=1, inplace=True)
    data = data[data.Sentiment.isnull() == False]
    data['Sentiment'] = data['Sentiment'].map( {4:1, 0:0})
    data = data[data['SentimentText'].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print('dataset loaded with shape', data.shape)    
    return data

data = prepare(data)
data.head(5)

data.shape

token = TreebankWordTokenizer()
#dictionary to replace contractions with expanded version  
negations = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
             "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
             "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
             "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
             "mustn't":"must not"
            }

#remove user handle mention, urls
mentions = r'@[A-Za-z0-9]+'
url_https = 'https?://[A-Za-z0-9./]+'
url_www = r'www.[^ ]+'
def tweet_cleaning(text):
    text = html.unescape(text)
    text = re.sub(mentions, '', text)
    text = re.sub(url_https, '', text)
    text = re.sub(url_www, '', text)
    text = text.lower()
    for a, b in negations.items():
        if a in text:
            text = text.replace(a,b)
    text = re.sub("[^a-zA-Z]", " ", text)
    
    #remove unwanted white spaces
    word_list = token.tokenize(text)
    text = " ".join(word_list).strip()
    return text

#clean tweets
data['SentimentText'] = data['SentimentText'].apply(lambda x: tweet_cleaning(x))
data.head()

#save cleaned dataset for later use
data.to_csv('cleaned_sent.csv')







