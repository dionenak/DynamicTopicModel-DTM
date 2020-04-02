# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:58:59 2020

@author: dio
"""

# setting working directory
#import os
#path=""
#os.chdir(path)

import nltk; nltk.download('stopwords')
import re
import numpy as np
#from pprint import pprint


import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

# spacy for lemmatization
#!pip install spacy
import spacy

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel

#Import dataset
df = pd.read_excel('march2.xlsx')
#Remove the empty columns
df.drop(df.columns[[20,21,22,23,24,25,26]], axis=1, inplace=True)
#Remove controversiality, downs, ups, edited, retrieved_on, score,name, archived, score_hidden,
#author_flair_css_class, author_flair_text, link_id, parent_id, subreddit_id, id,gilded
#KEEP distinguished, body, created_utc, author,subreddit
df.drop(df.columns[[1,2,3,5,6,8,9,10,11,12,13,14,15,17,19]], axis=1, inplace=True)
print(df.columns)
#Inspection
#Which columns have NAs
df.isna().sum()
#body:0, aythor:51, created_utc:97,subreddit:1, distinguished:17524
#We have to drop rows with NAs in created_utc
df=df.dropna(subset=['created_utc'])
df.isna().sum()
#We have body:0, author:0,created_utc:0, subreddit:1, distinguished:17435
#Some rows dont have dates, so we have to get rid of them
for i,j in df.iterrows():
    if type(df.loc[i,'created_utc'])!= int:
        print(df.loc[i,'created_utc'])
        df=df.drop([i])

#Let's turn dates utc to iso        
from datetime import datetime
list_dateiso=[]
for i,j in df.iterrows():
 #ISO8601
 list_dateiso.append(datetime.utcfromtimestamp(df.loc[i,'created_utc']).isoformat())
df['created_iso']=list_dateiso
#Sort the rows based on their date
df.sort_values(by='created_iso', inplace=True)

#DUPLICATES
duplicateRowsDF = df[df.duplicated(['body'])]
#Some authors are bots, let's check
#After some vision inspection and manual inspection
botnames=["bot","Bot","Bawt"]
isitbot=[]
for bot in botnames:
  isitbot.append( df[df['author'].str.contains(bot,na=False)]) 
  

#Now let's see more closely some of the ones we found
isitbot2=df[df.author.isin(['autotldr','mvea','ClickableLinkBot','comment_preview_bot','TweetMirrorBot',
                           'Decronym', 'TotesMessenger','Automoderator','LocationBot','Mentioned_Videos',
                           'DoMoreWithLess','BernardJOrtcutt','ImagesOfNetwork','SteamKiwi','TrendingCommenterBot',
                           'shortscience_dot_org','WorldNewsMods', '_youtubot_','rtbot2','tweettranscriberbot',
                           'transcribot','topredditbot'])]
#Remove the bots
botlist=['autotldr', 'alternate-source-bot', 'ClickableLinkBot', 'mvea', 'Decronym',
          'TotesMessenger','AutoModerator', 'LocationBot', 'Mentioned_Videos', 'DoMoreWithLess',
          'ImagesOfNetwork', 'shortscience_dot_org', 'WorldNewsMods', 'TrendingCommenterBot',
          'comment_preview_bot','TweetMirrorBot','_youtubot_', 'rtbot2', 'tweettranscriberbot',
          'transcribot', 'topredditbot', "RepostBawt", "WikiTextBot", "SteamKiwi"]
for bot in botlist:
 df = df[df.author != bot ]
 
# We just droping the rest of the duplicates
df=df.drop_duplicates(subset=['body'], keep='last')
df = df.reset_index(drop=True)
df=df.drop(df.index[499]) #one duplicate that our function didn't detect
# Also remove moderators messages.
df=df[df.distinguished!='moderator']

#Find the dates we want
df = df[(df['created_iso'] > '2018-03-12') & (df['created_iso'] < '2018-03-28') ]
df = df[(df['created_iso'] < '2018-03-18') | (df['created_iso'] > '2018-03-20')]
df = df[(df['created_iso'] < '2018-03-22') | (df['created_iso'] > '2018-03-23')]


df = df.reset_index(drop=True)

#How many comments per timeslice
print(len(df.loc[(df['created_iso']< '2018-03-18')])) ##1816
print(len(df.loc[(df['created_iso']> '2018-03-18') & (df['created_iso'] < '2018-03-22')])) ## 2817
print(len(df.loc[(df['created_iso']> '2018-03-22')])) ## 2164


# Uses a regex to detect and remove links and html  
def replace_link(in_string):
    return re.sub(r'http\S+', '',in_string)

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

# map the text to itself, applying each of the functions
df['clean']=df.body.map(replace_link)
df.clean=df.clean.map(cleanhtml)

#stopwords
!pip install stop_words
from nltk.corpus import stopwords
from stop_words import get_stop_words
stop= list(get_stop_words('en'))         #About 900 stopwords
nltk_words = list(stopwords.words('english')) #About 150 stopwords
stop.extend(nltk_words)
df['clean'] = df['clean'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))

#Lowercasing
df['clean']= df['clean'].map(lambda x: x.lower())
df['clean'].replace(regex=True,inplace=True,to_replace=r'self.?driving (car|cars)|autonomous (car|cars)|autonomous.?(vehicle|vehicles)|automated (car|cars)|automated driving|driver.?less (car|cars)|automated vehicle',value=r'selfdrivingcar ')
df['clean'].replace(regex=True,inplace=True,to_replace=r'self driving ',value=r'selfdriving ')
df['clean'].replace(regex=True,inplace=True,to_replace=r'car|cars',value=r'car ')

#Make the comments a list
data = df.clean.values.tolist()
#Tokenize
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))
print(data_words[:1])

##Bigrams are two words frequently occurring together in the document. Trigrams are 3 words frequently occurring.
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
bigram_mod = gensim.models.phrases.Phraser(bigram)



###LET'S DO IT
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


# Form Bigrams
data_words_bigrams = make_bigrams(data_words)
print(data_words_bigrams[11])

# Do lemmatization keeping only noun, vb
##Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp=spacy.load('en_core_web_sm',disable=['parse', 'ner'])

def lemmatization(texts, allowed_postags=['NOUN','VERB']):
    
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'VERB'])

# Create Dictionary and Save it as a pickle file
id2word = corpora.Dictionary(data_lemmatized)
id2word_1 = corpora.Dictionary(data_lemmatized[0:1816])
id2word_2 = corpora.Dictionary(data_lemmatized[1816:4633])
id2word_3 = corpora.Dictionary(data_lemmatized[4633:])



import pickle
f = open("dict.pkl","wb")
pickle.dump(id2word,f)
f.close()
f = open("dict_1.pkl","wb")
pickle.dump(id2word_1,f)
f.close()
f = open("dict_2.pkl","wb")
pickle.dump(id2word_2,f)
f.close()
f = open("dict_3.pkl","wb")
pickle.dump(id2word_3,f)
f.close()
# Create Corpus
texts = data_lemmatized
f1 = open("corp.pkl","wb")
pickle.dump(texts,f1)
f1.close()


