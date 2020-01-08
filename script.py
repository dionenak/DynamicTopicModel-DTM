# -*- coding: utf-8 -*-
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

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
 date_iso=datetime.utcfromtimestamp(df.loc[i,'created_utc']).isoformat()
 list_dateiso.append(date_iso)

df['created_iso']=list_dateiso
df.sort_values(by='created_iso', inplace=True)


#DUPLICATES
duplicateRowsDF = df[df.duplicated(['body'])]
#some authors are bots, let's check
isitbot=df[df.author.isin(['autotldr'])]
df = df[df.author != 'autotldr']
isitbot=df[df.author.isin(['mvea','ClickableLinkBot','comment_preview_bot','TweetMirrorBot','Decronym', 'TotesMessenger','Automoderator','LocationBot','Mentioned_Videos','DoMoreWithLess','BernardJOrtcutt','ImagesOfNetwork','SteamKiwi','TrendingCommenterBot','shortscience_dot_org','WorldNewsMods'])]
isitbot=df[df.author.isin(['_youtubot_','rtbot2','tweettranscriberbot','transcribot','topredditbot'])]
df = df[df.author != 'alternate-source-bot']
df = df[df.author != 'ClickableLinkBot']
df = df[df.author != 'mvea']
df = df[df.author != 'Decronym']
df = df[df.author != 'TotesMessenger']
df = df[df.author != 'AutoModerator']
df = df[df.author != 'LocationBot']
df = df[df.author != 'Mentioned_Videos']
df = df[df.author != 'DoMoreWithLess']
df = df[df.author != 'ImagesOfNetwork']
df = df[df.author != 'shortscience_dot_org']
df = df[df.author != 'WorldNewsMods']
df = df[df.author != 'TrendingCommenterBot']
df = df[df.author != 'comment_preview_bot']
df = df[df.author != 'TweetMirrorBot']
df = df[df.author != '_youtubot_']
df = df[df.author != 'rtbot2']
df = df[df.author != 'tweettranscriberbot']
df = df[df.author != 'transcribot']
df = df[df.author != 'topredditbot']

# we just droping the rest of the duplicates
df=df.drop_duplicates(subset=['body'], keep='last')
df = df.reset_index(drop=True)
df=df.drop(df.index[499])
#also remove moderators messages.
df=df[df.distinguished!='moderator']

#FOR FREQUENCY
df['created_iso'].value_counts()
df['length'] = df.body.map(len)
!pip install plotly=='4.0.0'
!pip install --upgrade plotly
import plotly.express as px
from plotly.offline import plot
# plot some random 50 tweets, including their text in the hover field
la=px.scatter(df,x='created_iso',y='length',hover_data=['body'])
plot(la)
fig = px.line(df, x='created_iso', y='count')
plot(fig)    
#let's see links
import re

# uses a regex to detect links and replaces them by LINK  
def replace_link(in_string):
    return re.sub(r'http\S+', '',in_string)

import re

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext
# map the text to itself, applying each of the functions
df['clean']=df.body.map(replace_link)
df.clean=df.clean.map(cleanhtml)
#stopwords

from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
stopwords = set(stopwords.words('english'))

def extract_ngram_freqs(token_list):
  listready=word_tokenize(token_list)
  grams = list(ngrams(listready, 1))
  cleaned_grams = []
  for word_tuple in grams:
      for word in word_tuple:
          if word not in stopwords:
              
              cleaned_grams.append(word_tuple)
              
  
  return cleaned_grams


df['stpw']=df.clean.map(extract_ngram_freqs)

print(df.loc[5971,'body'])

isitbot2= df[df['author'].str.contains("bot",na=False)] 
isitbot2= df[df['body'].str.contains("Bot",na=False)] 