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

#stopwords
!pip install stop_words
from nltk.corpus import stopwords
from stop_words import get_stop_words
stop= list(get_stop_words('en'))         #About 900 stopwords
nltk_words = list(stopwords.words('english')) #About 150 stopwords
stop.extend(nltk_words)
df['clean'] = df['clean'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))



#Lemmatize
import nltk

lemmatizer = nltk.stem.WordNetLemmatizer()
df['clean'] = df['clean'].apply(lambda x: ' '.join([lemmatizer.lemmatize(item) for item in x.split()]))

#Stemming
#import pandas as pd
#from nltk.stem.snowball import SnowballStemmer
#stemmer = SnowballStemmer("english")
#df['clean'] = df['clean'].apply(lambda x: ' '.join([stemmer.stem(item) for item in x.split()]))
 # Stem every word.

# Remove punctuation, didn't remove number because they might be important

df['clean'] = df['clean'].map(lambda x: re.sub("[,\.!?~*')()\%\–\-\$\[\]\\/\=\:’]", '', x))
df['clean'] = df['clean'].map(lambda x: re.sub('[\“\”\"_]', '', x))
#Lowercasing
df['clean']= df['clean'].map(lambda x: x.lower())


###WORDCLOUD
# Import the wordcloud library
!pip install wordcloud
from wordcloud import WordCloud
# Join the different processed titles together.
long_string = ','.join(list(df['clean'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()


#We will remove phrases like "self driving" because we expect those
commonwords=['driverless','self driving','selfdriving','car', 'vehicle','autonomous', 'driver']
df['clean'] = df['clean'].apply(lambda x: ' '.join([item for item in x.split() if item not in commonwords]))
#again wordcloud
long_string = ','.join(list(df['clean'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()

##Ten most frequent words
# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
# Helper function
def plot_20_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:20]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(df['clean'])
# Visualise the 10 most common words
plot_20_most_common_words(count_data, count_vectorizer)


#LDA
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
 
# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Tweak the two parameters below
number_topics = 10
number_words = 10
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)

##Interpretting

!pip install pyLDAvis
from pyLDAvis import sklearn as sklearn_lda
import pickle 
import pyLDAvis
import os
%%time
LDAvis_data_filepath = os.path.join('./ldavis_prepared_'+str(number_topics))
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1==1:
 LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
 with open(LDAvis_data_filepath, 'w') as f:
        pickle.dump(LDAvis_prepared, f)
        
 # load the pre-prepared pyLDAvis data from disk
 with open(LDAvis_data_filepath) as f:
    LDAvis_prepared = pickle.load(f)
 pyLDAvis.save_html(LDAvis_prepared, './ldavis_prepared_'+ str(number_topics) +'.html')

print(df.loc[5971,'body'])

isitbot2= df[df['author'].str.contains("bot",na=False)] 
isitbot2= df[df['body'].str.contains("Bot",na=False)] 


df['stpw']=df.clean.map(extract_ngram_freqs)

print(df.loc[5971,'body'])

isitbot2= df[df['author'].str.contains("bot",na=False)] 
isitbot2= df[df['body'].str.contains("Bot",na=False)] 