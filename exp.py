# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""

import nltk; nltk.download('stopwords')
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
!pip install spacy
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

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
df=df[df.author!="RepostBawt"]
df=df[df.author!="WikiTextBot"]

# we just droping the rest of the duplicates
df=df.drop_duplicates(subset=['body'], keep='last')
df = df.reset_index(drop=True)
df=df.drop(df.index[499])
#also remove moderators messages.
df=df[df.distinguished!='moderator']

   
#let's see links
import re

# uses a regex to detect links and replaces them by LINK  
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

# Convert to list
data = df.clean.values.tolist()
print(data[:1])

#Tokenize
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

print(data_words[:1])
##Bigrams are two words frequently occurring together in the document. Trigrams are 3 words frequently occurring.
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[100]]])


###LET'S DO IT AGAIN


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# Form Bigrams
data_words_bigrams = make_bigrams(data_words)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)

nlp=spacy.load('en_core_web_sm',disable=['parse', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized[0:99])

# Create Corpus
texts = data_lemmatized[0:99]

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])
#Gensim creates a unique id for each word in the document. The produced corpus shown above is a mapping of (word_id, word_frequency).
id2word[0]
# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
##BUILDING TOPIC MODEL
from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary, bleicorpus
import numpy
from gensim.matutils import hellinger
time_slice = [5647, 8035, 3276]
time_slice=[25,50,25]
ldaseq = ldaseqmodel.LdaSeqModel(corpus=corpus, id2word=id2word, time_slice=time_slice, num_topics=5)
ldaseq.print_topic_times(topic=0) # evolution of 1st topic

###FOR EVALUATION.
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
 
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus,texts=texts, start=2, limit=40, step=6)
# Show graph
import matplotlib.pyplot as plt
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
import numpy
v1 = numpy.asarray([0., 2.], dtype='f')
v2 = numpy.asarray([0., 1.], dtype='f')
print(numpy.dot(v1, v2))