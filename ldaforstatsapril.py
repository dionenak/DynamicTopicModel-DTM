# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 13:29:39 2020

@author: dio
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 12:37:42 2020

@author: dio
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
from gensim.models.ldamodel import LdaModel

# spacy for lemmatization
!pip install spacy
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline


df = pd.read_excel('april2.xlsx')

#Remove the empty columns
df.drop(df.columns[[20]], axis=1, inplace=True)
#Remove controversiality, downs, ups, edited, retrieved_on, score,name, archived, score_hidden,
#author_flair_css_class, author_flair_text, link_id, parent_id, subreddit_id, id,gilded
#KEEP distinguished, body, created_utc, author,subreddit
df.drop(df.columns[[1,2,3,5,6,8,9,10,11,12,13,14,15,17,19]], axis=1, inplace=True)
print(df.columns)
#Inspection
#Which columns have NAs
df.isna().sum()
#body:0, aythor:14, created_utc:49,subreddit:0, distinguished:9384
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

df = df[df.author != 'autotldr']
df = df[df.author != 'ClickableLinkBot']
df = df[df.author != 'alternate-source-bot']
df = df[df.author != 'Decronym']
df = df[df.author != 'Mentioned_Videos']
df = df[df.author != 'AutoModerator']
df = df[df.author != 'LocationBot']
df = df[df.author != 'TotesMessenger']
df = df[df.author != 'DoMoreWithLess']
df = df[df.author != 'rtbot2']
df = df[df.author != 'transcribot']
df = df[df.author != 'ama_compiler_bot']
df = df[df.author != 'TweetTranscriber']

isitbot=df[df.author.isin(['mvea','ClickableLinkBot','comment_preview_bot','TweetMirrorBot','Decronym', 'TotesMessenger','Automoderator','LocationBot','Mentioned_Videos','DoMoreWithLess','BernardJOrtcutt','ImagesOfNetwork','SteamKiwi','TrendingCommenterBot','shortscience_dot_org','WorldNewsMods'])]
isitbot=df[df.author.isin(['_youtubot_','rtbot2','tweettranscriberbot','transcribot','topredditbot'])]


isitbot2= df[df['author'].str.contains("bot",na=False)] 
isitbot2= df[df['author'].str.contains("bawt",na=False)]
isitbot2= df[df['body'].str.contains(" bot ",na=False)] 


# we just droping the rest of the duplicates
df=df.drop_duplicates(subset=['body'], keep='last')
df = df.reset_index(drop=True)

#also remove moderators messages.
df=df[df.distinguished!='moderator']


#1-7
df = df[~(df['created_iso'] > '2018-04-08')]
#PREPROCESSING
  
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



#Lowercasing
df['clean']= df['clean'].map(lambda x: x.lower())
df['clean'].replace(regex=True,inplace=True,to_replace=r'self.?driving (car|cars)|autonomous (car|cars)|autonomous.?(vehicle|vehicles)|automated (car|cars)|automated driving|driver.?less (car|cars)|automated vehicle',value=r'selfdrivingcar ')
df['clean'].replace(regex=True,inplace=True,to_replace=r'self driving ',value=r'selfdriving ')



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

#See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])


###LET'S DO IT 


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'VERB']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# Form Bigrams
data_words_bigrams = make_bigrams(data_words)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
import spacy
nlp=spacy.load('en_core_web_sm',disable=['parse', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'VERB'])

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

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
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
        model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=2, limit=40, step=6)
# Show graph
import matplotlib.pyplot as plt
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
#UMASS
def compute_coherence_values_UMASS(dictionary, corpus, texts, limit, start=2, step=2):
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
        model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values
model_list, coherence_values = compute_coherence_values_UMASS(dictionary=id2word, corpus=corpus, texts=texts, start=2, limit=40, step=6)
# Show graph
import matplotlib.pyplot as plt
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
#Let's do two lda models
##topic 8
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=8, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))
#topic 14
# Visualize the topics

vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.show(vis)
#which docu to which topic
optimal_model=lda_model
model_topics = optimal_model.show_topics(formatted=False)
print(optimal_model.print_topics(num_words=20))

def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=texts)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(20)

# Display setting to show more characters in column
pd.options.display.max_colwidth = 100

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

# Show
sent_topics_sorteddf_mallet.head(20)
##topic 14
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=14, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))
#topic 14
# Visualize the topics

vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.show(vis)
#SEE TOP DOCUM
optimal_model=lda_model
model_topics = optimal_model.show_topics(formatted=False)
print(optimal_model.print_topics(num_words=10))
#which docu to which topic
def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=texts)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(20)

# Display setting to show more characters in column
pd.options.display.max_colwidth = 100

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

# Show
sent_topics_sorteddf_mallet.head(10)