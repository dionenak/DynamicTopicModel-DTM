# -*- coding: utf-8 -*-
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

df = pd.read_excel('marchxl.xlsx')
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
 
 date_iso=datetime.utcfromtimestamp(df.loc[i,'created_utc']).isoformat()
 list_dateiso.append(date_iso)

df['created_iso']=list_dateiso


