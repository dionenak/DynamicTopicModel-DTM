# Dynamic Topic Modeling on Reddit dataset
____________________________________________________________________
***Dataset***

We collected our data from dataset “fh-bigquery: reddit_comments”
and table “2018_03”2, using BigQuery . The table contained all
public Reddit comments from March 2018.Dataset published and compiled
by /u/Stuck_In_the_Matrix, in r/datasets. Access
https://bigquery.cloud.google.com/table/fhbigquery:reddit_comments.2018_03

______________________________________________________________________

***Scripts***

In folder "Analysis", we have three scripts:
- PrepDS: includes the preprocessing steps. Import our dataset and cleaning processes 
(NAs, dates, duplicates etc). Also, we removed posts from several bot accounts, after 
inspecting their profiles, in order to confirm that they were indeed bots. Next, we removed
html and urls, stopwords, found bigrams and lemmatized. Lastly, we saved as pickle files what
we would need further in our analysis.
