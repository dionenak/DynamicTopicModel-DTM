
import pickle #import dictionary and text
from gensim.test.utils import datapath #save our model
from gensim.models import LdaSeqModel

import pandas as pd

from gensim.models.wrappers.dtmmodel import DtmModel
from gensim.corpora import Dictionary, bleicorpus
import pyLDAvis

f_dic = open("dict.pkl",'rb')  
id2word= pickle.load(f_dic)
f_cor=open("corp.pkl","rb")
texts=pickle.load(f_cor)

##BUILDING TOPIC MODEL
from gensim.models import ldaseqmodel
#from gensim.corpora import Dictionary, bleicorpus
#import numpy
#from gensim.matutils import hellinger
time_slice = [1816, 2817, 2164]
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

ldaseq = ldaseqmodel.LdaSeqModel(corpus=corpus, id2word=id2word, time_slice=time_slice, num_topics=8)

#save MODEL and Load Model

temp_file = datapath("model_seq")
ldaseq.save(temp_file)
temp_file2 = datapath("model_seq")
ldaseq_same = LdaSeqModel.load(temp_file2)




#see the topics and the numbers for each topic for each time slice in one dataframe
dfk_topics=pd.DataFrame()
count=0

for j in range(8): 
 model=ldaseq.print_topic_times(topic=j)

 for topic in range(len(model)):
    list1=[]
    list2=[]
    for i in range(len(model[topic])):
           list1.append(format(model[topic][i][0]))
           list2.append(format(model[topic][i][1]))
    dfk_topics[count] = pd.Series(list1)
    count=count+1
    dfk_topics[count] = pd.Series(list2)
    count=count+1

#see time=0


doc_topic, topic_term, doc_lengths, term_frequency, vocab = ldaseq.dtm_vis(time=0, corpus=corpus)
vis_dtm = pyLDAvis.prepare(topic_term_dists=topic_term, doc_topic_dists=doc_topic, doc_lengths=doc_lengths, vocab=vocab, term_frequency=term_frequency)
# This works best for me (then view dtm.html in a browser)
with open("slice0.html", "w") as f:
  pyLDAvis.save_html(vis_dtm, f)
  
#see time=1
doc_topic, topic_term, doc_lengths, term_frequency, vocab = ldaseq.dtm_vis(time=1, corpus=corpus)
vis_dtm1 = pyLDAvis.prepare(topic_term_dists=topic_term, doc_topic_dists=doc_topic, doc_lengths=doc_lengths, vocab=vocab, term_frequency=term_frequency)
# This works best for me (then view dtm.html in a browser)
with open("slice1.html", "w") as f:
  pyLDAvis.save_html(vis_dtm1, f)
  
#see time=2
doc_topic, topic_term, doc_lengths, term_frequency, vocab = ldaseq.dtm_vis(time=2, corpus=corpus)
vis_dtm2 = pyLDAvis.prepare(topic_term_dists=topic_term, doc_topic_dists=doc_topic, doc_lengths=doc_lengths, vocab=vocab, term_frequency=term_frequency)
# This works best for me (then view dtm.html in a browser)
with open("slice2.html", "w") as f:
  pyLDAvis.save_html(vis_dtm2, f)





#close the files
f_dic.close()
f_cor.close()

