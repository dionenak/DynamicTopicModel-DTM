# -*- coding: utf-8 -*-
import gensim
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
#Open/Load our dictionary and corpus
import pickle
f_dic1 = open("dict_1.pkl",'rb')  
id2word_1= pickle.load(f_dic1)
f_dic2 = open("dict_2.pkl",'rb')  
id2word_2= pickle.load(f_dic2)
f_dic3 = open("dict_3.pkl",'rb') 
id2word_3= pickle.load(f_dic3)
f_cor=open("corp.pkl","rb")
texts=pickle.load(f_cor)

#But first we have to split our corpus in each time slice
text_1=texts[:1816]
text_2=texts[1816:4633]
text_3=texts[4633:]

corpus_1 = [id2word_1.doc2bow(text) for text in text_1]
corpus_2 = [id2word_2.doc2bow(text) for text in text_2]
corpus_3 = [id2word_3.doc2bow(text) for text in text_3]


#For plot
import matplotlib.pyplot as plt
#We picked these numbers
limit=40; start=2; step=6;

#COHERENCE for slice 1
#C_V MEASURE
def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
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
## Cv for timeslice1
model_list, coherence_values = compute_coherence_values(dictionary=id2word_1,
                                                        corpus=corpus_1, texts=text_1, start=2, limit=40, step=6)

# Show graph
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
## Cv for timeslice2
model_list, coherence_values = compute_coherence_values(dictionary=id2word_2,
                                                        corpus=corpus_2, texts=text_2, start=2, limit=40, step=6)
# Show graph
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
##Cv for timeslice3
model_list, coherence_values = compute_coherence_values(dictionary=id2word_3,
                                                        corpus=corpus_3, texts=text_3, start=2, limit=40, step=6)
# Show graph
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
#UMASS MEASURES
def compute_coherence_values_UMASS(dictionary, corpus, texts, limit, start=2, step=1):
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
#Umass for timeslice1
model_list, coherence_values = compute_coherence_values_UMASS(dictionary=id2word_1, 
                                                             corpus=corpus_1, texts=text_1, start=2, limit=40, step=6)
# Show graph
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
##Umass for timeslice2
model_list, coherence_values = compute_coherence_values_UMASS(dictionary=id2word_2,
                                                             corpus=corpus_2, texts=text_2, start=2, limit=40, step=6)
# Show graph
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
##Cv for timeslice3
model_list, coherence_values = compute_coherence_values_UMASS(dictionary=id2word_3, corpus=corpus_3, texts=text_3, start=2, limit=40, step=6)
# Show graph
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


f_dic1.close()
f_dic2.close()
f_dic3.close()
f_cor.close()