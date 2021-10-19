import pandas as pd
import numpy as np
import os
import gensim
import random
from sklearn.feature_extraction.text import CountVectorizer
from gensim.matutils import Sparse2Corpus
from gensim.models.ldamodel import LdaModel
from gensim import models, similarities
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import re
import gensim.corpora as corpora
from wordcloud import WordCloud
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
pyLDAvis.enable_notebook()
import pickle
from pprint import pprint
import matplotlib.pyplot as plt
import re

data = pd.read_csv('dataset/recipes/recipes-mallet_new.txt', names=('index', 'cuisine', 'ingredients'), delimiter='|')


def create_word_cloud():
    # Join the different processed titles together.
    combine_words = ','.join(list(data['ingredients'].values))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(combine_words)
    # Visualize the word cloud
    wordcloud= plt.imshow(wordcloud)

create_word_cloud()


def treat_ingredients(ing_list):
    output = []
    for ingredient in ing_list:
        ingredient_list = ingredient.split(' ')
        output.append("_".join(ingredient_list))
    return output


def get_similarity(lda, query_vector):
    index = similarities.MatrixSimilarity(lda[corpus])
    sims = index[query_vector]
    
    return sims


def train_lda_model(ingredients,num_topics = 300,passes = 15,random_state = 0):
    # get list of ingredients transformed
    ingredients_all = ingredients.apply(lambda x: treat_ingredients(x))
    
    #build dict (for Gensim vectorizer)
    dictionary = Dictionary([ing for ing in list(ingredients_all)])
    
    # build corpus: BOW
    corpus = [dictionary.doc2bow(text) for text in list(ingredients_all)]
    
    #train lda
    ldamodel = LdaModel(corpus,num_topics = num_topics, passes = passes,random_state = random_state, id2word = dictionary)

    return ldamodel,dictionary,corpus


# define treat input function, returning a list of tokenized ingredients
def treat_words (words):
    list_words = words.split(",")
    output = []
    for w in list_words:
        output.append("_".join(w.strip().split(" ")))
    return output


def calculate_similarity(query,ldamodel,dct):
    # treat input words
    words_bow = dct.doc2bow(treat_words(query))
    query_vector = ldamodel[words_bow]
    
    #calculate ranking
    sim_rank = get_similarity(lda = ldamodel, query_vector = query_vector)
    sim_rank = sorted(enumerate(sim_rank), key=lambda item: -item[1])
    
    return sim_rank



def calculate_recommendation(sim_rank,groups,n_reco = 10):
    results = [sim_rank[0][0]]
    results_prob = [sim_rank[0][1]]
    result_group = [sim_rank[0][1]]
        
    for recipe,group in zip(sim_rank[1:],groups[1:]):
        if group not in set(result_group):
            results.append(recipe[0])
            result_group.append(group)
            results_prob.append(recipe[1])
        if len(results) == n_reco:
            break
    print(result_group,"\n",results_prob)
    return results


# this is a wrapper function for calculate simu and calculate reco
def get_similarity_reco (query,ldamodel,dct,corpus,n_reco = 10):
    #calculate rank
    sim_rank = calculate_similarity(query,ldamodel,dct)
    #find groups according to lda model
    groups = []
    for l in ldamodel[corpus]:
        try:
            groups.append(l[0][0])
        except:
            groups.append(random.randint(1, 100))
            
    return calculate_recommendation(sim_rank,groups,n_reco)


def print_reco(results):
    return data.iloc[results]


def pretty_name (name):
    return " ".join([ word.capitalize() for word in name.split(" ") if word != ""])


ldamodel,dictionary,corpus = train_lda_model(data.ingredients)

query = 'apple,cabbage,cider vinegar'

results = get_similarity_reco (query, ldamodel, dct = dictionary, corpus = corpus,n_reco = 1)

print_reco(results)
