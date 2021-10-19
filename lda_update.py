import os
import csv
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim import similarities
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel


def index2word(features, feature_names):
  filename = 'word_features.pkl'
  if os.path.exists(filename):
    with open(filename, 'rb') as file:
      word_features = pickle.load(file)
  else:
    word_features = []
    for row in range(features.shape[0]):
      word_feature = []
      for col in range(features.shape[1]):
        if features[row, col]:
          word_feature.append(feature_names[col].replace(' ', '_'))
      word_features.append(word_feature)
    with open(filename, 'wb') as file:
      pickle.dump(word_features, file)
  return word_features


def get_features(args):
  recipes = pd.read_csv(os.path.join(args.input_dir, 'recipes.csv'), header=0)
  header = recipes.columns.values.tolist()
  feature_names = {i: header[i].strip("'") for i in range(len(header))}
  features = recipes.to_numpy()[:, :-1]
  # shuffle dataset
  indexes = np.arange(len(features))
  np.random.shuffle(indexes)
  features = features[indexes]
  # convert index to words
  features = index2word(features, feature_names)
  return features


def train_lda_model(ingredients, num_topics=200, passes=15, random_state=1234):
  filename = 'lda_model.pkl'
  if os.path.exists(filename):
    with open(filename, 'rb') as file:
      checkpoint = pickle.load(file)
    lda_model = checkpoint['lda_model']
    dictionary = checkpoint['dictionary']
    corpus = checkpoint['corpus']
  else:
    dictionary = Dictionary(ingredients)
    corpus = [dictionary.doc2bow(text) for text in ingredients]
    lda_model = LdaModel(corpus,
                         num_topics=num_topics,
                         passes=passes,
                         random_state=random_state,
                         id2word=dictionary)
    with open(filename, 'wb') as file:
      pickle.dump(
          {
              'lda_model': lda_model,
              'dictionary': dictionary,
              'corpus': corpus
          }, file)
  return lda_model, dictionary, corpus


def get_similarity(lda_model, query_vector, corpus):
  index = similarities.MatrixSimilarity(lda_model[corpus])
  return index[query_vector]


def calculate_similarity(query, lda_model, dictionary, corpus):
  assert type(query) == list
  words_bow = dictionary.doc2bow(query)
  query_vector = lda_model[words_bow]
  similarity_rank = get_similarity(lda_model=lda_model,
                                   query_vector=query_vector,
                                   corpus=corpus)
  return sorted(enumerate(similarity_rank), key=lambda item: -item[1])


def calculate_recommendation(similarity_rank, groups, num_recommendation=10):
  results = [similarity_rank[0][0]]
  results_probabilities = [similarity_rank[0][1]]
  result_group = [similarity_rank[0][1]]
  for recipe, group in zip(similarity_rank[1:], groups[1:]):
    if group not in set(result_group):
      results.append(recipe[0])
      result_group.append(group)
      results_probabilities.append(recipe[1])
    if len(results) == num_recommendation:
      break
  return results, results_probabilities


def get_similarity_recommendation(query,
                                  lda_model,
                                  dictionary,
                                  corpus,
                                  num_recommendation=10):
  similarity_rank = calculate_similarity(query, lda_model, dictionary, corpus)
  groups = []
  for l in lda_model[corpus]:
    try:
      groups.append(l[0][0])
    except IndexError:
      groups.append(random.randint(1, 100))
  return calculate_recommendation(similarity_rank, groups, num_recommendation)


def get_missing_ingredients(query, results, features):
  missing_ingredients = []
  for result in results:
    recipe = features[result]
    missing_ingredient = []
    for ingredient in recipe:
      if ingredient not in query:
        missing_ingredient.append(ingredient)
    missing_ingredients.append(missing_ingredient)
  return missing_ingredients


def main(args):
  """ 
LDA based collaborative filtering

Blog: https://medium.com/analytics-vidhya/how-to-build-personalized-recommendation-from-scratch-recipes-from-food-com-c7da4507f98
LDA implementation: https://github.com/ZeeTsing/Recipe_reco
"""
  random.seed(args.seed)
  np.random.seed(args.seed)

  print('get dataset')
  features = get_features(args)

  print('train LDA model')
  lda_model, dictionary, corpus = train_lda_model(features)

  queries = [
           ['egg', 'onion'],
      ['haddock','milk'],
      [
          'carrot','lamb','potato'
      ],
      [ 'egg','green_onion','rice','shrimp','soy sauce'],

          ['apple','butter','chicken stock','nutmeg','onion','cinnamon', 'egg'
      ],
      ['butter','egg','onion','garlic','pea','rice','pork'],

          ['fish_sauce', 'jalapeno_pepper', 'lettuce', 'lime', 'onion', 'pepper',
          'salt'
      ],
      [
          'garlic', 'ginger', 'green_onion', 'lemon_grass', 'lime', 'turkey',
          'water'
      ],
      ['red_pepper', 'flakes', 'soy_sauce'],
      ['cilantro', 'fish_sauce', 'ginger', 'green_onion', 'lemon'],
      ['fish_sauce', 'green_bean', 'peanut_oil', 'sugar', 'tofu'],
  ]

  for query in queries:
    print(f'\nInput query: {query}')
    recommendations, similarity_scores = get_similarity_recommendation(
        query,
        lda_model,
        dictionary=dictionary,
        corpus=corpus,
        num_recommendation=5)
    suggestions = get_missing_ingredients(query, recommendations, features)
    print('Top recommendations')
    for i, suggestion in enumerate(suggestions):
      print(f'\tscore: {similarity_scores[i]:.4f}'
            f'\t\tmissing ingredients: {suggestion}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', type=str, default='dataset/recipes')
  parser.add_argument('--output_dir', type=str, default='runs')
  parser.add_argument('--clear_output_dir', action='store_true')
  parser.add_argument('--seed', type=int, default=1234)
  parser.add_argument('--verbose', default=True, type=bool)
  main(parser.parse_args())
