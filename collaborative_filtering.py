import pandas as pd
import numpy as np
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import SVD
from surprise import BaselineOnly
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise.model_selection import KFold
from surprise import CoClustering
from surprise.accuracy import rmse
from surprise import accuracy
from surprise.model_selection import train_test_split
import os


def build_newdataframe():
    df = pd.read_csv("dataset/recipes/recipes.csv", delimiter = ',')
    # Dropping cuisine column
    df = df.drop(['cuisine'], axis = 1)
    number = df.sum(axis=0)
    few_n_ingredient = list(number<=4)
    df1=df.drop(df.columns[few_n_ingredient],axis=1)
    # print(number)
    # exit()
    df1 = df.T.reset_index()
    df1 = df1[['index']]
    df1['index'] = df1.astype(str)
    df1 = pd.concat([df1]*len(df1), axis=0)
    x=list(range(1, len(df1)+1))
    df1['userid']= np.repeat(x,len(df1.columns))
    df1 = pd.DataFrame(df1)
    df1 = df1.reset_index(drop=True)
    print(df1.shape)
    return df1

df1 = build_newdataframe()


def final_merge():
    df3= pd.read_csv("dataset/recipes/recipes.csv", delimiter = ',')
    df3= df3.drop(['cuisine'], axis = 1)
    number = df3.sum(axis=0)
    few_n_ingredient = list(number<=4)
    df3=df3.drop(df3.columns[few_n_ingredient],axis=1)
    df3=df3.T.reset_index()
    df_ratings= df3.drop([0], axis=1)
    new_column= df_ratings.set_index('index').stack()
    new_column= new_column.astype(float)
    new_column= new_column.reset_index(drop=True, level=1).reset_index(name='rating')
    new_column = new_column.reset_index(drop=True)
    new_column= new_column.drop(['index'], axis=1) 
    final_dataset = pd.DataFrame(pd.concat([df1, new_column], join="inner",names=["index"],axis=1))
    columns_titles = ['userid', 'index','rating']
    final_dataset=final_dataset.reindex(columns=columns_titles)
    print(final_dataset.shape)
    # save_dataset= final_dataset.to_pickle('final_data.pkl')
    save_dataset= final_dataset.to_csv('final_data.csv', index=False)

    return final_dataset

final_merge()


def training_SVD():
    # path to dataset file
    file_path = os.path.expanduser('final_data.csv')
    data = pd.read_csv(file_path, encoding = 'unicode_escape', delimiter = ',')
    print(data.columns)
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(data[['userid', 'index','rating']], reader=reader)
  # define a cross-validation iterator
    kf = KFold(n_splits=2)
    algo = KNNWithZScore()
    for trainset, testset in kf.split(data):
        # train and test algorithm.
        algo.fit(trainset)
        predictions = algo.test(testset)
        # Compute and print Root Mean Squared Error
        accuracy.rmse(predictions, verbose=True)
    for uid in range(1, 4237):
        prediction = algo.predict(uid, 'sour cream')
    if prediction.est > 0.03:
        print(prediction.est, uid)


training_SVD()


