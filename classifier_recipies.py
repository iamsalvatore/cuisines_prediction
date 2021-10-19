import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def dataframe():
    recipes = pd.read_csv("recipes.csv", delimiter = ',')
    X = recipes.loc[:, recipes.columns != 'cuisine']
    X_arr = X.to_numpy()
    y = recipes['cuisine']
    vocabulary = recipes.columns.tolist()

    return X_arr, y

X_arr,y =dataframe()


def tf_idf_transform(X_arr,y):
    tfidf_transformer = TfidfTransformer()
    X_transformed = tfidf_transformer.fit_transform(X_arr)
    X_train,X_test,y_train,y_test = train_test_split(X_transformed, y, test_size=0.25)

    return X_train,X_test,y_train,y_test

X_train,X_test,y_train,y_test=tf_idf_transform(X_arr,y)



def simple_logistic_classify(X_tr, y_tr, X_test, y_test, description):
    m = LogisticRegression().fit(X_tr, y_tr)
    s = m.score(X_test, y_test)
    print ('Test score with', description, 'features:', s)
    return m

m1 = simple_logistic_classify(X_train, y_train, X_test, y_test, 'tf-idf')



def svm(X_tr, y_tr, X_test, y_test, description):
    m = SVC().fit(X_tr, y_tr)
    s = m.score(X_test, y_test)
    print ('Test score with', description, 'features:', s)
    return m

m2 = svm(X_train, y_train, X_test, y_test, 'tf-idf_svm')

