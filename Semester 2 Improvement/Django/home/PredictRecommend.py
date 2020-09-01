from .Global import Global
from .Global import Model1
from .Global import Model2

import feedparser

import math
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import pickle
from joblib import dump, load

const = Global()

def predict(sentiment, path, filename_clf, filename_vectorizer):
    clf = load(path + '\\' + filename_clf)
    vectorizer = load(path + '\\' + filename_vectorizer)

    text = [sentiment]
    vectors = vectorizer.transform(text)
    predicted = clf.predict(vectors)[0]
    return predicted

def import_csv_to_Dataframe(path, filename):
    path = path + '\\' + filename
    df = pd.read_csv(path)
    return df

def get_recommendation(class1, class2, path, filename_categories, filename_article_database):
    df_categories = import_csv_to_Dataframe(path, filename_categories)
    df_database = import_csv_to_Dataframe(path, filename_article_database)

    df_temp = df_categories.loc[(df_categories['Class_Model1'] == class1) & (df_categories['Class_Model2'] == class2)]
    article_id = df_temp.iloc[0, 2]
    try:
        recommendation_article = df_database.iloc[int(article_id)].to_numpy()
    except ValueError as error:
        recommendation_article = [None] * (len(df_database.columns) - 2)

    return recommendation_article

class Sentiment:

    def __init__(self, sentiment):
        self.sentiment = sentiment
        self.class_model1 = Model1(predict(sentiment, const.path, const.filename_model1, const.filename_vectorizer_model1))
        self.class_model2 = Model2(predict(sentiment, const.path, const.filename_model2, const.filename_vectorizer_model2))

        self.tag_model1 = self.class_model1.tag
        self.tag_model2 = self.class_model2.tag

        recommendation_article = get_recommendation(self.class_model1.modelclass, self.class_model2.modelclass, const.path, const.filename_article_recommendation, const.filename_article_database)

        self.title = recommendation_article[0]
        self.description = recommendation_article[1]
        self.author = recommendation_article[2]
        self.category = recommendation_article[3]
        self.source = recommendation_article[4]
        self.link = recommendation_article[5]
