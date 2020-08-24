import feedparser

import math
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.datasets import fetch_20newsgroups

import pickle
from joblib import dump, load

path = r'C:\Users\calda\OneDrive\Documents\University\Semester 1 2020\Engineering Project (Part A) 10004\Test Folder'
filename_model2 = 'model2.joblib'
filename_model2_class_dataframe = 'Model2_Class.csv'

filename_article_database = 'article_database.csv'

vectorizer = TfidfVectorizer(stop_words='english')
shuffle = True
remove = ['footers','quotes']

newsgroup_train = fetch_20newsgroups(subset='train',
                                     shuffle=shuffle,
                                     remove=remove)

X_train = vectorizer.fit_transform(newsgroup_train.data)

feed_source = [['http://www.abc.net.au/news/feed/2942460/rss.xml', 'ABC Australia'],
               ['http://www.9news.com.au/rss', '9NEWS'],
               ['https://www.dailytelegraph.com.au/news/breaking-news/rss', 'Daily Telegraph'],
               ['http://feeds.smh.com.au/rssheadlines/top.xml', 'SMH Australian Breaking News'],
               ['https://www.news.com.au/feed/', 'News.com.au'],
               ['https://www.theaustralian.com.au/feed/', 'The Australian RSS Feed']]

no_of_articles_per_source = 40

df_articles = pd.DataFrame(columns=['Title',
                                    'Description',
                                    'Author',
                                    'Category',
                                    'Source',
                                    'Link',
                                    'Class_Model1',
                                    'Class_Model2'])

def load_model(path, filename):
    path = path + '\\' + filename
    clf = load(path)
    print('Loaded %r from %r' % (filename, path))
    return clf

def set_recommendation_Dataframe_model2(df):
    for i in range(20):
        df = df.append({'Class_No' : i,
                        'Class_Name' : newsgroup_train.target_names[i],
                        'Article_ID' : None,
                        'Article_Title' : None},
                       ignore_index=True)
    return df

def remove_html_variables(text):
    text = text.replace('<p>', '')
    text = text.replace('</p>', '')
    return text


def rss_feed_to_dataframe(feed, df, no_of_article, source):
    count = 0
    for i in range(no_of_article):
        article_tuple = [None] * 8

        try:
            article_tuple[0] = remove_html_variables(feed.entries[i].title)
        except AttributeError as error:
            article_tuple[0] = None
        except IndexError as error:
            break

        try:
            article_tuple[1] = remove_html_variables(feed.entries[i].description)
        except AttributeError as error:
            article_tuple[1] = None

        try:
            article_tuple[2] = feed.entries[i].author
        except AttributeError as error:
            article_tuple[2] = None

        try:
            article_tuple[3] = feed.entries[i].category
        except AttributeError as error:
            article_tuple[3] = None

        article_tuple[4] = source

        try:
            article_tuple[5] = feed.entries[i].link
        except AttributeError as error:
            article_tuple[5] = None

        article_tuple[6] = None
        article_tuple[7] = None

        df = df.append({'Title': article_tuple[0],
                        'Description': article_tuple[1],
                        'Author': article_tuple[2],
                        'Category': article_tuple[3],
                        'Source': article_tuple[4],
                        'Link': article_tuple[5],
                        'Class_Model1': article_tuple[6],
                        'Class_Model2': article_tuple[7]},
                       ignore_index=True)
        count = count + 1

    return df

def count_articles(feed, max_article):
    count = 0
    test = None

    for i in range(max_article):
        try:
            test = feed.entries[i].title
        except IndexError as error:
            break

        count = count + 1

    return count

def rss_feed(feed_source, df, no_of_articles_per_source):
    for idx, source in enumerate(feed_source):
        feed = feedparser.parse(source[0])
        df = rss_feed_to_dataframe(feed, df, no_of_articles_per_source, source[1])
        count = count_articles(feed,no_of_articles_per_source)
        print("Stored %i articles from %r source" % (count, source[1]))
    return df


def article_preprocess_model2(df):
    length = len(df)

    title_desc_list = [None] * length
    for i in range(length):
        title_desc_list[i] = df.iloc[i, 0] + '. ' + df.iloc[i, 1]

    return title_desc_list


def predict_article_model2(df, clf, vectorizer):
    text = article_preprocess_model2(df)

    vectors = vectorizer.transform(text)
    predicted = clf.predict(vectors)

    return predicted

def set_recommendation_articles_model2(df_recommendation,df_article):
    classes_model2 = df_article['Class_Model2'].to_numpy()

    for idx, i in enumerate(classes_model2):
        if math.isnan(df_recommendation.iloc[i, 2]):
            df_recommendation.iloc[i, 2] = idx
            df_recommendation.iloc[i, 3] = df_article.iloc[idx, 0]

    return df_recommendation

def export_Dataframe_to_csv(df, path, filename):
    path = path + '\\' + filename
    df.to_csv(path, index=False)

    print('Dataframe exported to ' + path)
    return 0

def import_csv_to_Dataframe(path, filename):
    path = path + '\\' + filename
    df = pd.read_csv(path)
    print('Dataframe imported from ' + path)
    return df

def import_model2_categories(path,filename):
    df = pd.DataFrame
    df = import_csv_to_Dataframe(path, filename)
    df.drop([0,1])
    return df

df_recommendation_article_by_model2 = import_model2_categories(path, filename_model2_class_dataframe)

df_articles = rss_feed(feed_source, df_articles, no_of_articles_per_source)
df_articles['Class_Model2'] = predict_article_model2(df_articles,load_model(path, filename_model2), vectorizer)

df_recommendation_article_by_model2 = set_recommendation_articles_model2(df_recommendation_article_by_model2,df_articles)

print(df_recommendation_article_by_model2.head(2))
#print(df_recommendation_article_by_model2.info())

export_Dataframe_to_csv(df_articles,path,filename_article_database)
export_Dataframe_to_csv(df_recommendation_article_by_model2,path,filename_model2_class_dataframe)