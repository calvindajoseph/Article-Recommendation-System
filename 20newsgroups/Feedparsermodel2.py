#this .py will retrieve the latest news articles using RSS feeds python.
#the problem is that the model's accuracy is 82 with LinearSVC and we have to increase it to get better predicitons.
#We have been trying various ways to improve the model's accuracy but with no use.
#at the end of this please chnage it to your path to export the dataframe in csv.
import feedparser

#imports numpy and pandas
import numpy as np
import pandas as pd
import logging
from time import time
import sys

#imports sklearn libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# ----------------------------Calvin rss feeds----------------------

vectorizer = TfidfVectorizer(stop_words='english')
shuffle = True
remove = ['footers','quotes']

line_length = 120

#fetch train dataset and test dataset from 20newsgroups
#each subsets are shuffled and got their headers, footers, and quotes removed
newsgroup_train = fetch_20newsgroups(subset='train',
                                     shuffle=shuffle,
                                     remove=remove)
newsgroup_test = fetch_20newsgroups(subset='test',
                                    shuffle=shuffle,
                                    remove=remove)

#vectorize the datasets using TFIDF
X_train = vectorizer.fit_transform(newsgroup_train.data)
y_train = newsgroup_train.target
X_test = vectorizer.transform(newsgroup_test.data)

def print_line():
    print("="*line_length)

def model_header(model_name):
    print('')
    print_line()
    print(model_name)
    print_line()
    print('')
    return 0

def model_footer():
    print_line()

def custom_test_vectorizer(doc):
    vectors = vectorizer.transform(doc)
    return vectors

def custom_test(clf,doc):
    predicted = clf.predict(custom_test_vectorizer(doc))

    for doc, category in zip(doc, predicted):
        print('%r => %s' % (doc, newsgroup_train.target_names[category]))
    return predicted

def show_top10(classifier, vectorizer, categories):
    feature_names = vectorizer.get_feature_names()
    feature_names = np.asarray(feature_names)
    print("Top 10 keyword per class:")
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s:\t\t %s" % (category, " ".join(feature_names[top10])))

def model_benchmark(model_name,text_clf):
    model_header(model_name)

    print("Model specification:")
    print(text_clf)

    t0 = time()
    text_clf.fit(X_train,y_train)
    train_time = time() - t0
    print("Train time: %0.3fs" % train_time)

    t0 = time()
    test_pred = text_clf.predict(X_test)
    test_time = time() - t0
    print("Test time:  %0.3fs" % test_time)

    print('')

    print("Classification Report:")
    print(metrics.classification_report(newsgroup_test.target, test_pred,
                                        target_names=newsgroup_test.target_names))

    show_top10(text_clf,vectorizer,newsgroup_train.target_names)

    print('')

    model_footer()

    return text_clf

final_model = model_benchmark('LinearSVC',
                              LinearSVC(loss='squared_hinge',
                              penalty='l2', dual=False,
                              tol=1e-3))

# The RSS FEED Starts Here
# I did not change anything above this comment

feed_source = [['http://www.abc.net.au/news/feed/2942460/rss.xml', 'ABC Australia'],
               ['http://www.9news.com.au/rss', '9NEWS'],
               ['http://www.dailytelegraph.com.au/feed', 'Daily Telegraph'],
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

df_recommendation_article_by_model2 = pd.DataFrame(columns=['Class_No',
                                                  'Class_Name',
                                                  'Article_ID',
                                                  'Article_Title'])


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

    return df


def rss_feed(feed_source, df, no_of_articles_per_source):
    for idx, source in enumerate(feed_source):
        feed = feedparser.parse(source[0])
        df = rss_feed_to_dataframe(feed, df, no_of_articles_per_source, source[1])
    return df


def article_preprocess_model2(df):
    length = len(df)

    title_desc_list = [None] * length
    for i in range(length):
        title_desc_list[i] = df.iloc[i, 0] + '. ' + df.iloc[i, 1]

    return title_desc_list


def predict_article_model2(df, clf):
    text = article_preprocess_model2(df)

    predicted = clf.predict(vectorizer.transform(text))

    return predicted

def set_recommendation_articles_model2(df_recommendation,df_article):
    classes_model2 = df_article['Class_Model2'].to_numpy()

    for idx, i in enumerate(classes_model2):
        if df_recommendation.iloc[i,2] is None:
            df_recommendation.iloc[i, 2] = idx
            df_recommendation.iloc[i, 3] = df_article.iloc[idx, 0]

    return df_recommendation

def export_Dataframe_to_csv(df, path, filename):
    path = path + '\\' + filename
    df.to_csv(path)

    print('Dataframe exported to ' + path)
    return 0

df_recommendation_article_by_model2 = set_recommendation_Dataframe_model2(df_recommendation_article_by_model2)

df_articles = rss_feed(feed_source, df_articles, no_of_articles_per_source)
df_articles['Class_Model2'] = predict_article_model2(df_articles,final_model)

df_recommendation_article_by_model2 = set_recommendation_articles_model2(df_recommendation_article_by_model2,df_articles)

#export_Dataframe_to_csv(df_articles,*PUT_YOUR_FILEPATH_HERE*,'articles.csv')
#export_Dataframe_to_csv(df_recommendation_article_by_model2,*PUT_YOUR_FILEPATH_HERE*,'articles_recommendation.csv')