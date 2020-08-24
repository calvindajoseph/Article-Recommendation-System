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

import pickle
from joblib import dump, load

# ----------------------------Calvin rss feeds----------------------

vectorizer = TfidfVectorizer(stop_words='english')
shuffle = True
remove = ['footers','quotes']

path = r'C:\Users\calda\OneDrive\Documents\University\Semester 1 2020\Engineering Project (Part A) 10004\Test Folder'
filename_model2 = 'model2.joblib'
filename_model2_class_dataframe = 'Model2_Class.csv'

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

def store_model(clf, path, filename):
    path = path + '\\' + filename
    dump(clf, path)
    print('Stored %r to %r' % (filename, path))
    return 0

def set_recommendation_Dataframe_model2(df):
    for i in range(20):
        df = df.append({'Class_No' : i,
                        'Class_Name' : newsgroup_train.target_names[i],
                        'Article_ID' : None,
                        'Article_Title' : None},
                       ignore_index=True)
    return df

def export_Dataframe_to_csv(df, path, filename):
    path = path + '\\' + filename
    df.to_csv(path, index=False)

    print('Dataframe exported to ' + path)
    return 0

final_model = model_benchmark('LinearSVC',
                              LinearSVC(loss='squared_hinge',
                              penalty='l2', dual=False,
                              tol=1e-3))

store_model(final_model, path, filename_model2)

df_recommendation_article_by_model2 = pd.DataFrame(columns=['Class_No',
                                                  'Class_Name',
                                                  'Article_ID',
                                                  'Article_Title'])

df_recommendation_article_by_model2 = set_recommendation_Dataframe_model2(df_recommendation_article_by_model2)

export_Dataframe_to_csv(df_recommendation_article_by_model2,path,filename_model2_class_dataframe)