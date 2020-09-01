from Global import Global

import pandas as pd
import numpy as np

import logging
from time import time
import sys

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

import pickle
from joblib import dump, load

const = Global()

def import_json_to_Dataframe(path, filename):
    path = path + '\\' + filename
    df = pd.read_json(path, lines=True)
    print('Dataframe imported from ' + path)
    return df

def import_csv_to_Dataframe(path, filename):
    path = path + '\\' + filename
    df = pd.read_csv(path)
    print('Dataframe imported from ' + path)
    return df

def export_Dataframe_to_csv(df, path, filename):
    path = path + '\\' + filename
    df.to_csv(path, index=False)

    #print('Dataframe exported to ' + path)
    return 0

def combine_two_columns_dataframe(df, column_1_name, column_2_name):
    length = len(df.index)
    temp_list = [None] * length

    for i in range(length):
        temp_list[i] = df[column_1_name] + ". " + df[column_2_name]

    return temp_list
#####################################################################################
#Pre-process dataset
#####################################################################################
def huffingtonpost_data_preprocessing(df):

    df.dropna(subset=['headline'], inplace=True)
    df['Text'] = df['headline'] + ". " + df ['short_description']

    df['category'] = df['category'].replace({
        "HEALTHY LIVING": "WELLNESS",
        "QUEER VOICES": "GROUPS VOICES",
        "BUSINESS": "BUSINESS & FINANCES",
        "PARENTS": "PARENTING",
        "BLACK VOICES": "GROUPS VOICES",
        "THE WORLDPOST": "WORLD NEWS",
        "STYLE": "STYLE & BEAUTY",
        "GREEN": "ENVIRONMENT",
        "TASTE": "FOOD & DRINK",
        "WORLDPOST": "WORLD NEWS",
        "SCIENCE": "SCIENCE & TECH",
        "TECH": "SCIENCE & TECH",
        "MONEY": "BUSINESS & FINANCES",
        "ARTS": "ARTS & CULTURE",
        "COLLEGE": "EDUCATION",
        "LATINO VOICES": "GROUPS VOICES",
        "CULTURE & ARTS": "ARTS & CULTURE",
        "FIFTY": "MISCELLANEOUS",
        "GOOD NEWS": "MISCELLANEOUS"
    })

    df['target'] = pd.factorize(df['category'])[0] + 20

    df_classes = pd.DataFrame(data={'no':df.target.unique(),'class':df.category.unique()})
    export_Dataframe_to_csv(df_classes, const.path, 'huffingtonpost_dataset_class.csv')

    #df.drop(columns=['headline', 'authors', 'link', 'short_description', 'date'], inplace=True)
    df.drop(columns=['category','headline', 'authors', 'link', 'short_description', 'date'], inplace=True)
    return df, df_classes

df = import_json_to_Dataframe(const.path, 'test.json')
df, df_classes = huffingtonpost_data_preprocessing(df)

#####################################################################################
#Split up dataset and feature extraction
#####################################################################################

vectorizer = TfidfVectorizer(stop_words='english')
X = df['Text'].to_numpy()
y = df['target'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

#####################################################################################
#Functions to start the training and produce the model
#####################################################################################
def model_training(text_clf):

    t0 = time()
    text_clf.fit(X_train,y_train)
    train_time = time() - t0

    t0 = time()
    test_pred = text_clf.predict(X_test)
    test_time = time() - t0

    return text_clf, test_pred, train_time, test_time

def train_model2():
    model2 = model_training(LinearSVC(loss='squared_hinge',
                                      penalty='l2', dual=False,
                                      tol=1e-3))
    return model2

model2_results = train_model2()
model2 = model2_results[0]
test_pred = model2_results[1]
train_time = model2_results[2]
test_time = model2_results[3]
del model2_results

#####################################################################################
#Functions to print and write logs
#####################################################################################

model_name = 'Linear SVC'

def write_model_classification(text_clf, file):
    file.write("Model name:\t%s\n" % model_name)
    file.write("\n")

    file.write("Model specification:\n")
    file.write(str(text_clf))
    file.write("\n")

    file.write("Train time: %0.3fs\n" % train_time)
    file.write("Test time:  %0.3fs\n" % test_time)
    file.write("\n")

    file.write("Classification Report:\n")
    file.write(metrics.classification_report(y_test, test_pred,
                                             target_names=df_classes['class'].to_numpy()))
    file.write("\n")

def write_top_10_keywords(text_clf, vectorizer, categories, file):
    feature_names = vectorizer.get_feature_names()
    feature_names = np.asarray(feature_names)
    file.write("Top 10 keyword per class:\n")
    for i, category in enumerate(categories):
        top10 = np.argsort(text_clf.coef_[i])[-10:]
        file.write("%s:\t\t %s\n" % (category, " ".join(feature_names[top10])))

def write_to_file(path, filename_model2_logs):
    file = open((path + '\\' + filename_model2_logs), 'w')
    write_model_classification(model2, file)
    write_top_10_keywords(model2,vectorizer, df_classes['class'].to_numpy(), file)
    file.close()



def main():
    write_to_file(const.path, 'test.txt')
    export_Dataframe_to_csv(df, const.path, 'huffingtonpost_dataset.csv')


main()