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

#####################################################################################
#Initial variables
#####################################################################################

vectorizer = TfidfVectorizer(stop_words='english')
shuffle = True
remove = ['footers','quotes']

const = Global()

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

#####################################################################################
#Pre-process dataset
#####################################################################################

def huffingtonpost_data_preprocessing(df):

    df.drop(columns=['ArticleId'], inplace=True)
    df.dropna(subset=['Text'], inplace=True)
    df.dropna(subset=['Category'], inplace=True)

    df['target'] = pd.factorize(df['Category'])[0] + 20

    df['Category'] = [s.capitalize() for s in df['Category']]
    df['Category'] = df['Category'].replace({"Politics": "talk.politics.misc"})
    df['target'] = df['target'].replace({
        22: 18,
        23: 22
    })

    df_classes = pd.DataFrame(data={'no': df.target.unique(), 'class_category': df.Category.unique()})
    df_classes = df_classes[df_classes.class_category != "talk.politics.misc"]
    df.drop(columns=['Category'], inplace=True)
    export_Dataframe_to_csv(df_classes, const.path, 'bbc_dataset_class.csv')
    return df, df_classes

df = import_csv_to_Dataframe(const.path, 'BBC_News.csv')
df,df_classes = huffingtonpost_data_preprocessing(df)

#####################################################################################
#Set up the datasets and feature extraction
#####################################################################################

newsgroup_train = fetch_20newsgroups(subset='train',
                                     shuffle=shuffle,
                                     remove=remove)
newsgroup_test = fetch_20newsgroups(subset='test',
                                    shuffle=shuffle,
                                    remove=remove)

X = df['Text'].to_numpy()
y = df['target'].to_numpy()

X_train_huffington, X_test_huffington, y_train_huffington, y_test_huffington = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y)



X_train_text = newsgroup_train.data
X_train_text = X_train_text + (X_train_huffington.tolist())

y_train_text = newsgroup_train.target
y_train_text = np.concatenate((y_train_text, y_train_huffington))

X_test_text = newsgroup_test.data + X_test_huffington.tolist()

X_train = vectorizer.fit_transform(X_train_text)
y_train = y_train_text
X_test = vectorizer.transform(X_test_text)

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

    y_test_text = np.concatenate((newsgroup_test.target, y_test_huffington))

    file.write("Classification Report:\n")
    file.write(metrics.classification_report(y_test_text, test_pred,
                                             target_names=newsgroup_test.target_names + df_classes['class_category'].tolist()))
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
    write_top_10_keywords(model2,vectorizer, newsgroup_test.target_names + df_classes['class_category'].tolist(), file)
    file.close()

#####################################################################################
# Store the model and feature extraction
#####################################################################################

def store_clf(clf, path, filename):
    path = path + '\\' + filename
    dump(clf, path)
    #print('Stored %r to %r' % (filename, path))

def store_vectorizer(vectorizer, path, filename):
    path = path + '\\' + filename
    dump(vectorizer, path)

def store_model2_and_vectorizer(clf, vectorizer, path, filename_model2, filename_vectorizer_model2):
    store_clf(clf, path, filename_model2)
    store_vectorizer(vectorizer, path, filename_vectorizer_model2)

#####################################################################################
# Create and store categories csv
#####################################################################################

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

def process_Dataframe(df, path, filename):
    df = set_recommendation_Dataframe_model2(df)
    export_Dataframe_to_csv(df, path, filename)

#####################################################################################
# Run Entire Code
#####################################################################################

def main():
    write_to_file(const.path,'model2_log_bbc.txt')
    store_model2_and_vectorizer(model2,vectorizer,const.path,const.filename_model2_bbc,const.filename_vectorizer_model2_bbc)
    process_Dataframe(df_recommendation_article_by_model2, const.path, 'bbc_dataset.csv')

main()