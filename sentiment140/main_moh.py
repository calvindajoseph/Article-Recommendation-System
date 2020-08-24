'''
 Tutorial source https://mathpn.github.io/sentiment-classification-twitter-part1/
'''

import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Define the models
models = (LogisticRegression(C = 1, penalty = 'l1', solver = 'liblinear',\
                multi_class = 'ovr', random_state = 42),)
models_name = ["LogisticRegression"]

# Load the data
print("Loading data...")
start_time = time.time()

data = pd.read_csv(r"C:\Users\Hamou\OneDrive\المستندات\Term 2 2019 uc\Engineering Project A - 10004\sentiment 140\training.1600000.processed.noemoticon.csv",
                   encoding = 'ISO-8859-1', header = None)

# data = data[0:800000]
print("The data shape is {}".format(data.shape))
data.columns = ['sentiment','id','date','flag','user','tweet']

print("Loading data has completed in {}s!\n".format(time.time() - start_time))



def preprocess_tweets(tweet):
    return tweet
    
    #Detect ALLCAPS words
    tweet = re.sub(r"([A-Z]+\s?[A-Z]+[^a-z0-9\W]\b)", r"\1 <ALLCAPS> ", tweet)
    #Remove URLs
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','<URL> ', tweet)
    #Separate words that are joined by / (e.g. black/brown)
    tweet = re.sub(r"/"," / ", tweet)
    #Remove user mentions
    tweet = re.sub('@[^\s]+', "<USER>", tweet)
    #Remove all special symbols
    tweet = re.sub('[^A-Za-z0-9<>/.!,?\s]+', '', tweet)
    #Detect puncutation repetition
    tweet = re.sub('(([!])\\2+)', '! <REPEAT> ', tweet)
    tweet = re.sub('(([?])\\2+)', '? <REPEAT> ', tweet)
    tweet = re.sub('(([.])\\2+)', '. <REPEAT> ', tweet)
    #Remove hashtags
    tweet = re.sub(r'#([^\s]+)', r'<HASHTAG> \1', tweet)
    #Detect word elongation (e.g. heyyyyyy)
    tweet = re.sub(r'(.)\1{2,}\b', r'\1 <ELONG> ', tweet)
    tweet = re.sub(r'(.)\1{2,}', r'\1)', tweet)
    #Expand english contractions
    tweet = re.sub(r"'ll", " will", tweet)
    tweet = re.sub(r"'s", " is", tweet)
    tweet = re.sub(r"'d", " d", tweet) # Would/Had ambiguity
    tweet = re.sub(r"'re", " are", tweet)
    tweet = re.sub(r"didn't", "did not", tweet)
    tweet = re.sub(r"couldn't", "could not", tweet)
    tweet = re.sub(r"can't", "cannot", tweet)
    tweet = re.sub(r"doesn't", "does not", tweet)
    tweet = re.sub(r"don't", "do not", tweet)
    tweet = re.sub(r"hasn't", "has not", tweet)
    tweet = re.sub(r"'ve", " have", tweet)
    tweet = re.sub(r"shouldn't", "should not", tweet)
    tweet = re.sub(r"wasn't", "was not", tweet)
    tweet = re.sub(r"weren't", "were not", tweet)
    #Remove extra spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Lower case
    tweet = tweet.lower()

    return tweet



print("Apply train_test_split...")
start_time = time.time()
train_data, test_data = train_test_split(data, train_size = 0.8, random_state = 42)
print("train_test_split has completed in {}s!\n".format(time.time() - start_time))


print("Apply preprocess_tweets...")
start_time = time.time()

sentiment = np.array(data['sentiment'])
tweets = np.array(data['tweet'].apply(preprocess_tweets))

sentiment_train = np.array(train_data['sentiment'])
tweets_train = np.array(train_data['tweet'].apply(preprocess_tweets))

sentiment_test = np.array(test_data['sentiment'])
tweets_test = np.array(test_data['tweet'].apply(preprocess_tweets))
print("preprocess_tweets has completed in {}s!\n".format(time.time() - start_time))



print("Do feature extraction...")
start_time = time.time()
vectorizer = TfidfVectorizer(min_df = 75)
vectorizer.fit(tweets)
tweets_bow_train = vectorizer.transform(tweets_train)
tweets_bow_test = vectorizer.transform(tweets_test)
print("Feature extraction has been completed in {}s!\n".format(time.time() - start_time))


print("Create models...")
for cls_name, clf in zip(models_name, models):
    print("Training {} ...".format(cls_name))
    start_time = time.time()
    clf.fit(tweets_bow_train, sentiment_train)
    print("Training models has been completed in {}s!\n".format(time.time() - start_time))
    
    print("Testing code...")
    start_time = time.time()
    pred1 = clf.predict(tweets_bow_test)
    pos_prob1 = clf.predict_proba(tweets_bow_test)[:, 1]
    auc1 = roc_auc_score(sentiment_test, pos_prob1)
    f11 = f1_score(sentiment_test, pred1, pos_label=4)
    print("Model 1: AUC {} F1 {}".format(auc1, f11))
    print("Testing code has completed in {}s!\n".format(time.time() - start_time))



