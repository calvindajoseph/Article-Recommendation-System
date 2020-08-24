'''
# project aim is to build a machine learning model that can extract the human emotions from text by using sentiment140.
 Tutorial source https://mathpn.github.io/sentiment-classification-twitter-part1/
'''
# Import libraries
import re # Regular Expression
import pandas as pd # Present data that is suitable for datta analysis via its series and dataframe data structures. it has variety of utilities to perform I/O operations in a seamless manner
import numpy as np # It provides a high-performance multidimensional array object
from matplotlib import pyplot as plt
import time
# import texttable as tt
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer # Converts a collection of raw documents to a matrix of TF-IDF features
# import plotly.graph_objects as go
import itertools
from sklearn.decomposition import TruncatedSVD


def make_meshgrid(x, y, h=0.5):
    """Create a mesh of points to plot in

    Parameters

    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional
    Returns

    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy



def plot_contours(ax, clf, xx, yy, **params):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out



# Define the models
models = (
          RandomForestClassifier(max_depth=5, n_estimators=10, max_features="auto"),
          DecisionTreeClassifier(max_depth=5),
          LogisticRegression(C = 1, penalty = 'l1', solver = 'liblinear',\
                multi_class = 'ovr', random_state = 42),)
models_name = ["RandomForestClassifier", "DecisionTreeClassifier", "LogisticRegression"]

# Load the data
print("Loading data...")
start_time = time.time() # This function is used to count the number of seconds elapsed since the epoch.

data = pd.read_csv(r"C:\Users\Hamou\OneDrive\المستندات\Term 2 2019 uc\Engineering Project A - 10004\sentiment 140\training.1600000.processed.noemoticon.csv",
                   encoding='ISO-8859-1', header=None)

# data = data[0:800000]
print("The data shape is {}".format(data.shape))
data.columns = ['sentiment', 'id', 'date', 'flag', 'user', 'tweet']

print("Loading data has completed in {}s!\n".format(time.time() - start_time))

# Directly influences the model’s performance
def preprocess_tweets(tweet):
    return tweet

    # Detect ALLCAPS words
    tweet = re.sub(r"([A-Z]+\s?[A-Z]+[^a-z0-9\W]\b)", r"\1 <ALLCAPS> ", tweet)
    # Remove URLs
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '<URL> ', tweet)
    # Separate words that are joined by / (e.g. black/brown)
    tweet = re.sub(r"/", " / ", tweet)
    # Remove user mentions
    tweet = re.sub('@[^\s]+', "<USER>", tweet)
    # Remove all special symbols
    tweet = re.sub('[^A-Za-z0-9<>/.!,?\s]+', '', tweet)
    # Detect puncutation repetition
    tweet = re.sub('(([!])\\2+)', '! <REPEAT> ', tweet)
    tweet = re.sub('(([?])\\2+)', '? <REPEAT> ', tweet)
    tweet = re.sub('(([.])\\2+)', '. <REPEAT> ', tweet)
    # Remove hashtags
    tweet = re.sub(r'#([^\s]+)', r'<HASHTAG> \1', tweet)
    # Detect word elongation (e.g. heyyyyyy)
    tweet = re.sub(r'(.)\1{2,}\b', r'\1 <ELONG> ', tweet)
    tweet = re.sub(r'(.)\1{2,}', r'\1)', tweet)
    # Expand english contractions
    tweet = re.sub(r"'ll", " will", tweet)
    tweet = re.sub(r"'s", " is", tweet)
    tweet = re.sub(r"'d", " d", tweet)  # Would/Had ambiguity
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
    # Remove extra spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # Lower case
    tweet = tweet.lower()

    return tweet


print("Apply train_test_split...")
start_time = time.time()

train_data, test_data = train_test_split(data, train_size=0.8, random_state=42)
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
vectorizer = TfidfVectorizer(min_df=75)
vectorizer.fit(tweets)
tweets_bow_train = vectorizer.transform(tweets_train)
tweets_bow_test = vectorizer.transform(tweets_test)
print("Feature extraction has been completed in {}s!\n".format(time.time() - start_time))


pca = TruncatedSVD(n_components=2, n_iter=1000, random_state=42)
pca.fit(tweets_bow_train)
tweets_bow_train = pca.transform(tweets_bow_train)
tweets_bow_test = pca.transform(tweets_bow_test)

print("Create models...")
fig, sub = plt.subplots(1, 3, figsize=(12, 15))
plt.subplots_adjust(wspace=0.2, hspace=0.4)

print("tweets_bow_train {}".format(tweets_bow_train.shape))
print("tweets_bow_test {}".format(tweets_bow_test.shape))

X0, X1 = tweets_bow_train[:, 0], tweets_bow_train[:, 1]
xx, yy = make_meshgrid(X0, X1)


for cls_name, clf, ax in zip(models_name, models, sub.flatten()):
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
    
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0[::1000], X1[::1000], c=sentiment_train[::1000], cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(cls_name)


plt.show()



