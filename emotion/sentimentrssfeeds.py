'''
download the dataset from here: http://help.sentiment140.com/for-students
'''
# Import libraries
import re # Regular Expression
import pandas as pd # Present data that is suitable for datta analysis via its series and dataframe data structures.
import numpy as np # It provides a high-performance multidimensional array object
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import feedparser

final_model = (LogisticRegression(C = 1, penalty = 'l1', solver = 'liblinear',\
                multi_class = 'ovr', random_state = 42),)
model_name = ["LogisticRegression"]

# Load the data
print("Loading data...")
start_time = time.time() # This function is used to count the number of seconds elapsed since the epoch.

data = pd.read_csv(r"C:\Users\Hamou\OneDrive\المستندات\Term 2 2019 uc\Engineering Project A - 10004\sentiment 140\training.1600000.processed.noemoticon.csv", #download abovementioned 140 sentiment from website and add the path here between ""
                   encoding='ISO-8859-1', header=None)
line_length = 120
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

print("The train_data shape after splitting is {}".format(train_data.shape))
print("The test_data shape after splitting is {}".format(test_data.shape))

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


print("Create models...")
for cls_name, clf, in zip(model_name, final_model):
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

# ------------------Check accuracy of news articles with rss feeds------------------
print_line()

title_list = []
desc_list = []
link_list = []

category_list = [None] * 20

title_desc_list = []
n = 10

def filter_and_store(predicted):
    for idx, i in enumerate(predicted):
        if category_list[i] == None:
            category_list[i] = idx
    return 0

def predict_with_model_1(clf, text):
    predicted = clf.predict(vectorizer.transform(text))
    return predicted

def show_filtered_results():
    for i in range(20):
        print_line()
        print(i)
        print(tweets_train.target_names[i])
        if category_list[i] == None:
            print('No article yet.')
        else:
            print('Title:      \t%r' % (title_list[category_list[i]]))
            print('Description:\t%r' % (desc_list[category_list[i]]))
            print('URL:        \t%r' % (link_list[category_list[i]]))
    return 0

def rss_feed(feed,n):
    for i in range(n):
        title_list.append(feed.entries[i].title)
        desc_list.append(feed.entries[i].description)
        link_list.append(feed.entries[i].link)
        title_desc_list.append(feed.entries[i].title + ". " + feed.entries[i].description)
    return 0

feed = feedparser.parse('http://www.abc.net.au/news/feed/2942460/rss.xml')
rss_feed(feed,25)

predicted = predict_with_model_1(final_model,title_desc_list)

filter_and_store(predicted)

show_filtered_results()
