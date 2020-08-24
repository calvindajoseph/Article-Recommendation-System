# project aim is to build a sentiment analysis model with given training dataset, so that it can be applied to tweets gathered through twitter api.
# source: https://towardsdatascience.com/another-twitter-sentiment-analysis-bb5b01ebad90
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from bs4 import BeautifulSoup
import re
from nltk.tokenize import WordPunctTokenizer
#plt.style.use('fivethirtyeight')

cols = ['sentiment','id','date','query_string','user','text']
df = pd.read_csv(r"C:\Users\Hamou\OneDrive\المستندات\Term 2 2019 uc\Engineering Project A - 10004\sentiment 140\training.1600000.processed.noemoticon.csv", encoding = "ISO-8859-1", header=None, names=cols)
# above line will be different depending on where you saved your data, and your file name
 #print(df.head())

#df = pd.read_csv(f, header=None, names=cols)
print(df.head())
print(df.info())
# Dataset has 1.6million entries, with no null entries. even though the dataset description mentioned neutral class, the training set has no neutral class. 50% of the data is with negative label, and another 50% with positive label.
print(df.sentiment.value_counts())







df.drop(['id', 'date', 'query_string', 'user'], axis=1,
        inplace=True)  # inplace=True returns None inplace=False returns a copy of the object with the operation performed
print(df[df.sentiment == 0].head(
    10))  # (0 = negative, 2 = neutral, 4 = positive). 50% of the data is with negative label, and another 50% with positive label.


print(df[df.sentiment == 4].head(10))  # By looking at some entries for each class,
print(df[df.sentiment == 0].index)
print(df[df.sentiment == 4].index)
df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
print(df.sentiment.value_counts())
# it seems like that all the negative class is from 0~799999th index, and the positive class entries start from 800000 to the end of the dataset.

df['pre_clean_len'] = [len(t) for t in df.text]  # The len() function returns the number of items in an object.
print(df)

# Data dictionary
data_dict = {
    'sentiment': {
        'type': df.sentiment.dtype,
        'description': 'sentiment class - 0:negative, 1:positive'
    },
    'text': {
        'type': df.text.dtype,
        'description': 'tweet text'
    },
    'pre_clean_len': {
        'type': df.pre_clean_len.dtype,
        'description': 'Length of the tweet before cleaning'
    },
    'dataset_shape': df.shape
}
pprint(data_dict)

fig, ax = plt.subplots(figsize=(5, 5))
plt.boxplot(df.pre_clean_len)
plt.show()  # to see overall distribution of length of strings in each entry
print(df[df.pre_clean_len > 140].head(10))

# Data Preparation HTML decoding
print(df.text[
          279])  # looks like HTML coding has not been converted to text. so we will decode HTML to general text and BeautifulSoup will be used for this.
# BeautifulSoup is a Python library for pulling data out of HTML and XML files.
example1 = BeautifulSoup(df.text[279], 'lxml')  # lxml handles XML and HTML files
print(example1.get_text())



# Data preparation @mention
print(df.text[343])  # Even though @mention carries a certain information (which another user that the tweet mentioned),
# this information doesn't add value to build sentiment analysis model.
print(re.sub(r'@[A-Za-z0-9]+', '', df.text[
    343]))  # A-Za-z0-9 means it can be among the all the Uppercase and lowercase letters and the number betwween 0 and 9, and the letter



# Data preparation URL links
print(df.text[0])
print(re.sub('https?://[A-Za-z0-9./]+', '', df.text[0]))


# Data preparation UTF-8 BOM (Byte Order Mark)
print(df.text[226])  # strange patterns of chrs \ï¿½
testing = df.text[226]
#testing = df.text[226]
#print(testing)

print(testing.replace(u"ï¿½", "?"))


# data preparation hashtag/numbes
print(df.text[175])  # remove the '#'
print(re.sub("[^a-zA-Z]", " ", df.text[175]))  # ^ means the Start of a string

tok = WordPunctTokenizer()

pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.replace(u"ï¿½", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()

testing = df.text[:100]
test_result = []
for t in testing:
    test_result.append(tweet_cleaner(t))
print(test_result)

nums = [0, 400000, 800000, 1200000, 1600000]
print("Cleaning and parsing the tweets...\n")
clean_tweet_texts = []
for i in range(nums[0], nums[1]):
    if ((i + 1) % 10000 == 0):
        print("Tweets %d of %d has been processed" % (i + 1, nums[1]))
        clean_tweet_texts.append(tweet_cleaner(df['text'][i]))
        print(len(clean_tweet_texts))

print("Cleaning and parsing the tweets...\n")
for i in range(nums[1], nums[2]):
    if ((i + 1) % 10000 == 0):
        print("Tweets %d of %d has been processed" % (i + 1, nums[2]))
        clean_tweet_texts.append(tweet_cleaner(df['text'][i]))
        print(len(clean_tweet_texts))

print("Cleaning and parsing the tweets...\n")
for i in range(nums[2], nums[3]):
    if ((i + 1) % 10000 == 0):
        print("Tweets %d of %d has been processed" % (i + 1, nums[3]))
        clean_tweet_texts.append(tweet_cleaner(df['text'][i]))
        print(len(clean_tweet_texts))

print("Cleaning and parsing the tweets...\n")
for i in range(nums[3], nums[4]):
    if ((i + 1) % 10000 == 0):
        print("Tweets %d of %d has been processed" % (i + 1, nums[4]))
        clean_tweet_texts.append(tweet_cleaner(df['text'][i]))
        print(len(clean_tweet_texts))

    # Saving cleaned data as csv
    clean_df = pd.DataFrame(clean_tweet_texts, columns=['text'])
    clean_df['target'] = df.sentiment
    #print(clean_df.head())

    # # clean_df.to_csv('clean_tweet.csv')
    csv = 'clean_tweet.csv'
    my_df = pd.read_csv(csv, index_col=0)
    print(my_df.head())
    print(my_df.info())

