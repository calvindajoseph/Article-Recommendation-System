# project aim is to build a machine learning model that can extract the human emotions from text by using sentiment140.
# source: Scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer #convert a collection of raw documents to a matrix of TF-IDF features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

cols = ['sentiment','id','date','query_string','user','text']
dataset = pd.read_csv(r"C:\Users\Hamou\OneDrive\المستندات\Term 2 2019 uc\Engineering Project A - 10004\sentiment 140\training.1600000.processed.noemoticon.csv", encoding = "ISO-8859-1", header=None, names=cols)
# above line will be different depending on where you saved your data, and your file name


dataset.drop(['id', 'date', 'query_string', 'user'], axis=1, #sentiment: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive). the text of the tweet (Lyx is cool).
        inplace=True)  # inplace=True returns None inplace=False returns a copy of the object with the operation performed
print(dataset[dataset.sentiment == 0].head(
    10)) #50% of the data is with negative label, and another 50% with positive label.
print(len(dataset))
# 1600000 rows and 2 columns
print(dataset.shape)
print(dataset.sentiment.value_counts())

# texts are the data and sentiments are the result which we have to predict by looking at the message
X=dataset["text"]
y=dataset["sentiment"]



# # With Tfidfvectorizer you compute the word counts, idf and tf-idf values all at once. It’s really simple. it counts number of words in the document.
# # settings that you use for count vectorizer will go here
# tfidf_vectorizer = TfidfVectorizer(use_idf=True)
# # just send in all your docs here
# tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(dataset)
# # get the first vector out (for the first document)
# first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]
# # place tf-idf values in a pandas data frame
# dataset = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
# dataset.sort_values(by=["tfidf"],ascending=False)
# print(dataset)

tfidf_vectorizer = TfidfVectorizer(min_df=1,stop_words='english', use_idf=True) #stop words are the useless words like are, you etc
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(dataset)


# pre-processing
#X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

print(X_train.head())

# x_train_cv = cv.fit_transform(dataset)
# print(x_train_cv.toarray())



models = (KNeighborsClassifier(n_neighbors=6, p=2, metric='minkowski'),
          KNeighborsClassifier(n_neighbors=30, p=2, metric='minkowski'),
          KNeighborsClassifier(n_neighbors=4, p=2, metric='minkowski'),
          KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski'),
          KNeighborsClassifier(n_neighbors=2, p=2, metric='minkowski'),
          KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski'))

# title for the plots
titles = ('KNN 6',
          'KNN 30',
          'KNN 4',
          'KNN 3',
          'KNN 2',
          'KNN 1')

models = (clf.fit(X_train, y_train) for clf in models)  # sklearn loop over the models
# now we are ready for predictions
#y_pred = clf.predict(X_test)
#print('Misclassfied samples: %d' % (
            #y_test != y_pred).sum())  # refers to the number of individual that we know that below a category that are classified by the method in a different category.
#print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
