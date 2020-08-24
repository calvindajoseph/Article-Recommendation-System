#in this exercise we will be working on 4 categories out of the 20 in order to get faster execution
#now we can load the dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42);
print(twenty_train.target_names)

#the files are loaded in memory in the data attribute
print(len(twenty_train.data))
print(len(twenty_train.filenames))

#let's print the first lines of the first loaded file:
print ("\n".join(twenty_train.data[0].split("\n")[:2]))
print(twenty_train.target_names[twenty_train.target[0]])

#in order to perform ML on text documents, we first need to trun the text content into numerical feature vectors.
count_vect = CountVectorizer(stop_words='english')
X_train_counts = count_vect.fit_transform(twenty_train.data)
#both tf and tf-idf can be computed as follows
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)


#now we will train a classifier. We will start with Naive Bayes classifier
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

docs_new = ['God is love', 'Graphics Library on the GPU is fast', 'the coronavirus vaccine will be out soon']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

#pipeline will behave like a compound classifier
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

#here we will train the model with a single command
text_clf.fit(twenty_train.data, twenty_train.target)

#evaluation of the performance on the test set
twenty_test = fetch_20newsgroups(
    subset='test', categories=categories,
    shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print(" Naïve Bayes: ",np.mean(predicted == twenty_test.target))
print(metrics.classification_report(
    twenty_test.target, predicted,
    target_names=twenty_test.target_names)
)
print("----------------------------")
#now we will try svm, which is widely used in text classification algorithms
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, n_iter_no_change=5)),
])
text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
print(" SVM: ",np.mean(predicted == twenty_test.target))

# we will provide more utilities for more detailed performance:
print(metrics.classification_report(
    twenty_test.target, predicted,
    target_names=twenty_test.target_names)
)
