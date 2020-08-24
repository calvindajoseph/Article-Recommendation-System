import pandas as pd #we use pandas because there are built-in function in python wich overlap with the pandas methods and call pandas methods using the pd prefix.
from sklearn.feature_extraction.text import TfidfTransformer #converts a colleciton of raw documents to a matrix of TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer #count number of words(term frequency), limit vocabulary size, apply stro words aand etc.

# this is a very toy example, do not try this at home unless you want to understand the usage differences
docs = ["the house had a tiny little mouse",
        "the cat saw the mouse",
        "the mouse ran away from the house",
        "the cat finally ate the mouse",
        "the end of the mouse story"
        ]
#instantiate CountVectorizer()
cv = CountVectorizer()

# this steps generates word counts for the words in your docs
word_count_vector = cv.fit_transform(docs)

# Now, let’s check the shape. We should have 5 rows (5 docs) and 16 columns (16 unique words, minus single character words):
print(word_count_vector)
exit

# If smooth_idf=True (the default), the constant “1” is added to the numerator and denominator of the idf as if an extra
# document was seen containing every term in the collection exactly once,
# which prevents zero divisions: idf(d, t) = log [ (1 + n) / (1 + df(d, t)) ] + 1.
# IDF) vector; only defined if use_idf is True.
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
# print idf values
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["tf_idf_weights"])
# sort ascending
print(df_idf.sort_values(by=['tf_idf_weights']))

# count matrix
count_vector=cv.transform(docs)
# tf-idf scores
tf_idf_vector=tfidf_transformer.transform(count_vector)
feature_names = cv.get_feature_names()
# get tfidf vector for first document
first_document_vector = tf_idf_vector[0]
# print the scores
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
print(df.sort_values(by=["tfidf"], ascending=False))

# With Tfidfvectorizer you compute the word counts, idf and tfidf values all at once.
# settings that you use for count vectorizer will go here
tfidf_vectorizer=TfidfVectorizer(use_idf=True)

# just send in all your docs here
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(docs)

# get the first vector out (for the first document)
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]

df2 = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
df2.sort_values(by=["tfidf"],ascending=False)