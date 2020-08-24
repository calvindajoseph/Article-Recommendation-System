'''
scikit-learn provides utilities for the most common ways to extract numerical features from text content, namely:
- tokenizing: it is the process of breaking a stream of text up into words, phrases, symbols, or other meaningful meanings
 elements called tokens. it converts a string like "My favorite color is blue" to a list of array like ["My", "favorite", "color", "is", "blue"] by using split() function.
- counting: method counts how many times an element has occurred in a list and returns it.
- normalizing: it works on the rows not columns
Source: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
'''
from sklearn.feature_extraction.text import CountVectorizer #CountVectorizer implements both tokenization and occurrence counting in a single class
#CountVectorizer under feature_extraction converts strings (or tokens) into numeical feature suiable for scikit-learn's meachine learning algorithms
# The sklearn.feature_extraction.text submodule gathers utilities to build feature vectors from text documents.
# feature_extraction.text.CountVectorizer Convert a collection of text documents to a matrix of token counts.
from sklearn.feature_extraction.text import TfidfTransformer #Transform a count matrix to a normalized tf or tf-idf representation

vectorizer = CountVectorizer()
# print(vectorizer)

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X = vectorizer.fit_transform(corpus)# .fit_transform method applies to feature extraction objects. the fit part applies
# to the feature extracture itself: it determines what features it will base future transformations on. The transform part
# is what it takes the data and return some transformed data back at you.
print(X) #will plot non-zero values in a sparse matrix which is 4x9 sparse matrix with stored 19 elements in compressed sparse
# compressed sparse often used to represent sparse matrices in machine learning given the efficient access and matrix multiplication that it supports.
#The default configuration tokenizes the string by extracting words of at least 2 letters.
# The specific function that does this step can be requested explicitly:
analyze = vectorizer.build_analyzer()
print(analyze("This is a text document to analyze.") == (
    ['this', 'is', 'text', 'document', 'to', 'analyze']))
# Each term found by the analyzer during the fit is assigned a unique integer index corresponding to a column in the resulting matrix.
# This interpretation of the columns can be retrieved as follows:
print(vectorizer.get_feature_names() == (
    ['and', 'document', 'first', 'is', 'one',
     'second', 'the', 'third', 'this']))
print(X.toarray())# toarray() is a method used to get an array which contains all the elements in ArrayList object in proper sequence (from the first to the last element)
print(vectorizer.vocabulary_.get('and'))#the converse mapping from feature name to column index is stored in the vocabulary_ attributre of the vectorizer
print(vectorizer.transform(['Something completely new.']).toarray())#Hence words that were not seen in the training corpus will be completely ignored in future calls to the transform method
# ngram_range is jsut a string of n words in a row. e.g. the sentence 'i am groot' contains the 2-grams 'i am' and'am groot'. the sentence itself a 3-gram. set the parameter ngram_range=(a,b)
#where a is the min and b is the max size of ngrams you want to include in your features.
#toekn-pattern regular expression identifying tokens-by default words that consist of a single character (e.g., 'a', '2') are ignored, setting token-pattern to '\b\w+\b' will include these tokens
#\b represents a word boundary between a word character and a non-word character. \w represents a word character
#min_df (default 1) remove terms from the vocabulary that occur in fewer than min_df documents (in a lrage corpus this may be set to 15 or higher to eliminate very rare words)
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                    token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
print(analyze('Bi-grams are cool!') == (
    ['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool']))
#The vocabulary extracted by this vectorizer is hence much bigger and can now resolve ambiguities encoded in local positioning patterns:
X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
print(X_2)
# In particular the interrogative form “Is this” is only present in the last document:
feature_index = bigram_vectorizer.vocabulary_.get('is this')
print(X_2[:, feature_index])

transformer = TfidfTransformer(smooth_idf=False)
print(transformer)

counts = [[3, 0, 1],
          [2, 0, 0],
          [3, 0, 0],
          [4, 0, 0],
          [3, 2, 0],
          [3, 0, 2]]

tfidf = transformer.fit_transform(counts)
print(tfidf)

print(tfidf.toarray())

transformer = TfidfTransformer()
print(transformer.fit_transform(counts).toarray())


