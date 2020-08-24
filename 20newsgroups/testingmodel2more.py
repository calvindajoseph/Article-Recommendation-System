# Khanh testing code


from time import time
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn import metrics


vectorizer = TfidfVectorizer()


#===================================================================

class Doc_test():

    test1_no=1
    test_docs_new = ['Jesus is Lord!',
                 'Free RAM! Download now!',
                 'You are such a loser!',
                 'I am very sad. My car broke down :(']

    test2_no=2
    test_docs222_new = ['Computer is good!',
                        'I have a car',
                        'I often go to church ',
                         'I like a new motocycle and a helmet',
                         'I buy a new ball']


#======================================================
D= Doc_test()

#======================================================

newsgroups_train =  fetch_20newsgroups(data_home=None, subset= 'train', 
                                           categories=None,
                                           shuffle=True, random_state=42, 
                                           remove = ('headers'),
                                           download_if_missing=True)
newsgroups_test =  fetch_20newsgroups(data_home=None, subset= 'test', 
                                           categories=None, 
                                           shuffle=True, random_state=42, 
                                           remove = ( 'headers'),
                                           download_if_missing=True)



#==================================================
 
class Xulydata():    
 vectors_train = vectorizer.fit_transform(newsgroups_train.data)
  

 X_test = vectorizer.transform(newsgroups_test.data)
 
 
#==========================================================
X=Xulydata()
vectors_train=X.vectors_train  
X_test=X.X_test  

   
def Print_header(model_name):
     print(''*5)
    
     print("                     ",model_name)
    
     print(''*5)

   
#==========================================     
     
def custom_test_vectorizer(doc):
    vectors = vectorizer.transform(doc)
    return vectors

def custom_test(no,clf,doc):
    predicted = clf.predict(custom_test_vectorizer(doc))

    print('Test %s:' % no)
    for doc, category in zip(D.test_docs_new, predicted):
        print('%r => %s' % (doc,newsgroups_train.target_names[category]))
    #return predicted

def custom_test_vectorizer_1(doc1):
    vectors = vectorizer.transform(doc1)
    return vectors

def custom_test_1(no,clf,doc1):
    predicted = clf.predict(custom_test_vectorizer(doc1))

    print('Test %s:' % no)
    for doc, category in zip(D.test_docs222_new, predicted):
        print('%r => %s' % (doc,newsgroups_train.target_names[category]))
    return predicted
    return 0

#============================================



def show_top10(classifier, vectorizer, categories):
    feature_names = vectorizer.get_feature_names()
    feature_names = np.asarray(feature_names)
    print("Top 10 keyword per class:")
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s:\t\t %s" % (category, " ".join(feature_names[top10])))




def Processing(model_clf,text_clf):
      Print_header(model_clf)

      t0 = time()
      text_clf.fit(vectors_train,newsgroups_train.target)
      t1 = time() 
      train_time= t1-t0
      print("Train time: %0.3fs" % train_time)


      
      t0 = time()
      test_pred = text_clf.predict(X_test)
      t1 = time()
      test_time = t1- t0
      print("Test time:  %0.3fs" % test_time)
      print('')

      print("Classification Report:")
      print("*************************************************\n", metrics.classification_report(newsgroups_test.target, test_pred,
                         target_names=newsgroups_test.target_names),"*************************************************\n") 




#===================================================================
      
class runNB() :  
  model_clf='MultinomialNB'
  text_clf=MultinomialNB(alpha=1)
  
  print("Model specification:")
  print(text_clf)
  Processing(model_clf, text_clf)

  print("*************************************************")
  show_top10(text_clf,vectorizer,newsgroups_train.target_names)
  text_clf.score(vectors_train,newsgroups_train.target)
  print("*************************************************")
  
  custom_test(D.test1_no,text_clf,D.test_docs_new)
  custom_test_1(D.test2_no,text_clf,D.test_docs222_new)

#========================================================
class runSCV() :  
 
  model_clf='LinearSVC'
  text_clf=LinearSVC(loss='squared_hinge',
                          penalty='l2', dual=False,
                          tol=1e-3)    
  Processing(model_clf, text_clf) 
  print("Model specification:")
  print(text_clf)
  Processing(model_clf, text_clf)
  test_pred = text_clf.predict(X_test)
  
  
  print("********         "*5)
  show_top10(text_clf,vectorizer,newsgroups_train.target_names)
  print("=========        "*5)
  
  custom_test(D.test1_no,text_clf,D.test_docs_new)
  custom_test_1(D.test2_no,text_clf,D.test_docs222_new)
  

 #========================================================
class runSGD() :    
 
 model_clf='SGDClassifier'
 text_clf=SGDClassifier(loss='hinge',
                              alpha=.0001, max_iter=50,
                              tol=None,penalty='l2')
 Processing(model_clf, text_clf)   

 print("Model specification:")
 print(text_clf)
 Processing(model_clf, text_clf)
 test_pred = text_clf.predict(X_test)
 
  
 print("********         "*5)
 show_top10(text_clf,vectorizer,newsgroups_train.target_names)
 print("=========        "*5)
  
 custom_test(D.test1_no,text_clf,D.test_docs_new)
 custom_test_1(D.test2_no,text_clf,D.test_docs222_new)

 