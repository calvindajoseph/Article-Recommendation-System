from Global import Global

import feedparser

import math
import numpy as np
import pandas as pd

import re

import pickle
from joblib import dump, load

#####################################################################################
#Initial variables
#####################################################################################

const = Global()

feed_source = [['https://www.atheists.org/feed/', 'American Atheists Magazine', [0]],  #alt.atheism
               ['http://feeds.windowscentral.com/wmexperts', 'Windows Central', [2, 5]],  #comp.os.ms-windows.misc and comp.windows.x
               ['http://blogs.windows.com/feed/', 'Microsoft Windows Blog', [2, 5]],  # comp.os.ms-windows.misc and comp.windows.x
               ['https://osxdaily.com/category/mac-os-x/feed/', 'Mac OS X Daily', [4]],  #comp.sys.mac.hardware
               ['https://developer.apple.com/news/rss/news.rss', 'Apple', [4]],  #comp.sys.mac.hardware
               ['https://cacm.acm.org/browse-by-subject/hardware.rss', 'Communications of the ACM', [3, 4]],  #hardware
               ['https://feeds.megaphone.fm/vergecast', 'The Verge', [1, 2, 3, 4, 5]],  #comp
               ['http://feeds.feedburner.com/TechCrunch/', 'TechCrunch', [1, 2, 3, 4, 5]],  #comp
               ['http://news.mit.edu/rss/topic/science-technology-and-society', 'MIT News - Science, Technology, and Society', [1, 2, 3, 4, 5]],  #comp
               ['https://australia.businessesforsale.com/australian/search/miscellaneous-construction-businesses-for-sale-in-ontario.rss',
                'BusinessForSale', [6]],  #misc.forsale
               ['https://repository.upenn.edu/miscellaneous_papers/recent.rss',
                'Penn Libraries University of Pennsylvania', [6]],  # misc.forsale
               ['https://www.automotive-iq.com/rss/articles', 'Automotive iQ', [7]],  # rec.autos
               ['https://www.automotiveaddicts.com/feed', 'Automotive Addicts', [7]],  # rec.autos
               ['https://www.goauto.com.au/rss/car-reviews/1.xml', 'GoAuto', [7]],  # rec.autos
               ['http://feeds.feedburner.com/MotorAuthority2', 'Motor Authority', [8]],  #rec.motorcycles
               ['https://www.hotbikeweb.com/rss.xml?loc=footer&lnk=rss', 'Hot Bike Magazine', [8]],  #rec.motorcycles
               ['https://www.mlb.com/tigers/feeds/news/rss.xml', 'Major League Baseball', [9]],  #rec.sport.baseball
               ['https://blogs.fangraphs.com/feed/', 'FanGraphs Baseball', [9]],  #rec.sport.baseball
               ['https://thehockeynews.com/section/news/feed', 'The Hockey News', [10]],  #rec.sport.hockey
               ['http://www.sportingnews.com/us/rss', 'Sporting News', [7, 8, 9, 10]],  #rec.sport
               ['https://www.skysports.com/rss/12040', 'Sky Sports', [7, 8, 9, 10]],  #rec.sport
               ['https://www.sportskeeda.com/feed', 'SportsKeeda', [7, 8, 9, 10]],  #rec.sport
               ['http://feeds.feedburner.com/sciencealert-latestnews', 'ScienceAlert', [11, 12, 13, 14]],  #science
               ['http://rss.sciam.com/ScientificAmerican-Global', 'Scientific American Magazine', [11, 12, 13, 14]],  #science
               ['https://www.nytimes.com/svc/collections/v1/publish/http://www.nytimes.com/topic/subject/religion-and-belief/rss.xml',
                'The New York Times - Religion and Belief', [15]],  #soc.religion.christian
               ['http://rss.nytimes.com/services/xml/rss/nyt/Politics.xml', 'The New York Times - Politics', [16, 17, 18, 19]],  #talk
               ['https://thepoliticalinsider.com/feed/', 'The Political Insider', [16, 17, 18, 19]],  #talk
               ['http://www.dailytelegraph.com.au/feed', 'Daily Telegraph'], #general rss feeds Aus Articles
               ['http://www.sbs.com.au/news/rss/Section/Top+Stories', 'SBS Australia'],  #general rss feeds Aus Articles
               ['https://www.canberratimes.com.au/rss.xml', 'The Canberra Times'],  #general rss feeds Aus Articles
               ['http://www.abc.net.au/news/feed/2942460/rss.xml', 'ABC Australia']]  #general rss feeds Aus Articles

no_of_articles_per_source = 40

df_articles = pd.DataFrame(columns=['Title',
                                    'Description',
                                    'Author',
                                    'Category',
                                    'Source',
                                    'Link',
                                    'Class_Model1',
                                    'Class_Model2'])

df_recommendation = pd.DataFrame(columns=['Class_Model1',
                                          'Class_Model2',
                                          'Article_ID'])

#####################################################################################
#Import and export functions
#####################################################################################

def load_model(path, filename):
    path = path + '\\' + filename
    clf = load(path)
    #print('Loaded %r from %r' % (filename, path))
    return clf

def load_feature_extraction(path, filename):
    path = path + '\\' + filename
    vectorizer = load(path)
    return vectorizer

def export_Dataframe_to_csv(df, path, filename):
    path = path + '\\' + filename
    df.to_csv(path, index=False)

    print('Dataframe exported to ' + path)
    return 0

def import_csv_to_Dataframe(path, filename):
    path = path + '\\' + filename
    df = pd.read_csv(path)
    print('Dataframe imported from ' + path)
    return df

def import_model2_categories(path,filename):
    df = pd.DataFrame
    df = import_csv_to_Dataframe(path, filename)
    df.drop([0,1])
    return df

#####################################################################################
#RSS Feed Functions
#####################################################################################

def process_text(text):
    text = re.sub('<[^<]+?>', '', text)
    text = re.sub(r"[^a-zA-Z0-9]+", ' ', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '<URL> ', text)

    return text


def rss_feed_to_dataframe(feed, df, no_of_article, source):
    count = 0
    for i in range(no_of_article):
        article_tuple = [None] * 8

        try:
            article_tuple[0] = process_text(feed.entries[i].title)
        except AttributeError as error:
            article_tuple[0] = None
        except IndexError as error:
            break

        try:
            article_tuple[1] = process_text(feed.entries[i].description)
        except AttributeError as error:
            article_tuple[1] = None

        try:
            article_tuple[2] = feed.entries[i].author
        except AttributeError as error:
            article_tuple[2] = None

        try:
            article_tuple[3] = feed.entries[i].category
        except AttributeError as error:
            article_tuple[3] = None

        article_tuple[4] = source

        try:
            article_tuple[5] = feed.entries[i].link
        except AttributeError as error:
            article_tuple[5] = None

        article_tuple[6] = None
        article_tuple[7] = None

        df = df.append({'Title': article_tuple[0],
                        'Description': article_tuple[1],
                        'Author': article_tuple[2],
                        'Category': article_tuple[3],
                        'Source': article_tuple[4],
                        'Link': article_tuple[5],
                        'Class_Model1': article_tuple[6],
                        'Class_Model2': article_tuple[7]},
                       ignore_index=True)
        count = count + 1

    return df

def count_articles(feed, max_article):
    count = 0
    test = None

    for i in range(max_article):
        try:
            test = feed.entries[i].title
        except IndexError as error:
            break

        count = count + 1

    return count

def rss_feed(feed_source, df, no_of_articles_per_source):
    for idx, source in enumerate(feed_source):
        feed = feedparser.parse(source[0])
        df = rss_feed_to_dataframe(feed, df, no_of_articles_per_source, source[1])
        count = count_articles(feed,no_of_articles_per_source)
        print("Stored %i articles from %r source" % (count, source[1]))
    return df

#####################################################################################
#Predict Functions
#####################################################################################

def article_preprocess(df):
    length = len(df)

    title_desc_list = [None] * length
    for i in range(length):
        title_desc_list[i] = str(df.iloc[i, 0]) + '. ' + str(df.iloc[i, 1])

    return title_desc_list


def predict_article(df, clf, vectorizer):
    text = article_preprocess(df)

    vectors = vectorizer.transform(text)
    predicted = clf.predict(vectors)

    return predicted

def set_recommendation_articles_model2(df_recommendation,df_article):
    classes_model2 = df_article['Class_Model2'].to_numpy()

    for idx, i in enumerate(classes_model2):
        if math.isnan(df_recommendation.iloc[i, 2]):
            df_recommendation.iloc[i, 2] = idx
            df_recommendation.iloc[i, 3] = df_article.iloc[idx, 0]

    return df_recommendation

def set_recommendation_df(df_recommendation, df_article, feed_source):

    #print(df_article.info())

    for i in range(20):
        priority = False
        source = []

        for idx, sourceinfo in enumerate(feed_source):
            try:
                #print(str(i) + ' ' + str(sourceinfo[2]))
                if i in sourceinfo[2]:
                    source.append(sourceinfo[1])
                    priority = True
            except IndexError as error:
                priority = priority

        for j in [0, 4]:
            df_temp = df_article[df_article.Class_Model2 == i]
            df_temp = df_temp[df_temp.Class_Model1 == j]

            try:
                df_final_recommendation =  pd.DataFrame()
                if priority:
                    for source_name in source:
                        if source_name in df_temp['Source'].values:
                            df_final_recommendation = df_final_recommendation.append(df_temp[df_temp.Source == source_name])


                if df_final_recommendation.empty:
                    df_final_recommendation = df_temp

                df_recommendation = df_recommendation.append({'Class_Model1': j,
                                                              'Class_Model2': i,
                                                              'Article_ID': df_final_recommendation.index[0]},
                                                             ignore_index=True)
            except IndexError as error:
                df_recommendation = df_recommendation.append({'Class_Model1': j,
                                                              'Class_Model2': i,
                                                              'Article_ID': None},
                                                             ignore_index=True)
    return df_recommendation

#####################################################################################
#Run the entire code
#####################################################################################

def main(df_articles, df_recommendation):
    df_recommendation_article_by_model2 = import_model2_categories(const.path, const.filename_model2_class_dataframe)

    df_articles = rss_feed(feed_source, df_articles, no_of_articles_per_source)

    df_articles['Class_Model1'] = predict_article(df_articles, load_model(const.path, const.filename_model1),
                                                  load_feature_extraction(const.path, const.filename_vectorizer_model1))

    df_articles['Class_Model2'] = predict_article(df_articles, load_model(const.path, const.filename_model2),
                                                  load_feature_extraction(const.path, const.filename_vectorizer_model2))

    df_recommendation_article_by_model2 = set_recommendation_articles_model2(df_recommendation_article_by_model2,
                                                                             df_articles)

    df_recommendation = set_recommendation_df(df_recommendation, df_articles, feed_source)

    export_Dataframe_to_csv(df_articles, const.path, const.filename_article_database)
    export_Dataframe_to_csv(df_recommendation_article_by_model2, const.path, const.filename_model2_class_dataframe)
    export_Dataframe_to_csv(df_recommendation, const.path, const.filename_article_recommendation)

main(df_articles, df_recommendation)