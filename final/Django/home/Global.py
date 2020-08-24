class Global:

    def __init__(self):
        self.path = r'D:\project1\pythonweb\home'

        self.filename_model1 = 'model1.joblib'
        self.filename_model2 = 'model2.joblib'

        self.filename_vectorizer_model1 = 'vectorizer_model1.pickle'
        self.filename_vectorizer_model2 = 'vectorizer_model2.pickle'

        self.filename_model2_logs = 'model2_logs.txt'

        self.filename_model1_class_dataframe = 'Model1_Class.csv'
        self.filename_model2_class_dataframe = 'Model2_Class.csv'

        self.filename_article_database = 'article_database.csv'
        self.filename_article_recommendation = 'article_recommendation.csv'

class RSS_Website:
    
    def __init__(self, link, source, priority):
        self.link = link
        self.source = source
        self.priority = priority

class Model1:

    def __init__(self, classifier):
        switchers = {
            0: 'Negative',
            2: 'Neutral',
            4: 'Positive'
        }
        self.modelclass = classifier
        self.tag = switchers.get(classifier)

class Model2:

    def __init__(self, classifier):
        switchers = {
            0: 'Religion: Atheism',
            1: 'Computer: Graphics',
            2: 'Computer: Windows',
            3: 'Computer: IBM PC Hardware',
            4: 'Computer: Apple Hardware',
            5: 'Computer: Windows X',
            6: 'Miscellaneous: For-Sale',
            7: 'Recreation: Automotives',
            8: 'Recreation: Motorcycles',
            9: 'Sports: Baseball',
            10: 'Sports: Hockey',
            11: 'Science: Cryptography',
            12: 'Science: Electronics',
            13: 'Science: Medical',
            14: 'Science: Space',
            15: 'Religion: Christian',
            16: 'Politics: Guns',
            17: 'Politics: Middle East',
            18: 'Politics: Miscellaneous',
            19: 'Religion: Miscellaneous',
        }
        self.modelclass = classifier
        self.tag = switchers.get(classifier)
