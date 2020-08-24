# Welcome to the emotion prediction project

In this project, we will build two machine learning models and a webpage which will have a text box and button. When the user enters any sentence in the text box and hits the button, there will be a recommendation given to the user based on what he/she wrote. It could be any recommendation, but we choose articles to recommend for users to read. For example, If the user enters a sentence like “My favorite operating system is Windows” and hits the button, then a recommendation will be given to read a news article. 
We will be dealing with different areas, a) Web development front-end and back-end, b) Machine learning models, c) Sentiment analysis, d) Recommender system. 

## To do

Our project will have three main parts: 

1-	Two machine learning models

2-	Recommender system

3- Django and front-end

The first machine learning model will have: positive and negative classes. The second model will have sub-classes and 20newsgroups will be used as the dataset. The second part of the project is a recommender system. It will recommend news articles to the user based on the user input. Finally, we will design a webpage and connect them all using Django which is a high-level Python-based free and open-source web framework.

## System Model

![System Model AI](https://user-images.githubusercontent.com/56243454/81346353-7b6ba800-90fd-11ea-8837-ac5c16d8c058.png)

As the system model depicts that when the user input any text it will go directly to Django (back-end). Then the back-end is connected to both models 1 and 2. Model 1 uses Sentiment140 as the dataset which has sentiment classes (positive or negative) and Model 2 uses 20newsgroups as the dataset and it has different 20 topics. After that, Set recommendation is a function in python will help retrieving all articles from rss feeds links and retrieve both machine learning models to the system and apply it to the database so we can predict the user input based on the sentiment and topic to recommend the new article. After predicting all articles, we create a recommendation database (Article Recommendation) consistes of (Class_Model1, Class_Model2 and Article_ID). Therefore, as the fourth row below in article_recommendation shows that if Model 1 and 2 classify the user input as sentiment 0 for model 1 and topic 1 for model 2 then article number 51 is recommended. 

![article_recommendation](https://user-images.githubusercontent.com/56243454/81346553-ceddf600-90fd-11ea-8bf2-29b9a8008ad9.png)

Finally, the user will see these outputs on the website:

1- Title 

2- Description 

3- Source

4- Article link.

## Web page design

![fullwebsite screenshot](https://user-images.githubusercontent.com/56243454/81629332-3f5e7d00-9446-11ea-8466-185b7cc3c571.png)

When the user input is "I do not like robots", then the output will be about an article in relation to robots as shown below:

![I don't like Robots](https://user-images.githubusercontent.com/56243454/81629729-2d310e80-9447-11ea-9087-155ad038d880.png)





