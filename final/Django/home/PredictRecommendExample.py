import PredictRecommend

input_sentiment = 'My favorite operating system is Windows'

sentiment = PredictRecommend.Sentiment(input_sentiment)

print('\nTag Class 1:         ' + sentiment.tag_model1)
print('Tag Class 2:         ' + sentiment.tag_model2 + '\n')

print('Recommendation Article:\nTitle:       ' + sentiment.title)
print('Description: ' + sentiment.description)
print('Source:      ' + sentiment.source)
print('Link:        ' + sentiment.link)