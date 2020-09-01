from django.shortcuts import render
from django.http import HttpResponse
from home.PredictRecommend import *



from .models import Post, Recommendation
def createpost(request):
        
        if request.method == 'POST':
            if  request.POST.get('content'):
                post=Post()
                post.content= request.POST.get('content')
                post.save()
                sentiment = Sentiment(post.content)
                return render(request, 'posts/recommend.html',{'input':post.content,'model1':sentiment.tag_model1,'model2':sentiment.tag_model2,'title':sentiment.title,'description':sentiment.description , 'source':sentiment.source, 'link':sentiment.link})
                  
        else:
                
                return render(request,'posts/recommend.html',)


