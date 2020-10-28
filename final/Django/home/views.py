from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from home.PredictRecommend import *
from .forms import UserRegistrationForm
from django.http import HttpResponse
from django.shortcuts import render
from django.template.loader import get_template
from django.http import HttpResponseRedirect
from .models import Post, UserPost, Recommendation
from django.contrib.auth.models import User
from datetime import datetime



def createpost(request):
        
        if request.method == 'POST':
            if  request.POST.get('content'):
                if  request.user.is_authenticated:
                    post=UserPost()
                    post.content= request.POST.get('content')
                    post.user= request.user
                    post.date= datetime.now()
                    post.save()
                    sentiment = Sentiment(post.content)
                    return render(request, 'posts/recommend.html',{'time':post.date,'user':post.user,'input':post.content,'model1':sentiment.tag_model1,'model2':sentiment.tag_model2,'title':sentiment.title,'description':sentiment.description , 'source':sentiment.source, 'link':sentiment.link})
                else:
                    post=Post()
                    post.content= request.POST.get('content')
                    post.save()
                    sentiment = Sentiment(post.content)
                    return render(request, 'posts/recommend.html',{'input':post.content,'model1':sentiment.tag_model1,'model2':sentiment.tag_model2,'title':sentiment.title,'description':sentiment.description , 'source':sentiment.source, 'link':sentiment.link})
                  
        else:
                
                return render(request,'posts/recommend.html',)




def register(request):
    form = UserRegistrationForm()
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            return HttpResponseRedirect('/')
    return render(request, 'posts/userregister.html', {'form': form})



def run(request):
    sentiment = request.GET.get('sentimentSelect')
    category = request.GET.get('categories')
    newsentiment= SentimentOverwrite(int(sentiment),int(category))
    return render(request, 'posts/recommend.html', {'senti':newsentiment.tag_model1,'cate':newsentiment.tag_model2,'newtitle': newsentiment.title,'newdescription': newsentiment.description,'newsource': newsentiment.source,'newlink': newsentiment.link })


def rating(request):
    rating= int(request.GET.get('rating'))
    return render(request, 'posts/recommend.html', {'rating':rating})


