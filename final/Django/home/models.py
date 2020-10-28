from django.db import models
from django.conf import settings
from django.contrib.auth.models import User

# Create your models here.
class Post(models.Model):
    #title= models.CharField(max_length=300, unique=True)
    content= models.TextField()
    def __str__(self):
        return self.content




class UserPost(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    post = models.TextField()
    date = models.DateTimeField(auto_now_add=True)
    
    
    
    def __str__(self):
        return self.post

class Recommendation(models.Model):
    post= models.ForeignKey(Post,on_delete=models.CASCADE)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    model1= models.CharField(max_length=255,null=True, blank=True)
    model2= models.CharField(max_length=255,null=True, blank=True)
    title= models.CharField(max_length=255)
    description= models.TextField()
    source= models.CharField(max_length=255)
    link= models.URLField(max_length = 200)
    ratings= models.IntegerField(default=0)
    
    def __str__(self):
        return str(self.user) + ':' + str(self.post) +':' + str(self.title)



    
