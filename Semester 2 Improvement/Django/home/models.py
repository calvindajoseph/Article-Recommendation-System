from django.db import models

# Create your models here.
class Post(models.Model):
    #title= models.CharField(max_length=300, unique=True)
    content= models.TextField()
    def __str__(self):
        return self.content

class Recommendation(models.Model):
    title= models.CharField(max_length=300, unique=True)
    description= models.CharField(max_length=300, unique=True)
    source= models.CharField(max_length=300, unique=True)
    link= models.CharField(max_length=300, unique=True)
    def __str__(self):
        return self.title
