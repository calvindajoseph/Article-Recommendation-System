from import_export import resources
from .models import Post

class PostResource(resources.ModelResource):
    class Meta:
        model = Post
    
