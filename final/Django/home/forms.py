from django import forms
import re
from django.contrib.auth.models import User

class UserRegistrationForm(forms.Form):
    username = forms.CharField(label='User name', max_length=30)
    email = forms.EmailField(label='Email')
    pass1 = forms.CharField(label='Password', widget=forms.PasswordInput())
    pass2 = forms.CharField(label='Password', widget=forms.PasswordInput())

    # clean user name data 
    # make sure it does not contain any special characters or it is already existed
    def clean_username(self):
        username = self.cleaned_data['username'] 
        if not re.search(r'^\w+$', username): #check special characters
            raise forms.ValidationError("The username contains special characters /n please re-enter")
        try:
            User.objects.get(username=username)
        except User.DoesNotExist: # check whether user name is taken
            return username
        raise forms.ValidationError("The user name is already existed")

    #clean password data
    #make sure the second password is the same as the first password 
    def clean_password2(self):
            if 'pass1' in self.cleaned_data:
                pass1 = self.cleaned_data['pass1']
                pass2 = self.cleaned_data['pass2']
                if pass1 == pass2 and pass1:
                    return pass2
            raise forms.ValidationError("Password is not matched")
    def save(self):
        User.objects.create_user(username=self.cleaned_data['username'], email=self.cleaned_data['email'], password=self.cleaned_data['pass1'])