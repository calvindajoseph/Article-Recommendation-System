from django.urls import path
from django.urls import reverse_lazy
from django.conf.urls import url, include
from . import views
from django.contrib.auth import views as auth_views


urlpatterns = [
    path('', views.createpost),
    path('register/', views.register, name="register"),
    path('accounts/', include('django.contrib.auth.urls')),
    path('login/',auth_views.LoginView.as_view(template_name="posts/login.html"), name="login"),
    path('logout/',auth_views.LogoutView.as_view(next_page='/'),name='logout'),
    path('password_reset/', auth_views.PasswordResetView.as_view(template_name="posts/password_reset.html",success_url = reverse_lazy('password_reset_done')) ,name='password_reset'),
    path('accounts/password_reset/done/', auth_views.PasswordChangeDoneView.as_view(template_name="posts/password_reset_done.html"), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name="posts/password_reset_confirm.html", success_url = reverse_lazy('password_reset_complete')),name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name="posts/password_reset_complete.html"), name='password_reset_complete'),
    path('analysis1/', views.run, name='run'),
    path('analysis2/', views.rating, name='rating'),
]

