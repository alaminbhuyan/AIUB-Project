from django.urls import path
from app1 import views


urlpatterns = [
    path(route='', view=views.home, name="home_page"),
    path(route='signup/', view=views.signup, name="signup_page"),
    path(route='login/', view=views.login, name="login_page"),
    path('logout/', views.logout, name="logout_page"),
    path(route='service/', view=views.service, name="service_page"),
    path(route='imageProcessing/', view=views.imageProcessing, name="img_process_page"),
]
