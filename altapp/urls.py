from django.urls import path
from . import views

urlpatterns = [
    path('save_message/', views.save_message, name='save_message'),
    path('load_conversation/', views.load_conversation, name='load_conversation'),
    path('create_new_session/', views.create_new_session, name='create_new_session'),
    path('get_sessions/', views.get_sessions, name='get_sessions'),
    path('load_resources/', views.load_resources, name='load_resources'),
    path('chatbot_view/', views.chatbot_view, name='chatbot_view'),
    path('set_csrf_token/', views.set_csrf_token, name='set_csrf_token'),
    path('handle_email/', views.handle_email, name='handle_email'),
    path('', views.home, name='home'),
]
