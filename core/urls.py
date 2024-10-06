
from django.urls import path
from . import views

urlpatterns = [
    path('', views.handle_pdf_upload, name='handle_pdf_upload'),
    path('ask/', views.handle_user_question, name='handle_user_question'),

]
