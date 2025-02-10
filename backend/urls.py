from django.urls import path
from .views import log_query
from .views import show_df

urlpatterns = [
    path('show_df/', show_df, name='show_df'),
    path('log_query/', log_query, name='log_query'),
]
