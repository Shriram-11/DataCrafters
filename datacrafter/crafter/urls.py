# crafter/urls.py
from django.urls import path
from .views import signup, LoginView, upload_csv
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    path('signup/', signup, name='signup'),
    path('login/', LoginView.as_view(), name='login'),  # JWT login endpoint
    path('token/refresh/', TokenRefreshView.as_view(),
         name='token_refresh'),  # Token refresh
    path('upload_csv/', upload_csv, name='upload_csv'),  # CSV upload
]
