from django.shortcuts import render
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password

from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.permissions import AllowAny
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.response import Response
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework import status, permissions
from django.core.cache import cache
from rest_framework.parsers import MultiPartParser
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

# eda helper functions

# views


@api_view(['POST'])
@permission_classes([AllowAny])
def signup(request):
    username = request.data.get('username')
    password = request.data.get('password')
    if not username or not password:
        return Response({"success": False, "message": "Data incomplete"}, status=status.HTTP_400_BAD_REQUEST)
    if User.objects.filter(username=username).exists():
        return Response({"success": False, "message": "Username already exists"}, status=status.HTTP_400_BAD_REQUEST)
    user = User.objects.create(
        username=username,
        password=make_password(password))
    return Response({"success": True, "message": "User Created Succesfully"}, status=status.HTTP_201_CREATED)


class LoginView(TokenObtainPairView):
    # Use a list without parentheses
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        response = super().post(request, *args, **kwargs)
        if response.status_code == 200:
            return Response({"success": True, "message": "Logged in successfully", "data": response.data}, status=status.HTTP_200_OK)

        return Response({"success": False, "message": "Incorrect username or password"}, status=status.HTTP_401_UNAUTHORIZED)


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
@parser_classes([MultiPartParser])
def upload_csv(request):
    if 'csv_file' not in request.FILES:
        return Response({"success": False, "message": "File not found"}, status=status.HTTP_400_BAD_REQUEST)

    csv_file = request.FILES['csv_file']
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Store DataFrame in cache
        cache.set('csv_data', df)

        # Prepare EDA report
        eda_report = {
            "shape": df.shape,
            "columns": df.columns.tolist()  # Convert columns to a list
        }

        return Response({"success": True, "message": "CSV uploaded successfully", "data": eda_report}, status=status.HTTP_200_OK)

    except Exception as e:
        # Handle any errors with reading the CSV
        return Response({"success": False, "message": f"Failed to process CSV: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
