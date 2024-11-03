from django.shortcuts import render
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from scipy.stats import shapiro, skew
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
import matplotlib
matplotlib.use('Agg')  #
# eda helper functions
# Helper function to encode plot images to base64


def encode_plot(plt):
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close()
    return image_base64

# 1. Dataset Overview


def dataset_overview(df):
    return {
        "shape": df.shape,
        "columns": df.dtypes.apply(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "unique_values": df.nunique().to_dict()
    }

# 2. Descriptive Statistics


def descriptive_statistics(df):
    stats = {}
    # Numeric columns
    numeric_summary = df.describe().to_dict()
    stats['numeric_summary'] = numeric_summary
    # Categorical columns
    categorical_cols = df.select_dtypes(include='object').columns
    stats['categorical_summary'] = {
        col: df[col].value_counts().to_dict() for col in categorical_cols
    }
    return stats

# 3. Data Quality Analysis


def data_quality_analysis(df):
    # Count duplicates
    duplicates_count = df.duplicated().sum()

    # Calculate missing values
    missing_data = df.isnull().mean() * 100  # Percentage of missing values
    # Filter columns with missing data
    missing_data = missing_data[missing_data > 0]

    # Check if there are any missing values to plot
    if not missing_data.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_data.index, y=missing_data.values)
        plt.xticks(rotation=45)
        plt.ylabel("Percentage of Missing Values")
        plt.title("Missing Data Percentage by Column")
        missing_data_pattern_image = encode_plot(plt)
    else:
        missing_data_pattern_image = None  # No missing data to visualize

    return {
        "duplicates_count": duplicates_count,
        "missing_data_pattern": missing_data_pattern_image,
        # Return missing data as a dictionary
        "missing_data_summary": missing_data.to_dict()
    }


# 4. Univariate Analysis


def univariate_analysis(df):
    histograms = {}
    for column in df.select_dtypes(include=['number']).columns:
        sns.histplot(df[column].dropna(), kde=True)
        histograms[column] = encode_plot(plt)
    barplots = {}
    for column in df.select_dtypes(include=['object']).columns:
        sns.countplot(y=df[column])
        barplots[column] = encode_plot(plt)
    return {"histograms": histograms, "barplots": barplots}


def normality_skewness_analysis(df):
    analysis = {}
    for column in df.select_dtypes(include=['number']).columns:
        col_data = df[column].dropna()
        skewness_value = skew(col_data)

        # Shapiro-Wilk Test for normality
        shapiro_test = shapiro(col_data)
        # If p-value > 0.05, data is likely normal
        is_normal = shapiro_test.pvalue > 0.05

        # Plot Q-Q plot for visual inspection
        fig = plt.figure()
        sns.histplot(col_data, kde=True)
        plt.title(f"{column} Distribution (Skewness: {skewness_value:.2f})")
        distribution_image = encode_plot(plt)

        # Record results
        analysis[column] = {
            "skewness": skewness_value,
            "normality_test_pvalue": shapiro_test.pvalue,
            "is_normal": is_normal,
            "distribution_plot": distribution_image
        }
    return analysis
# 5. Bivariate Analysis


def bivariate_analysis(df):
    correlation_matrix = sns.heatmap(df.corr(), annot=True)
    correlation_image = encode_plot(plt)
    return {"correlation_matrix": correlation_image}

# 6. Multivariate Analysis


def multivariate_analysis(df):
    # Placeholder for multivariate techniques (e.g., PCA) if required
    return {"multivariate_analysis": "Not implemented"}
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
            "dataset_overview": dataset_overview(df),
            "descriptive_statistics": descriptive_statistics(df),
            "data_quality_analysis": data_quality_analysis(df),
            "univariate_analysis": univariate_analysis(df),
            "bivariate_analysis": bivariate_analysis(df),
            "multivariate_analysis": multivariate_analysis(df),
            "normality_skewness_analysis": normality_skewness_analysis(df)
        }
        print(eda_report['normality_skewness_analysis'])

        return Response({"success": True, "message": "CSV uploaded successfully", "data": eda_report}, status=status.HTTP_200_OK)

    except Exception as e:
        # Handle any errors with reading the CSV
        return Response({"success": False, "message": f"Failed to process CSV: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
