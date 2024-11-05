from django.shortcuts import render
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from scipy.stats import shapiro, skew, chi2_contingency
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

matplotlib.use('Agg')


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

# 4. Outlier Analysis


def outlier_analysis(df):
    outlier_info = {}
    for column in df.select_dtypes(include=['number']).columns:
        col_data = df[column].dropna()
        # IQR method
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = col_data[(col_data < (Q1 - 1.5 * IQR))
                                | (col_data > (Q3 + 1.5 * IQR))].count()

        # Visualize outliers using boxplot
        sns.boxplot(x=col_data)
        boxplot_image = encode_plot(plt)

        # Save analysis
        outlier_info[column] = {
            "iqr_outliers_count": int(iqr_outliers),
            "boxplot_image": boxplot_image
        }
    return outlier_info

# 5. Univariate Analysis


def univariate_analysis(df):
    analysis = {}

    for column in df.select_dtypes(include=['number']).columns:
        col_data = df[column].dropna()

        # Skewness and Shapiro-Wilk normality test
        skewness_value = skew(col_data)
        shapiro_test = shapiro(col_data)
        is_normal = shapiro_test.pvalue > 0.05

        # Plot histogram with KDE
        plt.figure()
        sns.histplot(col_data, kde=True)
        t = f"{column} Distribution\nSkewness: {
            skewness_value:.2f}, p-value: {shapiro_test.pvalue:.3f}"
        plt.title(t)
        plt.xlabel(column)
        plt.ylabel('Frequency')

        # Encode the plot
        distribution_image = encode_plot(plt)

        # Record results in analysis dictionary
        analysis[column] = {
            "skewness": skewness_value,
            "normality_test_pvalue": shapiro_test.pvalue,
            "is_normal": is_normal,
            "distribution_plot": distribution_image
        }

    # Categorical columns
    barplots = {}
    for column in df.select_dtypes(include=['object']).columns:
        plt.figure()
        sns.countplot(y=df[column])
        plt.title(f"{column} Distribution")
        plt.xlabel('Frequency')

        # Encode the plot
        barplots[column] = encode_plot(plt)

    return {"numeric_analysis": analysis, "categorical_barplots": barplots}

# 6. Bivariate Analysis


def bivariate_analysis(df):
    correlation_matrix = sns.heatmap(df.corr(), annot=True)
    correlation_image = encode_plot(plt)
    return {"correlation_matrix": correlation_image}

# 7. Advanced Correlation Analysis


def advanced_correlation_analysis(df, target_col=None):
    results = {}

    # Correlation with Target Variable (if applicable)
    if target_col and target_col in df.columns:
        target_correlation = df.corr()[target_col].sort_values(
            ascending=False).to_dict()
        results['target_correlation'] = target_correlation

    # Categorical variable dependencies using Chi-Square
    categorical_cols = df.select_dtypes(include='object').columns
    chi2_results = {}
    for col in categorical_cols:
        contingency_table = pd.crosstab(df[col], df[target_col])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        chi2_results[col] = {"chi2_statistic": chi2, "p_value": p}

    results['chi_square_tests'] = chi2_results

    # Pairwise plots for numeric columns
    sns.pairplot(df.select_dtypes(include=['number']))
    pairplot_image = encode_plot(plt)
    results['pairplot'] = pairplot_image

    return results

# 8. Clustering Analysis


def clustering_analysis(df, n_clusters=3):
    # Select numeric columns
    numeric_data = df.select_dtypes(include=['number']).dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(numeric_data)
    df['cluster'] = labels

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(numeric_data)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis')
    plt.title('Clustering Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    cluster_plot = encode_plot(plt)

    return {"cluster_plot": cluster_plot}

# 9. Data Cleaning and Scaling Suggestions


def data_cleaning_suggestions(df):
    cleaning_suggestions = {
        "missing_value_imputation": {"mean_median": [], "mode_forward_fill": []},
        "scaling_normalization": []
    }

    # Missing value imputation recommendations
    missing_cols = df.columns[df.isnull().mean() > 0]
    for col in missing_cols:
        col_type = df[col].dtype
        if col_type in ['int64', 'float64']:
            # Consider mean/median imputation.
            cleaning_suggestions["missing_value_imputation"]["mean_median"].append(
                col)
        else:
            # Consider mode or forward-fill imputation.
            cleaning_suggestions["missing_value_imputation"]["mode_forward_fill"].append(
                col)
    # Scaling recommendations
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].max() > 100:  # Arbitrary threshold for scaling suggestion
            cleaning_suggestions["scaling_normalization"].append(col)

    return cleaning_suggestions

# 10. Feature Engineering


def feature_engineering_suggestions(df):
    suggestions = {"encoding": [],
                   "transformation": []}

    # High cardinality categorical columns
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        if df[col].nunique() > 3:  # Define high cardinality threshold as needed
            # "Consider encoding techniques like target encoding."
            suggestions["encoding"].append(col)

    # Numeric column transformations
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        skewness_value = skew(df[col].dropna())
        if abs(skewness_value) > 1:  # Threshold for suggesting transformation
            # "Consider log or square-root transformation"
            suggestions["transformation"].append(col)

    return suggestions

# endponits


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
            "outlier_analysis": outlier_analysis(df),
            "univariate_analysis": univariate_analysis(df),
            "bivariate_analysis": bivariate_analysis(df),
            "advanced_correlation_analysis": advanced_correlation_analysis(df),
            "clustering_analysis": clustering_analysis(df),
            "data_cleaning_suggestions": data_cleaning_suggestions(df),
            "feature_engineering_suggestions": feature_engineering_suggestions(df)
        }
        return Response({"success": True, "message": "CSV uploaded successfully", "data": eda_report}, status=status.HTTP_200_OK)

    except Exception as e:
        # Handle any errors with reading the CSV
        return Response({"success": False, "message": f"Failed to process CSV: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
