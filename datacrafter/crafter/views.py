from django.views.decorators.csrf import csrf_exempt
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
    """
    Encodes a matplotlib plot as a base64 string for JSON response.

    Parameters:
    plt (matplotlib.pyplot): The pyplot instance with the plot.

    Returns:
    str: Base64 encoded string of the plot image.

    Raises:
    Exception: If an error occurs in encoding, prints a message with details.
    """
    with io.BytesIO() as buf:
        # Use tight layout for cleaner images
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()  # Ensures plot closes after encoding
    return image_base64

# 1. Dataset Overview


def dataset_overview(df):
    """
    Generate a summary overview of the dataset.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
    dict: A dictionary containing dataset shape, column types,
          missing values per column, and unique values per column.

    Raises:
    Exception: If an error occurs in processing, prints a message with details.
    """
    try:
        return {
            "shape": df.shape,
            "columns": df.dtypes.apply(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "unique_values": df.nunique().to_dict()
        }
    except Exception as e:
        print(f"Error in dataset_overview function: {str(e)}")

 # 2. Descriptive Statistics


def descriptive_statistics(df):
    """
    Calculate descriptive statistics for numeric columns in the dataset.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
    dict: A dictionary with descriptive statistics (count, mean, std, min, 25%, 50%, 75%, max)
          for each numeric column.

    Raises:
    Exception: If an error occurs in processing, prints a message with details.
    """
    try:
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
    except Exception as e:
        print(f"Error in descriptive_statistics function: {str(e)}")


def data_quality_analysis(df):
    """
    Perform data quality analysis to detect missing and duplicated values.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
    dict: A dictionary with missing values per column and total duplicated rows.

    Raises:
    Exception: If an error occurs in processing, prints a message with details.
    """
    try:
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
    except Exception as e:
        print(f"Error in data_quality_analysis function: {str(e)}")


def outlier_analysis(df):
    """
    Identify and visualize outliers in the numeric columns using the IQR method.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
    dict: A dictionary containing outlier counts and encoded boxplot images for each numeric column.

    Raises:
    Exception: If an error occurs in processing, prints a message with details.
    """
    try:
        outlier_info = {}
        for column in df.select_dtypes(include=['number']).columns:
            col_data = df[column].dropna()
            # IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = col_data[(col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))].count()

            # Visualize outliers using boxplot
            sns.boxplot(x=col_data)
            boxplot_image = encode_plot(plt)

            # Save analysis
            outlier_info[column] = {
                "iqr_outliers_count": int(iqr_outliers),
                "boxplot_image": boxplot_image
            }
        return outlier_info
    except Exception as e:
        print(f"Error in outlier_analysis function: {str(e)}")


def univariate_analysis(df):
    """
    Perform univariate analysis to calculate frequency distributions for categorical columns.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
    dict: A dictionary with frequency distributions for each categorical column.

    Raises:
    Exception: If an error occurs in processing, prints a message with details.
    """
    try:
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
            t = f"{column} Distribution\nSkewness:{skewness_value:.2f}, p-value: {shapiro_test.pvalue:.3f}"
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
    except Exception as e:
        print(f"Error in univariate_analysis function: {str(e)}")

# 6. Bivariate Analysis


def bivariate_analysis(df):
    try:
        numeric_columns = df.select_dtypes(include=['number'])

        if numeric_columns.empty:
            return {"message": "No numeric columns available for correlation analysis"}

        # Generate correlation matrix for numeric columns only
        correlation_matrix = numeric_columns.corr()

        # Plotting the heatmap
        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix")

        # Encode the plot
        correlation_image = encode_plot(plt)
        return {"correlation_matrix": correlation_image}
    except Exception as e:
        print(f"Error in bivariate: {e}")


# 7. Advanced Correlation Analysis


'''
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
'''
# 8. Clustering Analysis


def clustering_analysis(df, n_clusters=3):
    """
    Perform KMeans clustering on numeric columns and visualize the clusters using PCA.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the dataset.
    n_clusters (int): The number of clusters to form with KMeans.

    Returns:
    dict: A dictionary containing the encoded cluster plot image.

    Raises:
    Exception: If an error occurs in processing, prints a message with details.
    """
    try:
        # Select numeric columns
        numeric_data = df.select_dtypes(include=['number']).dropna()

        if numeric_data.shape[1] < 2:
            return {"message": "Not enough numeric features for clustering analysis"}

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(numeric_data)
        df['cluster'] = labels

        # Perform PCA for 2D visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(numeric_data)

        # Retrieve explained variance ratio
        explained_var = pca.explained_variance_ratio_

        # Plotting the clusters with PCA components labeled by explained variance
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.6,
            edgecolor='k'
        )
        plt.title('Clustering Visualization with PCA')
        plt.xlabel(f'PCA Component 1 ({explained_var[0]*100:.2f}% Variance)')
        plt.ylabel(f'PCA Component 2 ({explained_var[1]*100:.2f}% Variance)')
        plt.colorbar(scatter, label='Cluster')

        # Encode the plot
        cluster_plot = encode_plot(plt)

        return {"cluster_plot": cluster_plot}
    except Exception as e:
        print(f"Error in clustering_analysis function: {str(e)}")


# 9. Data Cleaning and Scaling Suggestions


def data_cleaning_suggestions(df):
    """
    Provide data cleaning suggestions, including imputation for missing values and scaling recommendations.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
    dict: A dictionary containing missing value imputation and scaling suggestions.

    Raises:
    Exception: If an error occurs in processing, prints a message with details.
    """
    try:
        cleaning_suggestions = {
            "missing_value_imputation": {"Consider mean/median imputation.": [], "Consider mode or forward-fill imputation.": []},
            "Arbitrary threshold for scaling": []
        }

        # Missing value imputation recommendations
        missing_cols = df.columns[df.isnull().mean() > 0]
        for col in missing_cols:
            col_type = df[col].dtype
            if col_type in ['int64', 'float64']:
                # Consider mean/median imputation.
                cleaning_suggestions["missing_value_imputation"]["Consider mean/median imputation."].append(
                    col)
            else:
                # Consider mode or forward-fill imputation.
                cleaning_suggestions["missing_value_imputation"]["Consider mode or forward-fill imputation."].append(
                    col)
        # Scaling recommendations
        for col in df.select_dtypes(include=['number']).columns:
            if df[col].max() > 100:  # Arbitrary threshold for scaling suggestion
                cleaning_suggestions["Arbitrary threshold for scaling"].append(
                    col)

        return cleaning_suggestions
    except Exception as e:
        print(f"Error in data_cleaning_suggestions function: {str(e)}")


def feature_engineering_suggestions(df):
    """
    Provide feature engineering suggestions, including encoding and transformation recommendations.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
    dict: A dictionary with encoding and transformation suggestions for feature engineering.

    Raises:
    Exception: If an error occurs in processing, prints a message with details.
    """
    try:
        suggestions = {"Consider one-hot encoding": [],
                       "Consider target encoding or frequency encoding": [],
                       "Consider log or square-root transformation": []}

        # High cardinality categorical columns
        categorical_cols = df.select_dtypes(include='object').columns
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            if 2 <= unique_vals <= 10:  # Adjust threshold as needed
                # Suggest One-Hot Encoding for low cardinality
                suggestions["Consider one-hot encoding"].append(col)
            else:
                # Suggest other techniques for high cardinality
                suggestions["Consider target encoding or frequency encoding"].append(
                    col)

        # Numeric column transformations
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            skewness_value = skew(df[col].dropna())
            if abs(skewness_value) > 1:  # Threshold for suggesting transformation
                # "Consider log or square-root transformation"
                suggestions["Consider log or square-root transformation"].append(
                    col)

        return suggestions
    except Exception as e:
        print(f"Error in feature_engineering_suggestions function: {str(e)}")


# endponits


@api_view(['POST'])
@permission_classes([AllowAny])
def signup(request):
    """
    API endpoint to handle user signup.

    Parameters:
    request (HttpRequest): The HTTP request containing 'username' and 'password' data.

    Returns:
    Response: A response with a success or failure message for the signup process.

    Raises:
    None
    """
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
        """
        API endpoint to handle user login using JWT tokens.

        Parameters:
        request (HttpRequest): The HTTP request containing 'username' and 'password' data.

        Returns:
        Response: A response with a success or failure message for the login process, including JWT tokens.

        Raises:
        None
        """
        response = super().post(request, *args, **kwargs)
        if response.status_code == 200:
            return Response({"success": True, "message": "Logged in successfully", "data": response.data}, status=status.HTTP_200_OK)

        return Response({"success": False, "message": "Incorrect username or password"}, status=status.HTTP_401_UNAUTHORIZED)


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
@parser_classes([MultiPartParser])
def upload_csv(request):
    """
    API endpoint to upload a CSV file and perform an EDA report generation.

    Parameters:
    request (HttpRequest): The HTTP request containing the CSV file in 'csv_file'.

    Returns:
    Response: A response with the EDA report or an error message if processing fails.

    Raises:
    Exception: If an error occurs in processing, prints a message with details.
    """
    if 'csv_file' not in request.FILES:
        return Response({"success": False, "message": "File not found"}, status=status.HTTP_400_BAD_REQUEST)

    csv_file = request.FILES['csv_file']
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Store DataFrame in cache
        cache.set('csv_data', df)
        # "advanced_correlation_analysis": advanced_correlation_analysis(df),

        # Prepare EDA report
        eda_report = {
            "dataset_overview": dataset_overview(df),
            "descriptive_statistics": descriptive_statistics(df),
            "data_quality_analysis": data_quality_analysis(df),
            "outlier_analysis": outlier_analysis(df),
            "univariate_analysis": univariate_analysis(df),
            "bivariate_analysis": bivariate_analysis(df),
            "clustering_analysis": clustering_analysis(df),
            "data_cleaning_suggestions": data_cleaning_suggestions(df),
            "feature_engineering_suggestions": feature_engineering_suggestions(df)
        }
        print("all okay")
        return Response({"success": True, "message": "CSV uploaded successfully", "data": eda_report}, status=status.HTTP_200_OK)

    except Exception as e:
        # Handle any errors with reading the CSV
        return Response({"success": False, "message": f"Failed to process CSV: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)


@csrf_exempt
@api_view(['GET'])
def health_check(request):
    """
    API endpoint to check the health/status of the server.

    Parameters:
    request (HttpRequest): The HTTP GET request.

    Returns:
    Response: A response indicating the server is working.

    Raises:
    None
    """
    return Response({"success": True, "message": "working"})
