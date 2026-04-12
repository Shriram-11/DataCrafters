"""Legacy EDA logic migrated from the original Django implementation."""
from __future__ import annotations

import base64
import io
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro, skew
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from utils.errors import CSVFormatError, ProcessingError

matplotlib.use("Agg")


class AnalysisService:
    """Runs the original data analysis workflow on uploaded CSV data."""

    @staticmethod
    def _encode_plot(current_plt) -> str:
        with io.BytesIO() as buffer:
            current_plt.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        current_plt.close()
        return image_base64

    @staticmethod
    def _sanitize(value):
        """Convert numpy/pandas values into JSON-serializable primitives."""
        if isinstance(value, dict):
            return {str(k): AnalysisService._sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [AnalysisService._sanitize(v) for v in value]
        if isinstance(value, np.generic):
            return AnalysisService._sanitize(value.item())
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value
        return value

    @staticmethod
    def dataset_overview(df: pd.DataFrame) -> dict:
        return {
            "shape": list(df.shape),
            "columns": df.dtypes.apply(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "unique_values": df.nunique().to_dict(),
        }

    @staticmethod
    def descriptive_statistics(df: pd.DataFrame) -> dict:
        stats: dict[str, dict] = {}
        stats["numeric_summary"] = df.describe(
            include="number").fillna(0).to_dict()
        categorical_cols = df.select_dtypes(include="object").columns
        stats["categorical_summary"] = {
            col: df[col].value_counts(dropna=False).to_dict() for col in categorical_cols
        }
        return stats

    @staticmethod
    def data_quality_analysis(df: pd.DataFrame) -> dict:
        duplicates_count = int(df.duplicated().sum())
        missing_data = (df.isnull().mean() * 100)
        missing_data = missing_data[missing_data > 0]

        missing_data_pattern_image = None
        if not missing_data.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=missing_data.index, y=missing_data.values)
            plt.xticks(rotation=45)
            plt.ylabel("Percentage of Missing Values")
            plt.title("Missing Data Percentage by Column")
            missing_data_pattern_image = AnalysisService._encode_plot(plt)

        return {
            "duplicates_count": duplicates_count,
            "missing_data_pattern": missing_data_pattern_image,
            "missing_data_summary": missing_data.to_dict(),
        }

    @staticmethod
    def outlier_analysis(df: pd.DataFrame) -> dict:
        outlier_info: dict[str, dict] = {}
        for column in df.select_dtypes(include=["number"]).columns:
            col_data = df[column].dropna()
            if col_data.empty:
                continue

            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            iqr_outliers = col_data[(col_data < (
                q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))].count()

            plt.figure()
            sns.boxplot(x=col_data)
            boxplot_image = AnalysisService._encode_plot(plt)

            outlier_info[column] = {
                "iqr_outliers_count": int(iqr_outliers),
                "boxplot_image": boxplot_image,
            }
        return outlier_info

    @staticmethod
    def univariate_analysis(df: pd.DataFrame) -> dict:
        analysis: dict[str, dict] = {}
        for column in df.select_dtypes(include=["number"]).columns:
            col_data = df[column].dropna()
            if col_data.empty:
                continue

            skewness_value = float(skew(col_data))
            shapiro_test = shapiro(col_data)
            is_normal = shapiro_test.pvalue > 0.05

            plt.figure()
            sns.histplot(col_data, kde=True)
            plt.title(
                f"{column} Distribution\\nSkewness:{skewness_value:.2f}, p-value: {shapiro_test.pvalue:.3f}"
            )
            plt.xlabel(column)
            plt.ylabel("Frequency")
            distribution_image = AnalysisService._encode_plot(plt)

            analysis[column] = {
                "skewness": skewness_value,
                "normality_test_pvalue": float(shapiro_test.pvalue),
                "is_normal": is_normal,
                "distribution_plot": distribution_image,
            }

        barplots: dict[str, str] = {}
        for column in df.select_dtypes(include=["object"]).columns:
            plt.figure()
            sns.countplot(y=df[column])
            plt.title(f"{column} Distribution")
            plt.xlabel("Frequency")
            barplots[column] = AnalysisService._encode_plot(plt)

        return {"numeric_analysis": analysis, "categorical_barplots": barplots}

    @staticmethod
    def bivariate_analysis(df: pd.DataFrame) -> dict:
        numeric_columns = df.select_dtypes(include=["number"])
        if numeric_columns.empty:
            return {"message": "No numeric columns available for correlation analysis"}

        correlation_matrix = numeric_columns.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True,
                    cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix")
        return {"correlation_matrix": AnalysisService._encode_plot(plt)}

    @staticmethod
    def clustering_analysis(df: pd.DataFrame, n_clusters: int = 3) -> dict:
        numeric_data = df.select_dtypes(include=["number"]).dropna()
        if numeric_data.shape[1] < 2 or numeric_data.empty:
            return {"message": "Not enough numeric features for clustering analysis"}

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(numeric_data)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(numeric_data)
        explained_var = pca.explained_variance_ratio_

        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            c=labels,
            cmap="viridis",
            alpha=0.6,
            edgecolor="k",
        )
        plt.title("Clustering Visualization with PCA")
        plt.xlabel(f"PCA Component 1 ({explained_var[0] * 100:.2f}% Variance)")
        plt.ylabel(f"PCA Component 2 ({explained_var[1] * 100:.2f}% Variance)")
        plt.colorbar(scatter, label="Cluster")

        return {"cluster_plot": AnalysisService._encode_plot(plt)}

    @staticmethod
    def data_cleaning_suggestions(df: pd.DataFrame) -> dict:
        suggestions = {
            "missing_value_imputation": {
                "Consider mean/median imputation.": [],
                "Consider mode or forward-fill imputation.": [],
            },
            "Arbitrary threshold for scaling": [],
        }

        missing_cols = df.columns[df.isnull().mean() > 0]
        for col in missing_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                suggestions["missing_value_imputation"]["Consider mean/median imputation."].append(
                    col)
            else:
                suggestions["missing_value_imputation"]["Consider mode or forward-fill imputation."].append(
                    col)

        for col in df.select_dtypes(include=["number"]).columns:
            if df[col].max() > 100:
                suggestions["Arbitrary threshold for scaling"].append(col)

        return suggestions

    @staticmethod
    def feature_engineering_suggestions(df: pd.DataFrame) -> dict:
        suggestions = {
            "Consider one-hot encoding": [],
            "Consider target encoding or frequency encoding": [],
            "Consider log or square-root transformation": [],
        }

        categorical_cols = df.select_dtypes(include="object").columns
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            if 2 <= unique_vals <= 10:
                suggestions["Consider one-hot encoding"].append(col)
            else:
                suggestions["Consider target encoding or frequency encoding"].append(
                    col)

        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            col_data = df[col].dropna()
            if col_data.empty:
                continue
            skewness_value = skew(col_data)
            if abs(skewness_value) > 1:
                suggestions["Consider log or square-root transformation"].append(
                    col)

        return suggestions

    @staticmethod
    async def analyze_csv_upload(file) -> dict:
        if not file.filename:
            raise CSVFormatError("Missing filename")
        if not file.filename.lower().endswith(".csv"):
            raise CSVFormatError("Only .csv files are supported")

        content = await file.read()
        if not content:
            raise CSVFormatError("Uploaded CSV is empty")

        try:
            df = pd.read_csv(io.BytesIO(content))
            if df.empty:
                raise CSVFormatError("Uploaded CSV has no rows")

            report = {
                "dataset_overview": AnalysisService.dataset_overview(df),
                "descriptive_statistics": AnalysisService.descriptive_statistics(df),
                "data_quality_analysis": AnalysisService.data_quality_analysis(df),
                "outlier_analysis": AnalysisService.outlier_analysis(df),
                "univariate_analysis": AnalysisService.univariate_analysis(df),
                "bivariate_analysis": AnalysisService.bivariate_analysis(df),
                "clustering_analysis": AnalysisService.clustering_analysis(df),
                "data_cleaning_suggestions": AnalysisService.data_cleaning_suggestions(df),
                "feature_engineering_suggestions": AnalysisService.feature_engineering_suggestions(df),
            }
            report = AnalysisService._sanitize(report)
            return {
                "success": True,
                "message": "CSV uploaded successfully",
                "data": report,
            }
        except CSVFormatError:
            raise
        except Exception as exc:
            raise ProcessingError(
                f"Failed to process CSV: {str(exc)}") from exc
        finally:
            await file.close()
