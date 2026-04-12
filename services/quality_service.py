"""Quality checks service for CSV datasets."""
from __future__ import annotations

import io

import pandas as pd
from fastapi import UploadFile

from utils.errors import CSVFormatError, ProcessingError


class QualityService:
    """Compute quality metrics from uploaded CSV files."""

    @staticmethod
    def quality_dataframe(df: pd.DataFrame) -> dict:
        total_rows = int(df.shape[0])
        total_columns = int(df.shape[1])
        total_cells = max(total_rows * total_columns, 1)

        duplicates_count = int(df.duplicated().sum())
        missing_count = int(df.isnull().sum().sum())
        completeness_score = round(
            (1 - (missing_count / total_cells)) * 100, 3)

        missing_by_column = {
            col: {
                "count": int(df[col].isnull().sum()),
                "percent": round(float(df[col].isnull().mean() * 100), 3),
            }
            for col in df.columns
        }

        outlier_counts = {}
        for col in df.select_dtypes(include=["number"]).columns:
            series = df[col].dropna()
            if series.empty:
                outlier_counts[col] = 0
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            outliers = series[(series < (q1 - 1.5 * iqr)) |
                              (series > (q3 + 1.5 * iqr))]
            outlier_counts[col] = int(outliers.count())

        return {
            "duplicates_count": duplicates_count,
            "missing_count": missing_count,
            "completeness_score": completeness_score,
            "missing_by_column": missing_by_column,
            "outlier_count_by_numeric_column": outlier_counts,
        }

    @staticmethod
    async def quality_csv_upload(file: UploadFile) -> dict:
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
            quality = QualityService.quality_dataframe(df)
            return {
                "success": True,
                "message": "Quality checks completed successfully",
                "data": quality,
            }
        except CSVFormatError:
            raise
        except Exception as exc:
            raise ProcessingError(
                f"Failed to run quality checks: {str(exc)}") from exc
        finally:
            await file.close()
