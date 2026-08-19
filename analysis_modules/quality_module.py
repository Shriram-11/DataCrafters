"""Quality analysis module."""
from __future__ import annotations

import pandas as pd


def build_quality(df: pd.DataFrame) -> dict:
    total_rows = int(df.shape[0])
    total_columns = int(df.shape[1])
    total_cells = max(total_rows * total_columns, 1)

    duplicates_count = int(df.duplicated().sum())
    missing_count = int(df.isnull().sum().sum())
    completeness_score = round((1 - (missing_count / total_cells)) * 100, 3)

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
