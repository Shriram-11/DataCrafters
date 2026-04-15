"""Exploration analysis module."""
from __future__ import annotations

import pandas as pd


def build_correlation(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return {
            "columns": [],
            "matrix": [],
            "top_correlations": [],
        }

    corr = numeric_df.corr().fillna(0)
    columns = corr.columns.tolist()
    matrix = [[float(v) for v in row] for row in corr.values.tolist()]

    pairs = []
    for i, c1 in enumerate(columns):
        for j, c2 in enumerate(columns):
            if i >= j:
                continue
            coeff = float(corr.iloc[i, j])
            pairs.append({"left": c1, "right": c2, "coefficient": coeff, "absolute": abs(coeff)})
    pairs.sort(key=lambda item: item["absolute"], reverse=True)

    return {
        "columns": columns,
        "matrix": matrix,
        "top_correlations": pairs[:10],
    }


def build_univariate(df: pd.DataFrame) -> dict:
    numeric_stats = {}
    for col in df.select_dtypes(include=["number"]).columns:
        series = df[col].dropna()
        if series.empty:
            continue
        numeric_stats[col] = {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)) if series.count() > 1 else 0.0,
            "min": float(series.min()),
            "p25": float(series.quantile(0.25)),
            "p50": float(series.quantile(0.50)),
            "p75": float(series.quantile(0.75)),
            "max": float(series.max()),
            "skewness": float(series.skew()) if series.count() > 2 else 0.0,
            "kurtosis": float(series.kurt()) if series.count() > 3 else 0.0,
        }

    categorical_stats = {}
    for col in df.select_dtypes(exclude=["number"]).columns:
        series = df[col].fillna("<NULL>")
        value_counts = series.value_counts(dropna=False)
        top_values = [
            {"value": str(idx), "count": int(cnt)}
            for idx, cnt in value_counts.head(10).items()
        ]
        categorical_stats[col] = {
            "count": int(series.count()),
            "unique": int(series.nunique(dropna=False)),
            "top_values": top_values,
        }

    return {
        "numeric": numeric_stats,
        "categorical": categorical_stats,
    }


def build_recommendations(df: pd.DataFrame) -> dict:
    recommendations = {}
    row_count = max(int(df.shape[0]), 1)

    for col in df.columns:
        series = df[col]
        recs = []

        missing_pct = float(series.isnull().mean() * 100)
        if missing_pct >= 30:
            recs.append("High missingness: consider dropping column or advanced imputation")
        elif missing_pct > 0:
            recs.append("Has missing values: apply median/mode imputation")

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if clean.count() > 3:
                skewness = float(clean.skew())
                if abs(skewness) > 1:
                    recs.append("Highly skewed: consider log/sqrt transform")

                q1 = clean.quantile(0.25)
                q3 = clean.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    outliers = clean[(clean < (q1 - 1.5 * iqr)) | (clean > (q3 + 1.5 * iqr))]
                    outlier_rate = float((outliers.count() / row_count) * 100)
                    if outlier_rate > 5:
                        recs.append("Contains many outliers: cap/winsorize or robust-scale")
        else:
            unique_ratio = float(series.nunique(dropna=False) / row_count)
            if unique_ratio > 0.6:
                recs.append("High-cardinality categorical: consider frequency/target encoding")
            elif 1 < series.nunique(dropna=False) <= 12:
                recs.append("Low-cardinality categorical: one-hot encoding is suitable")

        if not recs:
            recs.append("No immediate issues detected")

        recommendations[col] = recs

    return {
        "column_recommendations": recommendations,
    }
