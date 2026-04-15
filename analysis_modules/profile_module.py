"""Profile analysis module."""
from __future__ import annotations

import pandas as pd


def build_profile(df: pd.DataFrame) -> dict:
    total_rows = int(df.shape[0])
    null_count = df.isnull().sum().to_dict()
    null_percent = ((df.isnull().mean() * 100).round(3)).to_dict()
    unique_count = df.nunique(dropna=False).to_dict()

    numeric_df = df.select_dtypes(include=["number"])
    numeric_summary = {}
    if not numeric_df.empty:
        desc = numeric_df.describe().transpose().fillna(0)
        for col, row in desc.iterrows():
            numeric_summary[col] = {
                "count": float(row.get("count", 0)),
                "mean": float(row.get("mean", 0)),
                "std": float(row.get("std", 0)),
                "min": float(row.get("min", 0)),
                "p25": float(row.get("25%", 0)),
                "p50": float(row.get("50%", 0)),
                "p75": float(row.get("75%", 0)),
                "max": float(row.get("max", 0)),
            }

    return {
        "shape": {"rows": total_rows, "columns": int(df.shape[1])},
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_count": null_count,
        "null_percent": null_percent,
        "unique_count": unique_count,
        "numeric_summary": numeric_summary,
    }
