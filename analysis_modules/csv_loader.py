"""CSV loading helpers for analysis workflows."""
from __future__ import annotations

import io

import pandas as pd
from fastapi import UploadFile

from utils.errors import CSVFormatError, ProcessingError


async def load_csv_upload(file: UploadFile) -> pd.DataFrame:
    """Validate and parse uploaded CSV into a DataFrame."""
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
        return df
    except CSVFormatError:
        raise
    except Exception as exc:
        raise ProcessingError(f"Failed to parse CSV: {str(exc)}") from exc
    finally:
        await file.close()
