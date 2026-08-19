"""Dataset domain models."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Dataset(BaseModel):
    dataset_id: str
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DatasetVersion(BaseModel):
    version_id: str
    dataset_id: str
    version_number: int
    source_filename: str
    storage_path: str
    rows: int
    columns: int
    column_names: list[str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
