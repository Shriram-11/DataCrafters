"""DTOs for dataset management endpoints."""
from pydantic import BaseModel


class DatasetVersionData(BaseModel):
    version_id: str
    dataset_id: str
    version_number: int
    source_filename: str
    rows: int
    columns: int
    column_names: list[str]
    created_at: str


class DatasetData(BaseModel):
    dataset_id: str
    name: str
    created_at: str
    updated_at: str
    latest_version: DatasetVersionData | None = None


class DatasetUploadResponse(BaseModel):
    success: bool
    message: str
    data: DatasetData


class DatasetListResponse(BaseModel):
    success: bool
    message: str
    data: list[DatasetData]


class DatasetDetailResponse(BaseModel):
    success: bool
    message: str
    data: DatasetData


class DatasetVersionListResponse(BaseModel):
    success: bool
    message: str
    data: list[DatasetVersionData]
