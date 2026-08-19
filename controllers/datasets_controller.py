"""Dataset management controller."""
from fastapi import APIRouter, File, Form, UploadFile

from domain_model.dtos.datasets_dto import (
    DatasetDetailResponse,
    DatasetListResponse,
    DatasetUploadResponse,
    DatasetVersionListResponse,
)
from services.dataset_service import DatasetService

router = APIRouter(prefix="/api/v2/datasets", tags=["datasets"])


@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    summary="Upload a new dataset as version 1",
)
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: str | None = Form(None),
) -> DatasetUploadResponse:
    result = await DatasetService.upload_dataset(file=file, dataset_name=dataset_name)
    return DatasetUploadResponse(**result)


@router.post(
    "/{dataset_id}/versions/upload",
    response_model=DatasetUploadResponse,
    summary="Upload a new version for an existing dataset",
)
async def upload_dataset_version(
    dataset_id: str,
    file: UploadFile = File(...),
) -> DatasetUploadResponse:
    result = await DatasetService.add_dataset_version(dataset_id=dataset_id, file=file)
    return DatasetUploadResponse(**result)


@router.get(
    "",
    response_model=DatasetListResponse,
    summary="List all datasets",
)
def list_datasets() -> DatasetListResponse:
    result = DatasetService.list_datasets()
    return DatasetListResponse(**result)


@router.get(
    "/{dataset_id}",
    response_model=DatasetDetailResponse,
    summary="Get dataset by ID",
)
def get_dataset(dataset_id: str) -> DatasetDetailResponse:
    result = DatasetService.get_dataset(dataset_id)
    return DatasetDetailResponse(**result)


@router.get(
    "/{dataset_id}/versions",
    response_model=DatasetVersionListResponse,
    summary="List versions for a dataset",
)
def list_dataset_versions(dataset_id: str) -> DatasetVersionListResponse:
    result = DatasetService.list_dataset_versions(dataset_id)
    return DatasetVersionListResponse(**result)
