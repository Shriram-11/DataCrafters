"""Quality controller."""
from fastapi import APIRouter, File, UploadFile

from domain_model.dtos.quality_dto import QualityResponse
from services.quality_service import QualityService

router = APIRouter(prefix="/api/v1/quality", tags=["quality"])


@router.post(
    "/csv",
    response_model=QualityResponse,
    summary="Run quality checks",
)
async def quality_csv(file: UploadFile = File(...)) -> QualityResponse:
    result = await QualityService.quality_csv_upload(file)
    return QualityResponse(**result)
