"""Correlation controller."""
from fastapi import APIRouter, File, UploadFile

from domain_model.dtos.correlation_dto import CorrelationResponse
from services.exploration_service import ExplorationService

router = APIRouter(prefix="/api/v1/correlation", tags=["correlation"])


@router.post(
    "/csv",
    response_model=CorrelationResponse,
    summary="Compute correlation matrix",
)
async def correlation_csv(file: UploadFile = File(...)) -> CorrelationResponse:
    result = await ExplorationService.correlation_csv_upload(file)
    return CorrelationResponse(**result)
