"""Univariate controller."""
from fastapi import APIRouter, File, UploadFile

from domain_model.dtos.univariate_dto import UnivariateResponse
from services.exploration_service import ExplorationService

router = APIRouter(prefix="/api/v1/univariate", tags=["univariate"])


@router.post(
    "/csv",
    response_model=UnivariateResponse,
    summary="Compute univariate statistics",
)
async def univariate_csv(file: UploadFile = File(...)) -> UnivariateResponse:
    result = await ExplorationService.univariate_csv_upload(file)
    return UnivariateResponse(**result)
