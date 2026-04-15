"""Recommendations controller."""
from fastapi import APIRouter, File, UploadFile

from domain_model.dtos.recommendations_dto import RecommendationsResponse
from services.exploration_service import ExplorationService

router = APIRouter(prefix="/api/v1/recommendations", tags=["recommendations"])


@router.post(
    "/csv",
    response_model=RecommendationsResponse,
    summary="Generate cleaning and feature recommendations",
)
async def recommendations_csv(file: UploadFile = File(...)) -> RecommendationsResponse:
    result = await ExplorationService.recommendations_csv_upload(file)
    return RecommendationsResponse(**result)
