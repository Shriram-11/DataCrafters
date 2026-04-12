"""Analysis controller."""
from fastapi import APIRouter, File, UploadFile

from domain_model.dtos.analysis_dto import AnalysisResponse
from services.analysis_service import AnalysisService

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])


@router.post(
    "/csv",
    response_model=AnalysisResponse,
    summary="Run legacy EDA analysis on CSV",
)
async def analyze_csv(file: UploadFile = File(...)) -> AnalysisResponse:
    result = await AnalysisService.analyze_csv_upload(file)
    return AnalysisResponse(**result)
