"""Analysis V2 controller with grouped analysis endpoints."""
from fastapi import APIRouter, File, UploadFile

from domain_model.dtos.correlation_dto import CorrelationResponse
from domain_model.dtos.profile_dto import ProfileResponse
from domain_model.dtos.quality_dto import QualityResponse
from domain_model.dtos.recommendations_dto import RecommendationsResponse
from domain_model.dtos.univariate_dto import UnivariateResponse
from services.analysis_service_v2 import AnalysisServiceV2

router = APIRouter(prefix="/api/v2/analysis", tags=["analysis-v2"])


@router.post("/profile/csv", response_model=ProfileResponse, summary="Generate dataset profile")
async def profile_csv(file: UploadFile = File(...)) -> ProfileResponse:
    result = await AnalysisServiceV2.profile(file)
    return ProfileResponse(**result)


@router.post("/quality/csv", response_model=QualityResponse, summary="Run quality checks")
async def quality_csv(file: UploadFile = File(...)) -> QualityResponse:
    result = await AnalysisServiceV2.quality(file)
    return QualityResponse(**result)


@router.post("/correlation/csv", response_model=CorrelationResponse, summary="Compute correlation matrix")
async def correlation_csv(file: UploadFile = File(...)) -> CorrelationResponse:
    result = await AnalysisServiceV2.correlation(file)
    return CorrelationResponse(**result)


@router.post("/univariate/csv", response_model=UnivariateResponse, summary="Compute univariate statistics")
async def univariate_csv(file: UploadFile = File(...)) -> UnivariateResponse:
    result = await AnalysisServiceV2.univariate(file)
    return UnivariateResponse(**result)


@router.post(
    "/recommendations/csv",
    response_model=RecommendationsResponse,
    summary="Generate cleaning and feature recommendations",
)
async def recommendations_csv(file: UploadFile = File(...)) -> RecommendationsResponse:
    result = await AnalysisServiceV2.recommendations(file)
    return RecommendationsResponse(**result)
