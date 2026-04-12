"""Health controller."""
from fastapi import APIRouter

from domain_model.dtos.health_dto import HealthResponse
from settings import settings
from utils.tempfiles import TempFileManager

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version="0.1.0",
        debug=settings.debug,
        temp_dir_size_mb=TempFileManager.get_directory_size_mb(),
        max_upload_size_mb=settings.max_upload_size_mb,
    )
