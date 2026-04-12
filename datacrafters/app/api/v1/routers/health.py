"""Health check endpoint."""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
import structlog

from datacrafters.app.core.config import settings
from datacrafters.app.core.tempfiles import TempFileManager

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["health"])


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    app_name: str
    version: str
    debug: bool
    temp_dir_size_mb: float
    max_upload_size_mb: int


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns service status and configuration metrics",
)
def health_check() -> HealthResponse:
    """Check service health and return diagnostics."""
    temp_size = TempFileManager.get_directory_size_mb()
    
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version="0.1.0",
        debug=settings.debug,
        temp_dir_size_mb=temp_size,
        max_upload_size_mb=settings.max_upload_size_mb,
    )
