"""Profile controller."""
from fastapi import APIRouter, File, UploadFile

from domain_model.dtos.profile_dto import ProfileResponse
from services.profile_service import ProfileService

router = APIRouter(prefix="/api/v1/profile", tags=["profile"])


@router.post(
    "/csv",
    response_model=ProfileResponse,
    summary="Generate dataset profile",
)
async def profile_csv(file: UploadFile = File(...)) -> ProfileResponse:
    result = await ProfileService.profile_csv_upload(file)
    return ProfileResponse(**result)
