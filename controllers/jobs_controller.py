"""Jobs controller for asynchronous analysis workflows."""
from fastapi import APIRouter, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse

from domain_model.dtos.jobs_dto import JobAcceptedResponse, JobStatusResponse
from services.job_service import JobService

router = APIRouter(prefix="/api/v2/jobs", tags=["jobs"])


@router.post(
    "/profile/csv",
    response_model=JobAcceptedResponse,
    summary="Queue profile analysis for CSV",
    status_code=202,
)
async def enqueue_profile_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> JSONResponse:
    result = await JobService.enqueue_profile_job(file, background_tasks)
    return JSONResponse(status_code=202, content=JobAcceptedResponse(**result).model_dump(mode="json"))


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Fetch asynchronous job status",
)
def get_job(job_id: str) -> JobStatusResponse:
    result = JobService.get_job(job_id)
    return JobStatusResponse(**result)
