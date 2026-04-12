"""Process controller."""
from fastapi import APIRouter, File, UploadFile

from domain_model.dtos.process_dto import ProcessCsvResponse
from services.csv_ingest_service import CsvIngestService

router = APIRouter(prefix="/api/v1/process", tags=["process"])


@router.post(
    "/csv",
    response_model=ProcessCsvResponse,
    summary="Upload and process CSV",
)
async def process_csv(file: UploadFile = File(...)) -> ProcessCsvResponse:
    result = await CsvIngestService.process_upload(file)
    return ProcessCsvResponse(**result)
