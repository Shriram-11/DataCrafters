"""CSV process endpoint."""
from pydantic import BaseModel
from fastapi import APIRouter, File, UploadFile

from datacrafters.app.services.csv_ingest_service import CsvIngestService


router = APIRouter(prefix="/api/v1/process", tags=["process"])


class ProcessCsvResponse(BaseModel):
    """Response schema for CSV processing endpoint."""

    status: str
    filename: str
    size_bytes: int
    storage_mode: str
    rows: int
    columns: int
    column_names: list[str]


@router.post(
    "/csv",
    response_model=ProcessCsvResponse,
    summary="Upload and process CSV",
    description="Processes uploaded CSV using memory or temp storage based on configured threshold.",
)
async def process_csv(file: UploadFile = File(...)) -> ProcessCsvResponse:
    """Upload and process a CSV file in an ephemeral workflow."""

    result = await CsvIngestService.process_upload(file)
    return ProcessCsvResponse(**result)
