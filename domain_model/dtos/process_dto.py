"""Process DTOs."""
from pydantic import BaseModel


class ProcessCsvResponse(BaseModel):
    status: str
    filename: str
    size_bytes: int
    storage_mode: str
    rows: int
    columns: int
    column_names: list[str]
