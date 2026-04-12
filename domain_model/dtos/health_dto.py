"""Health DTOs."""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    debug: bool
    temp_dir_size_mb: float
    max_upload_size_mb: int
