"""DTOs for analysis endpoint."""
from typing import Any

from pydantic import BaseModel


class AnalysisResponse(BaseModel):
    success: bool
    message: str
    data: dict[str, Any]
