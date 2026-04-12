"""DTOs for quality endpoint."""
from typing import Any

from pydantic import BaseModel


class QualityResponse(BaseModel):
    success: bool
    message: str
    data: dict[str, Any]
