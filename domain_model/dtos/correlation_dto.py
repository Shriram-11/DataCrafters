"""DTOs for correlation endpoint."""
from typing import Any

from pydantic import BaseModel


class CorrelationResponse(BaseModel):
    success: bool
    message: str
    data: dict[str, Any]
