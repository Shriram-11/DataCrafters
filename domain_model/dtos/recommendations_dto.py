"""DTOs for recommendations endpoint."""
from typing import Any

from pydantic import BaseModel


class RecommendationsResponse(BaseModel):
    success: bool
    message: str
    data: dict[str, Any]
