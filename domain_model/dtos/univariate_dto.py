"""DTOs for univariate endpoint."""
from typing import Any

from pydantic import BaseModel


class UnivariateResponse(BaseModel):
    success: bool
    message: str
    data: dict[str, Any]
