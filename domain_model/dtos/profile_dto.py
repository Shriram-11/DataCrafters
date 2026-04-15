"""DTOs for profile endpoint."""
from typing import Any

from pydantic import BaseModel


class ProfileResponse(BaseModel):
    success: bool
    message: str
    data: dict[str, Any]
