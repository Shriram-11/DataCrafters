"""DTOs for asynchronous jobs."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from domain_model.models.job import JobStatus


class JobAcceptedData(BaseModel):
    job_id: str
    status: JobStatus


class JobAcceptedResponse(BaseModel):
    success: bool
    message: str
    data: JobAcceptedData


class JobStatusData(BaseModel):
    job_id: str
    job_type: str
    filename: str
    status: JobStatus
    created_at: str
    updated_at: str
    result: dict[str, Any] | None = None
    error: str | None = None


class JobStatusResponse(BaseModel):
    success: bool
    message: str
    data: JobStatusData
