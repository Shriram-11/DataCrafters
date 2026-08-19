"""Job domain model."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class AnalysisJob(BaseModel):
    """Represents an asynchronous analysis job lifecycle."""

    job_id: str
    job_type: str
    filename: str
    status: JobStatus = JobStatus.queued
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    result: dict[str, Any] | None = None
    error: str | None = None
