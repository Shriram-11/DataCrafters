"""Repository interface for job persistence."""
from __future__ import annotations

from typing import Protocol

from domain_model.models.job import AnalysisJob, JobStatus


class JobRepositoryInterface(Protocol):
    def create(self, job: AnalysisJob) -> None:
        ...

    def get(self, job_id: str) -> AnalysisJob | None:
        ...

    def update_status(self, job_id: str, status: JobStatus) -> AnalysisJob | None:
        ...

    def set_result(self, job_id: str, result: dict) -> AnalysisJob | None:
        ...

    def set_error(self, job_id: str, error: str) -> AnalysisJob | None:
        ...
