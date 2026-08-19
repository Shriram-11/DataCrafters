"""In-memory repository implementation for analysis jobs."""
from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock

from domain_model.models.job import AnalysisJob, JobStatus


class InMemoryJobRepository:
    def __init__(self) -> None:
        self._store: dict[str, AnalysisJob] = {}
        self._lock = Lock()

    def create(self, job: AnalysisJob) -> None:
        with self._lock:
            self._store[job.job_id] = job

    def get(self, job_id: str) -> AnalysisJob | None:
        with self._lock:
            return self._store.get(job_id)

    def update_status(self, job_id: str, status: JobStatus) -> AnalysisJob | None:
        with self._lock:
            job = self._store.get(job_id)
            if not job:
                return None
            job.status = status
            job.updated_at = datetime.now(timezone.utc)
            self._store[job_id] = job
            return job

    def set_result(self, job_id: str, result: dict) -> AnalysisJob | None:
        with self._lock:
            job = self._store.get(job_id)
            if not job:
                return None
            job.result = result
            job.error = None
            job.status = JobStatus.completed
            job.updated_at = datetime.now(timezone.utc)
            self._store[job_id] = job
            return job

    def set_error(self, job_id: str, error: str) -> AnalysisJob | None:
        with self._lock:
            job = self._store.get(job_id)
            if not job:
                return None
            job.error = error
            job.status = JobStatus.failed
            job.updated_at = datetime.now(timezone.utc)
            self._store[job_id] = job
            return job
