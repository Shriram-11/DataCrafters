"""Asynchronous job service using repository pattern."""
from __future__ import annotations

import io
from uuid import uuid4

import pandas as pd
from fastapi import BackgroundTasks, UploadFile

from analysis_modules.profile_module import build_profile
from domain_model.models.job import AnalysisJob, JobStatus
from domain_model.repositories.in_memory_job_repository import InMemoryJobRepository
from domain_model.repositories.job_repository_interface import JobRepositoryInterface
from domain_model.repositories.sqlalchemy_job_repository import SQLAlchemyJobRepository
from settings import settings
from utils.errors import CSVFormatError, NotFoundError


class JobService:
    """Orchestrates job lifecycle and background execution."""

    _repo: JobRepositoryInterface | None = None

    @classmethod
    def _get_repo(cls) -> JobRepositoryInterface:
        if cls._repo is None:
            if settings.job_repository_backend == "sqlalchemy":
                cls._repo = SQLAlchemyJobRepository()
            else:
                cls._repo = InMemoryJobRepository()
        return cls._repo

    @classmethod
    async def enqueue_profile_job(cls, file: UploadFile, background_tasks: BackgroundTasks) -> dict:
        if not file.filename:
            raise CSVFormatError("Missing filename")
        if not file.filename.lower().endswith(".csv"):
            raise CSVFormatError("Only .csv files are supported")

        content = await file.read()
        await file.close()
        if not content:
            raise CSVFormatError("Uploaded CSV is empty")

        job_id = str(uuid4())
        job = AnalysisJob(
            job_id=job_id,
            job_type="profile_csv",
            filename=file.filename,
            status=JobStatus.queued,
        )
        cls._get_repo().create(job)
        background_tasks.add_task(cls._run_profile_job, job_id, content)

        return {
            "success": True,
            "message": "Job queued",
            "data": {
                "job_id": job_id,
                "status": JobStatus.queued,
            },
        }

    @classmethod
    def get_job(cls, job_id: str) -> dict:
        job = cls._get_repo().get(job_id)
        if not job:
            raise NotFoundError("Job not found", details={"job_id": job_id})

        return {
            "success": True,
            "message": "Job fetched successfully",
            "data": {
                "job_id": job.job_id,
                "job_type": job.job_type,
                "filename": job.filename,
                "status": job.status,
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat(),
                "result": job.result,
                "error": job.error,
            },
        }

    @classmethod
    def _run_profile_job(cls, job_id: str, content: bytes) -> None:
        repo = cls._get_repo()
        repo.update_status(job_id, JobStatus.running)
        try:
            df = pd.read_csv(io.BytesIO(content))
            if df.empty:
                repo.set_error(job_id, "Uploaded CSV has no rows")
                return
            result = build_profile(df)
            repo.set_result(job_id, result)
        except Exception as exc:
            repo.set_error(job_id, f"Failed to process job: {str(exc)}")
