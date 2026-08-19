"""SQLAlchemy-backed job repository."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from domain_model.models.job import AnalysisJob, JobStatus
from infrastructure.db.models import AnalysisJobORM
from infrastructure.db.session import SessionLocal, init_db


class SQLAlchemyJobRepository:
    def __init__(self) -> None:
        init_db()

    def create(self, job: AnalysisJob) -> None:
        with SessionLocal() as session:
            orm = AnalysisJobORM(
                job_id=job.job_id,
                job_type=job.job_type,
                filename=job.filename,
                status=job.status.value,
                created_at=job.created_at,
                updated_at=job.updated_at,
                result_json=json.dumps(job.result) if job.result is not None else None,
                error=job.error,
            )
            session.add(orm)
            session.commit()

    def get(self, job_id: str) -> AnalysisJob | None:
        with SessionLocal() as session:
            orm = session.get(AnalysisJobORM, job_id)
            if orm is None:
                return None
            return self._to_domain(orm)

    def update_status(self, job_id: str, status: JobStatus) -> AnalysisJob | None:
        with SessionLocal() as session:
            orm = session.get(AnalysisJobORM, job_id)
            if orm is None:
                return None
            orm.status = status.value
            orm.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(orm)
            return self._to_domain(orm)

    def set_result(self, job_id: str, result: dict) -> AnalysisJob | None:
        with SessionLocal() as session:
            orm = session.get(AnalysisJobORM, job_id)
            if orm is None:
                return None
            orm.result_json = json.dumps(result)
            orm.error = None
            orm.status = JobStatus.completed.value
            orm.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(orm)
            return self._to_domain(orm)

    def set_error(self, job_id: str, error: str) -> AnalysisJob | None:
        with SessionLocal() as session:
            orm = session.get(AnalysisJobORM, job_id)
            if orm is None:
                return None
            orm.error = error
            orm.status = JobStatus.failed.value
            orm.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(orm)
            return self._to_domain(orm)

    @staticmethod
    def _to_domain(orm: AnalysisJobORM) -> AnalysisJob:
        parsed_result = json.loads(orm.result_json) if orm.result_json else None
        return AnalysisJob(
            job_id=orm.job_id,
            job_type=orm.job_type,
            filename=orm.filename,
            status=JobStatus(orm.status),
            created_at=orm.created_at,
            updated_at=orm.updated_at,
            result=parsed_result,
            error=orm.error,
        )
