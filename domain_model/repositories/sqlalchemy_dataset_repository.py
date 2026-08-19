"""SQLAlchemy-backed dataset repository."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import desc, select

from domain_model.models.dataset import Dataset, DatasetVersion
from infrastructure.db.models import DatasetORM, DatasetVersionORM
from infrastructure.db.session import SessionLocal, init_db


class SQLAlchemyDatasetRepository:
    def __init__(self) -> None:
        init_db()

    def create_dataset(self, dataset: Dataset) -> None:
        with SessionLocal() as session:
            orm = DatasetORM(
                dataset_id=dataset.dataset_id,
                name=dataset.name,
                created_at=dataset.created_at,
                updated_at=dataset.updated_at,
            )
            session.add(orm)
            session.commit()

    def get_dataset(self, dataset_id: str) -> Dataset | None:
        with SessionLocal() as session:
            orm = session.get(DatasetORM, dataset_id)
            if orm is None:
                return None
            return self._to_dataset(orm)

    def list_datasets(self) -> list[Dataset]:
        with SessionLocal() as session:
            query = select(DatasetORM).order_by(desc(DatasetORM.updated_at))
            rows = session.execute(query).scalars().all()
            return [self._to_dataset(row) for row in rows]

    def get_latest_version(self, dataset_id: str) -> DatasetVersion | None:
        with SessionLocal() as session:
            query = (
                select(DatasetVersionORM)
                .where(DatasetVersionORM.dataset_id == dataset_id)
                .order_by(desc(DatasetVersionORM.version_number))
                .limit(1)
            )
            row = session.execute(query).scalars().first()
            return self._to_version(row) if row else None

    def list_versions(self, dataset_id: str) -> list[DatasetVersion]:
        with SessionLocal() as session:
            query = (
                select(DatasetVersionORM)
                .where(DatasetVersionORM.dataset_id == dataset_id)
                .order_by(desc(DatasetVersionORM.version_number))
            )
            rows = session.execute(query).scalars().all()
            return [self._to_version(row) for row in rows]

    def count_versions(self, dataset_id: str) -> int:
        with SessionLocal() as session:
            query = select(DatasetVersionORM).where(
                DatasetVersionORM.dataset_id == dataset_id
            )
            return len(session.execute(query).scalars().all())

    def create_version(self, version: DatasetVersion) -> None:
        with SessionLocal() as session:
            orm = DatasetVersionORM(
                version_id=version.version_id,
                dataset_id=version.dataset_id,
                version_number=version.version_number,
                source_filename=version.source_filename,
                storage_path=version.storage_path,
                rows=version.rows,
                columns=version.columns,
                column_names_json=json.dumps(version.column_names),
                created_at=version.created_at,
            )
            session.add(orm)
            dataset = session.get(DatasetORM, version.dataset_id)
            if dataset is not None:
                dataset.updated_at = datetime.now(timezone.utc)
            session.commit()

    @staticmethod
    def _to_dataset(orm: DatasetORM) -> Dataset:
        return Dataset(
            dataset_id=orm.dataset_id,
            name=orm.name,
            created_at=orm.created_at,
            updated_at=orm.updated_at,
        )

    @staticmethod
    def _to_version(orm: DatasetVersionORM) -> DatasetVersion:
        return DatasetVersion(
            version_id=orm.version_id,
            dataset_id=orm.dataset_id,
            version_number=orm.version_number,
            source_filename=orm.source_filename,
            storage_path=orm.storage_path,
            rows=orm.rows,
            columns=orm.columns,
            column_names=json.loads(orm.column_names_json),
            created_at=orm.created_at,
        )
