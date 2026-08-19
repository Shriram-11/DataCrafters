"""Dataset service for metadata and versioning workflows."""
from __future__ import annotations

import io
import uuid
from pathlib import Path

import pandas as pd
from fastapi import UploadFile

from domain_model.models.dataset import Dataset, DatasetVersion
from domain_model.repositories.dataset_repository_interface import DatasetRepositoryInterface
from domain_model.repositories.sqlalchemy_dataset_repository import SQLAlchemyDatasetRepository
from settings import settings
from utils.errors import CSVFormatError, NotFoundError, ProcessingError


class DatasetService:
    _repo: DatasetRepositoryInterface | None = None

    @classmethod
    def _get_repo(cls) -> DatasetRepositoryInterface:
        if cls._repo is None:
            cls._repo = SQLAlchemyDatasetRepository()
        return cls._repo

    @classmethod
    async def upload_dataset(cls, file: UploadFile, dataset_name: str | None = None) -> dict:
        if not file.filename:
            raise CSVFormatError("Missing filename")
        if not file.filename.lower().endswith(".csv"):
            raise CSVFormatError("Only .csv files are supported")

        content = await file.read()
        await file.close()
        if not content:
            raise CSVFormatError("Uploaded CSV is empty")

        try:
            df = pd.read_csv(io.BytesIO(content))
            if df.empty:
                raise CSVFormatError("Uploaded CSV has no rows")
        except CSVFormatError:
            raise
        except Exception as exc:
            raise ProcessingError(f"Failed to parse CSV: {str(exc)}") from exc

        repo = cls._get_repo()
        dataset_id = str(uuid.uuid4())
        dataset = Dataset(
            dataset_id=dataset_id,
            name=(dataset_name or Path(file.filename).stem),
        )
        repo.create_dataset(dataset)

        version_number = 1
        version = cls._create_dataset_version(
            dataset_id=dataset_id,
            version_number=version_number,
            source_filename=file.filename,
            content=content,
            df=df,
        )
        repo.create_version(version)

        return {
            "success": True,
            "message": "Dataset uploaded successfully",
            "data": cls._dataset_to_dict(dataset, version),
        }

    @classmethod
    async def add_dataset_version(cls, dataset_id: str, file: UploadFile) -> dict:
        if not file.filename:
            raise CSVFormatError("Missing filename")
        if not file.filename.lower().endswith(".csv"):
            raise CSVFormatError("Only .csv files are supported")

        repo = cls._get_repo()
        dataset = repo.get_dataset(dataset_id)
        if dataset is None:
            raise NotFoundError("Dataset not found", details={"dataset_id": dataset_id})

        content = await file.read()
        await file.close()
        if not content:
            raise CSVFormatError("Uploaded CSV is empty")

        try:
            df = pd.read_csv(io.BytesIO(content))
            if df.empty:
                raise CSVFormatError("Uploaded CSV has no rows")
        except CSVFormatError:
            raise
        except Exception as exc:
            raise ProcessingError(f"Failed to parse CSV: {str(exc)}") from exc

        version_number = repo.count_versions(dataset_id) + 1
        version = cls._create_dataset_version(
            dataset_id=dataset_id,
            version_number=version_number,
            source_filename=file.filename,
            content=content,
            df=df,
        )
        repo.create_version(version)
        latest = repo.get_latest_version(dataset_id)

        return {
            "success": True,
            "message": "Dataset version uploaded successfully",
            "data": cls._dataset_to_dict(dataset, latest),
        }

    @classmethod
    def list_datasets(cls) -> dict:
        repo = cls._get_repo()
        datasets = repo.list_datasets()
        data = []
        for dataset in datasets:
            latest = repo.get_latest_version(dataset.dataset_id)
            data.append(cls._dataset_to_dict(dataset, latest))
        return {
            "success": True,
            "message": "Datasets fetched successfully",
            "data": data,
        }

    @classmethod
    def get_dataset(cls, dataset_id: str) -> dict:
        repo = cls._get_repo()
        dataset = repo.get_dataset(dataset_id)
        if dataset is None:
            raise NotFoundError("Dataset not found", details={"dataset_id": dataset_id})
        latest = repo.get_latest_version(dataset_id)
        return {
            "success": True,
            "message": "Dataset fetched successfully",
            "data": cls._dataset_to_dict(dataset, latest),
        }

    @classmethod
    def list_dataset_versions(cls, dataset_id: str) -> dict:
        repo = cls._get_repo()
        dataset = repo.get_dataset(dataset_id)
        if dataset is None:
            raise NotFoundError("Dataset not found", details={"dataset_id": dataset_id})
        versions = repo.list_versions(dataset_id)
        return {
            "success": True,
            "message": "Dataset versions fetched successfully",
            "data": [cls._version_to_dict(version) for version in versions],
        }

    @staticmethod
    def _create_dataset_version(
        dataset_id: str,
        version_number: int,
        source_filename: str,
        content: bytes,
        df: pd.DataFrame,
    ) -> DatasetVersion:
        version_id = str(uuid.uuid4())
        storage_path = DatasetService._persist_uploaded_file(
            dataset_id=dataset_id,
            version_number=version_number,
            filename=source_filename,
            content=content,
        )
        return DatasetVersion(
            version_id=version_id,
            dataset_id=dataset_id,
            version_number=version_number,
            source_filename=source_filename,
            storage_path=storage_path,
            rows=int(df.shape[0]),
            columns=int(df.shape[1]),
            column_names=[str(column) for column in df.columns.tolist()],
        )

    @staticmethod
    def _persist_uploaded_file(dataset_id: str, version_number: int, filename: str, content: bytes) -> str:
        safe_name = Path(filename).name
        base_dir = Path(settings.temp_directory) / "datasets" / dataset_id / f"v{version_number}"
        base_dir.mkdir(parents=True, exist_ok=True)
        full_path = base_dir / safe_name
        full_path.write_bytes(content)
        return str(full_path.resolve())

    @staticmethod
    def _dataset_to_dict(dataset: Dataset, latest_version: DatasetVersion | None) -> dict:
        return {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "created_at": dataset.created_at.isoformat(),
            "updated_at": dataset.updated_at.isoformat(),
            "latest_version": DatasetService._version_to_dict(latest_version) if latest_version else None,
        }

    @staticmethod
    def _version_to_dict(version: DatasetVersion | None) -> dict | None:
        if version is None:
            return None
        return {
            "version_id": version.version_id,
            "dataset_id": version.dataset_id,
            "version_number": version.version_number,
            "source_filename": version.source_filename,
            "rows": version.rows,
            "columns": version.columns,
            "column_names": version.column_names,
            "created_at": version.created_at.isoformat(),
        }
