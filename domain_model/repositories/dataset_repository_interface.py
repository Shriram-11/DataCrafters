"""Repository interface for datasets and dataset versions."""
from __future__ import annotations

from typing import Protocol

from domain_model.models.dataset import Dataset, DatasetVersion


class DatasetRepositoryInterface(Protocol):
    def create_dataset(self, dataset: Dataset) -> None:
        ...

    def get_dataset(self, dataset_id: str) -> Dataset | None:
        ...

    def list_datasets(self) -> list[Dataset]:
        ...

    def get_latest_version(self, dataset_id: str) -> DatasetVersion | None:
        ...

    def list_versions(self, dataset_id: str) -> list[DatasetVersion]:
        ...

    def count_versions(self, dataset_id: str) -> int:
        ...

    def create_version(self, version: DatasetVersion) -> None:
        ...
