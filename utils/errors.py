"""Custom exception definitions."""
from typing import Any


class DataCraftersException(Exception):
    """Base exception for all app errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class CSVFormatError(DataCraftersException):
    """Raised when CSV format is invalid."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=400, details=details)


class FileSizeError(DataCraftersException):
    """Raised when uploaded file exceeds size limits."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=413, details=details)


class ProcessingError(DataCraftersException):
    """Raised when CSV processing fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=422, details=details)
