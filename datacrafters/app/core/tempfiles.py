"""Temporary file management and cleanup utilities."""
import os
import shutil
import tempfile
import time
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timedelta

import structlog

from datacrafters.app.core.config import settings
from datacrafters.app.core.errors import TemporaryFileError


logger = structlog.get_logger()


class TempFileManager:
    """Manages temporary files with automatic cleanup."""

    @staticmethod
    def get_temp_path(prefix: str = "dc_", suffix: str = ".csv") -> Path:
        """Create a new temporary file path in the configured temp directory."""
        temp_name = next(tempfile._get_candidate_names())
        filename = f"{prefix}{temp_name}{suffix}"
        return Path(settings.temp_directory) / filename

    @staticmethod
    @contextmanager
    def managed_temp_file(prefix: str = "dc_", suffix: str = ".csv"):
        """Context manager for temporary file with guaranteed cleanup.

        Usage:
            with TempFileManager.managed_temp_file() as temp_path:
                # write/process file
                pass
            # file is deleted automatically
        """
        temp_path = TempFileManager.get_temp_path(prefix=prefix, suffix=suffix)
        try:
            yield temp_path
        finally:
            TempFileManager.cleanup_file(temp_path)

    @staticmethod
    def cleanup_file(path: Path) -> None:
        """Delete a file, handling errors gracefully."""
        try:
            if path.exists():
                path.unlink()
                logger.debug("temp_file_deleted", path=str(path))
        except Exception as e:
            logger.warning("temp_file_cleanup_failed",
                           path=str(path), error=str(e))

    @staticmethod
    def cleanup_directory(max_age_seconds: int = None) -> int:
        """Clean up old temporary files in the temp directory.

        Args:
            max_age_seconds: Files older than this are deleted. Defaults to TTL setting.

        Returns:
            Number of files deleted.
        """
        if max_age_seconds is None:
            max_age_seconds = settings.temp_file_ttl_seconds

        temp_dir = Path(settings.temp_directory)
        if not temp_dir.exists():
            return 0

        cutoff_time = time.time() - max_age_seconds
        deleted_count = 0

        try:
            for file_path in temp_dir.glob("dc_*"):
                if file_path.is_file():
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug("stale_temp_file_deleted",
                                     path=str(file_path))
        except Exception as e:
            logger.warning("temp_directory_cleanup_failed", error=str(e))

        return deleted_count

    @staticmethod
    def get_directory_size_mb() -> float:
        """Get total size of temp directory in MB."""
        temp_dir = Path(settings.temp_directory)
        if not temp_dir.exists():
            return 0.0

        total_size = sum(
            f.stat().st_size for f in temp_dir.rglob("*") if f.is_file())
        return total_size / (1024 * 1024)
