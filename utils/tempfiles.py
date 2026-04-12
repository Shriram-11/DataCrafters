"""Temporary file utilities."""
import tempfile
import time
from pathlib import Path

import structlog

from settings import settings

logger = structlog.get_logger()


class TempFileManager:
    """Manages temporary files with automatic cleanup."""

    @staticmethod
    def get_temp_path(prefix: str = "dc_", suffix: str = ".csv") -> Path:
        temp_name = next(tempfile._get_candidate_names())
        return Path(settings.temp_directory) / f"{prefix}{temp_name}{suffix}"

    @staticmethod
    def cleanup_file(path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
                logger.debug("temp_file_deleted", path=str(path))
        except Exception as exc:  # pragma: no cover
            logger.warning("temp_file_cleanup_failed",
                           path=str(path), error=str(exc))

    @staticmethod
    def cleanup_directory(max_age_seconds: int | None = None) -> int:
        if max_age_seconds is None:
            max_age_seconds = settings.temp_file_ttl_seconds

        temp_dir = Path(settings.temp_directory)
        if not temp_dir.exists():
            return 0

        cutoff_time = time.time() - max_age_seconds
        deleted_count = 0

        try:
            for file_path in temp_dir.glob("dc_*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
                    logger.debug("stale_temp_file_deleted",
                                 path=str(file_path))
        except Exception as exc:  # pragma: no cover
            logger.warning("temp_directory_cleanup_failed", error=str(exc))

        return deleted_count

    @staticmethod
    def get_directory_size_mb() -> float:
        temp_dir = Path(settings.temp_directory)
        if not temp_dir.exists():
            return 0.0

        total_size = sum(
            f.stat().st_size for f in temp_dir.rglob("*") if f.is_file())
        return total_size / (1024 * 1024)
