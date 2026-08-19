"""Application settings."""
import os
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App-wide settings loaded from environment variables."""

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
    )

    app_name: str = "DataCrafters"
    debug: bool = False
    max_upload_size_mb: int = 100
    in_memory_threshold_mb: int = 10
    temp_directory: str = os.path.join(os.getcwd(), "temp")
    temp_file_ttl_seconds: int = 900
    log_level: str = "INFO"
    database_url: str = "sqlite:///./datacrafters.db"
    job_repository_backend: Literal["memory", "sqlalchemy"] = "sqlalchemy"

    @field_validator("debug", mode="before")
    @classmethod
    def coerce_debug(cls, value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on", "debug", "dev"}:
            return True
        if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
            return False
        return False


settings = Settings()
Path(settings.temp_directory).mkdir(parents=True, exist_ok=True)
