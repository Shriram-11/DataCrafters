"""Application configuration management."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """App-wide settings loaded from environment variables."""

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
    )

    # API
    app_name: str = "datacrafters"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # File handling
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))
    in_memory_threshold_mb: int = int(
        os.getenv("IN_MEMORY_THRESHOLD_MB", "10"))
    temp_directory: str = os.getenv(
        "TEMP_DIRECTORY", os.path.join(os.getcwd(), "temp"))
    temp_file_ttl_seconds: int = int(
        os.getenv("TEMP_FILE_TTL_SECONDS", "900"))  # 15 min

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()

# Ensure temp directory exists at startup
Path(settings.temp_directory).mkdir(parents=True, exist_ok=True)
