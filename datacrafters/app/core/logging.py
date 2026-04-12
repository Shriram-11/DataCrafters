"""Structured logging setup."""
import logging
import sys
import structlog
from datacrafters.app.core.config import settings


def setup_logging():
    """Configure structured logging with structlog."""

    structlog.configure(
        processors=[
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # JSON formatter for structured logs
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
    )

    # Console handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.log_level.upper())

    return structlog.get_logger()


logger = setup_logging()
