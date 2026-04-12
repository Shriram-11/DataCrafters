"""FastAPI application factory and startup/shutdown logic."""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import structlog

from datacrafters.app.core.config import settings
from datacrafters.app.core.logging import logger
from datacrafters.app.core.errors import DataCraftersException
from datacrafters.app.core.tempfiles import TempFileManager
from datacrafters.app.api.v1.routers import health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle: startup and shutdown."""
    # Startup
    logger.info("app_startup", app_name=settings.app_name, debug=settings.debug)
    TempFileManager.cleanup_directory()
    
    yield
    
    # Shutdown
    logger.info("app_shutdown")
    TempFileManager.cleanup_directory(max_age_seconds=0)  # Clean all temp files


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title=settings.app_name,
        description="Ephemeral CSV processing API",
        version="0.1.0",
        lifespan=lifespan,
        debug=settings.debug,
    )
    
    # Exception handlers
    @app.exception_handler(DataCraftersException)
    async def datacrafters_exception_handler(request, exc: DataCraftersException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.message,
                "details": exc.details,
            },
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc: Exception):
        logger.error(
            "unhandled_exception",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "details": {} if not settings.debug else {"exception": str(exc)},
            },
        )
    
    # Include routers
    app.include_router(health.router)
    
    return app
