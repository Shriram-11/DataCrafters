"""FastAPI app entrypoint."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from controllers.health_controller import router as health_router
from controllers.analysis_controller import router as analysis_router
from controllers.process_controller import router as process_router
from controllers.profile_controller import router as profile_router
from controllers.quality_controller import router as quality_router
from settings import settings
from utils.errors import DataCraftersException
from utils.logging import logger
from utils.tempfiles import TempFileManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("app_startup", app_name=settings.app_name,
                debug=settings.debug)
    TempFileManager.cleanup_directory()
    yield
    logger.info("app_shutdown")
    TempFileManager.cleanup_directory(max_age_seconds=0)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="Ephemeral CSV processing API",
        version="0.1.0",
        lifespan=lifespan,
        debug=settings.debug,
    )

    @app.exception_handler(DataCraftersException)
    async def datacrafters_exception_handler(request, exc: DataCraftersException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "details": exc.details},
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

    app.include_router(health_router)
    app.include_router(process_router)
    app.include_router(analysis_router)
    app.include_router(profile_router)
    app.include_router(quality_router)
    return app


app = create_app()
