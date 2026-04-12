"""Application entry point."""
if __name__ == "__main__":
    import uvicorn
    from datacrafters.app.core.config import settings
    from datacrafters.app.main import create_app

    app = create_app()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
    )
