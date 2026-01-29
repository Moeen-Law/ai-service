"""
AI Service Entry Point

Starts the FastAPI application using Uvicorn.
"""

import uvicorn

from app.infrastructure.config.settings import get_settings

if __name__ == "__main__":
    settings = get_settings()

    uvicorn.run(
        "app.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
