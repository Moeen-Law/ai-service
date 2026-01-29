"""
FastAPI Application Factory

Creates and configures the FastAPI application instance.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.middleware import RequestIDMiddleware
from app.api.routes import health, tasks
from app.api.schemas.common import ErrorDetail, ErrorResponse
from app.api.schemas.responses import TaskErrorResponse
from app.infrastructure.config.settings import get_settings
from app.infrastructure.logging.logger import configure_logging, get_logger

# Configure logging on module import
settings = get_settings()
configure_logging(
    level="DEBUG" if settings.DEBUG else "INFO",
    json_format=settings.is_production,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    logger.info(
        "Application starting",
        service=settings.SERVICE_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        debug=settings.DEBUG,
    )

    yield

    # Shutdown
    logger.info("Application shutting down", service=settings.SERVICE_NAME)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="AI Service - Legal Tech Platform",
        description=(
            "A clean, scalable FastAPI backend for AI task orchestration "
            "in a legal-tech microservices system."
        ),
        version=settings.VERSION,
        docs_url="/docs" if settings.ENABLE_DOCS else None,
        redoc_url="/redoc" if settings.ENABLE_DOCS else None,
        openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
        lifespan=lifespan,
    )

    # Register middleware
    register_middleware(app)

    # Register exception handlers
    register_exception_handlers(app)

    # Register routes
    register_routes(app)

    return app


def register_middleware(app: FastAPI) -> None:
    """
    Register application middleware.

    Middleware is applied in reverse order (last added = first executed).
    """
    # Request ID middleware - extracts/generates request IDs
    app.add_middleware(RequestIDMiddleware)


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register global exception handlers.

    Provides consistent error response format across the API.
    """

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Handle Pydantic validation errors."""
        details = []
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            details.append(
                ErrorDetail(
                    field=field,
                    message=error["msg"],
                )
            )

        logger.warning(
            "Validation error",
            path=str(request.url.path),
            errors=[d.model_dump() for d in details],
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=TaskErrorResponse(
                error=ErrorResponse(
                    code="VALIDATION_ERROR",
                    message="Invalid request data",
                    details=details,
                )
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        settings = get_settings()

        logger.exception(
            "Unhandled exception",
            path=str(request.url.path),
            exception_type=type(exc).__name__,
        )

        # In development, include exception details
        message = (
            str(exc) if settings.is_development else "An unexpected error occurred"
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=TaskErrorResponse(
                error=ErrorResponse(
                    code="INTERNAL_ERROR",
                    message=message,
                )
            ).model_dump(),
        )


def register_routes(app: FastAPI) -> None:
    """
    Register API routes.

    All routes are organized under versioned prefixes.
    """
    settings = get_settings()

    # Health check routes (unversioned)
    app.include_router(health.router)

    # V1 API routes
    app.include_router(
        tasks.router,
        prefix=settings.API_V1_PREFIX,
    )


# Create the application instance
app = create_app()
