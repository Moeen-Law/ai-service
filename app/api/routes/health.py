"""
Health Check Routes

Provides health and readiness endpoints for the service.
"""

from fastapi import APIRouter

from app.api.schemas.responses import HealthResponse
from app.infrastructure.config.settings import get_settings

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns the service health status.",
)
async def health_check() -> HealthResponse:
    """
    Check service health.

    Returns basic health information including service name and version.
    """
    settings = get_settings()

    return HealthResponse(
        status="healthy",
        service=settings.SERVICE_NAME,
        version=settings.VERSION,
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness Check",
    description="Returns the service readiness status.",
)
async def readiness_check() -> HealthResponse:
    """
    Check service readiness.

    Verifies the service is ready to accept requests.
    Future: Check database connections, external services, etc.
    """
    settings = get_settings()

    # TODO: Add actual readiness checks (database, external services)

    return HealthResponse(
        status="healthy",
        service=settings.SERVICE_NAME,
        version=settings.VERSION,
    )
