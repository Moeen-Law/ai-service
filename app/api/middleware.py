"""
Request ID Middleware

Extracts or generates request ID and propagates it through the request context.
"""

from typing import Callable
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.infrastructure.logging.logger import get_logger, set_request_id

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that extracts or generates a request ID.

    The request ID is:
    1. Extracted from X-Request-ID header if present
    2. Generated as a UUID if not present
    3. Propagated to the response headers
    4. Available in logging context throughout the request
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        # Extract or generate request ID
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid4())

        # Set in context for logging
        set_request_id(request_id)

        # Add to request state for access in handlers
        request.state.request_id = request_id

        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else None,
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.exception(
                "Request failed with exception",
                method=request.method,
                path=request.url.path,
            )
            raise

        # Add request ID to response headers
        response.headers[REQUEST_ID_HEADER] = request_id

        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
        )

        return response


__all__ = ["RequestIDMiddleware", "REQUEST_ID_HEADER"]
