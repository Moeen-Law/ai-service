"""
Structured Logging Configuration

Provides structured, JSON-formatted logging for observability.
Uses structlog for structured logging with request ID propagation.
"""

import logging
import sys
from contextvars import ContextVar
from typing import Any, Dict, Optional
from uuid import uuid4

# Context variable for request ID propagation
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    """Get current request ID from context."""
    return request_id_var.get()


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set request ID in context. Generates one if not provided."""
    rid = request_id or str(uuid4())
    request_id_var.set(rid)
    return rid


class StructuredLogger:
    """
    Structured logger that outputs JSON-formatted logs.

    Includes request ID propagation and contextual information.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._logger = logging.getLogger(name)

    def _build_event(
        self,
        message: str,
        level: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build structured log event."""
        event = {
            "message": message,
            "level": level,
            "logger": self.name,
        }

        # Add request ID if available
        request_id = get_request_id()
        if request_id:
            event["request_id"] = request_id

        # Add extra context
        event.update(kwargs)

        return event

    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """Internal log method."""
        level_name = logging.getLevelName(level).lower()
        event = self._build_event(message, level_name, **kwargs)
        self._logger.log(level, event)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def warn(self, message: str, **kwargs: Any) -> None:
        """Alias for warning."""
        self.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        import traceback

        kwargs["traceback"] = traceback.format_exc()
        self._log(logging.ERROR, message, **kwargs)


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured output."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime

        if isinstance(record.msg, dict):
            log_data = record.msg.copy()
        else:
            log_data = {"message": str(record.msg)}

        # Add timestamp
        log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Add standard fields
        log_data.setdefault("level", record.levelname.lower())
        log_data.setdefault("logger", record.name)

        # Handle exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False, default=str)


def configure_logging(
    level: str = "INFO",
    json_format: bool = True,
) -> None:
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON formatting (True for production)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        # Human-readable format for development
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root_logger.addHandler(handler)

    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured StructuredLogger instance
    """
    return StructuredLogger(name)


# Module exports
__all__ = [
    "StructuredLogger",
    "configure_logging",
    "get_logger",
    "get_request_id",
    "set_request_id",
    "request_id_var",
]
