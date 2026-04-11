"""
Structured Logging Configuration

Provides structured, JSON-formatted logging for observability.
Uses structlog for structured logging with request ID propagation.
"""

import logging
import sys
from contextvars import ContextVar
from pathlib import Path
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
    log_file_path: str = "logs/ai-service.log",
    overwrite_log_file: bool = True,
) -> None:
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON formatting (True for production)
        log_file_path: File path for persisted logs
        overwrite_log_file: Truncate log file on startup when True
    """
    root_logger = logging.getLogger()
    resolved_level = getattr(logging, level.upper())
    root_logger.setLevel(resolved_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    if json_format:
        formatter: logging.Formatter = JSONFormatter()
    else:
        # Human-readable format for development
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Keep console logs for local visibility
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(resolved_level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Persist logs to a single file; mode='w' overwrites each run
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "w" if overwrite_log_file else "a"
    file_handler = logging.FileHandler(log_path, mode=file_mode, encoding="utf-8")
    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

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
