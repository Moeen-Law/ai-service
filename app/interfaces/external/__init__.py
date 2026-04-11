"""
External Service Interfaces - Ports for other external systems.
"""

from app.interfaces.external.file_service import (
    ExternalFile,
    ExternalFileServiceInterface,
)

__all__ = [
    "ExternalFile",
    "ExternalFileServiceInterface",
]
