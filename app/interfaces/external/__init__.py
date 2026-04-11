"""
External Service Interfaces - Ports for other external systems.
"""

from app.interfaces.external.file_service import (
    ExternalFile,
    ExternalFileServiceInterface,
)
from app.interfaces.external.file_generation_service import (
    FileGenerationRequest,
    FileGenerationServiceInterface,
    GeneratedFile,
)

__all__ = [
    "ExternalFile",
    "ExternalFileServiceInterface",
    "FileGenerationRequest",
    "FileGenerationServiceInterface",
    "GeneratedFile",
]
