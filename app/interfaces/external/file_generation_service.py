"""
External File Generation Service Interface

Abstract interface for generating file artifacts from text content
and returning stored file identifiers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class FileGenerationRequest:
    """Request payload for generating a document artifact."""

    source_prompt: str
    content: str
    filename: str
    content_type: str = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    metadata: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class GeneratedFile:
    """Result of document generation and mocked storage."""

    file_id: str
    filename: str
    content_type: str
    size_bytes: int


class FileGenerationServiceInterface(ABC):
    """Contract for generating and storing files from model output."""

    @abstractmethod
    async def generate_docx(self, request: FileGenerationRequest) -> GeneratedFile:
        """Generate a DOCX artifact and return storage metadata."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if generation tool is available."""
        pass
