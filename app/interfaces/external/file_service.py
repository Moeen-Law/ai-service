"""
External File Service Interface

Abstract interface defining the contract for retrieving uploaded files
from the Files service by file ID.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ExternalFile:
    """Represents a binary file fetched from an external files service."""

    file_id: str
    content: bytes
    filename: Optional[str] = None
    content_type: Optional[str] = None

    @property
    def size_bytes(self) -> int:
        """Return the raw binary size of the file."""
        return len(self.content)


class ExternalFileServiceInterface(ABC):
    """Contract for retrieving uploaded files from an external service."""

    @abstractmethod
    async def fetch_file(self, file_id: str) -> ExternalFile:
        """Fetch one file by ID from the external files service."""
        pass

    @abstractmethod
    async def fetch_files(self, file_ids: List[str]) -> List[ExternalFile]:
        """Fetch multiple files by IDs while preserving input order."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the external files service is reachable."""
        pass
