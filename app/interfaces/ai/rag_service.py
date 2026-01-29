"""
RAG Service Interface

Abstract interface defining the contract for RAG service interactions.
This is a port in the hexagonal architecture pattern.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Document:
    """
    Represents a retrieved document from the RAG service.
    """

    id: str
    content: str
    source: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class RAGQuery:
    """
    Query object for RAG service calls.
    """

    query: str
    jurisdiction: str
    language: str
    domain: Optional[str] = None
    top_k: int = 5
    threshold: float = 0.7
    filters: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class RAGResponse:
    """
    Response object from RAG service calls.
    """

    documents: List[Document]
    query: str
    total_found: int
    metadata: Optional[Dict[str, Any]] = None


class RAGServiceInterface(ABC):
    """
    Abstract interface for RAG (Retrieval Augmented Generation) service.

    Defines the contract for retrieving relevant documents
    from a knowledge base based on queries and context.
    """

    @abstractmethod
    async def retrieve(self, query: RAGQuery) -> RAGResponse:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The RAG query with search parameters

        Returns:
            RAGResponse with matching documents

        Raises:
            RAGServiceError: If the RAG service call fails
        """
        pass

    @abstractmethod
    async def retrieve_by_ids(
        self,
        document_ids: List[str],
    ) -> List[Document]:
        """
        Retrieve specific documents by their IDs.

        Args:
            document_ids: List of document IDs to retrieve

        Returns:
            List of Document objects

        Raises:
            RAGServiceError: If the retrieval fails
        """
        pass

    @abstractmethod
    async def search_similar(
        self,
        text: str,
        jurisdiction: str,
        top_k: int = 5,
    ) -> List[Document]:
        """
        Search for documents similar to the given text.

        Args:
            text: The text to find similar documents for
            jurisdiction: Legal jurisdiction to search within
            top_k: Number of documents to return

        Returns:
            List of similar Document objects
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the RAG service is available.

        Returns:
            True if service is healthy, False otherwise
        """
        pass
