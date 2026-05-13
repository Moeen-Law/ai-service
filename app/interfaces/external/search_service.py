"""
Search Service Interface

Abstract interface for web search capabilities.
Enables swappable search providers (Tavily, Google, Bing, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class SearchResult:
    """Single search result from web search."""

    title: str
    url: str
    snippet: str
    source: Optional[str] = None


@dataclass(frozen=True)
class SearchResponse:
    """Response from web search."""

    results: List[SearchResult]
    query: str
    total_results: int


class SearchServiceInterface(ABC):
    """
    Abstract interface for web search service.

    Implementations may connect to different search providers
    (Tavily, Google Search, Bing, etc.)
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 10,
        timeout_seconds: int = 10,
    ) -> SearchResponse:
        """
        Perform a web search.

        Args:
            query: The search query string
            max_results: Maximum number of results to return
            timeout_seconds: Timeout for the search request

        Returns:
            SearchResponse with results

        Raises:
            SearchServiceError: If the search fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the search service is available.

        Returns:
            True if service is healthy, False otherwise
        """
        pass
