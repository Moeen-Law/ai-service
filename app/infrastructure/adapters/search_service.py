"""
Tavily Search Service Adapter

Implements web search using TavilySearchResults from LangChain.
"""

import asyncio
import os
from typing import Optional

from langchain_community.tools.tavily_search import TavilySearchResults

from app.infrastructure.logging.logger import get_logger
from app.interfaces.external.search_service import (
    SearchResponse,
    SearchResult,
    SearchServiceInterface,
)
from app.shared.errors.exceptions import AIServiceError
from app.infrastructure.config.settings import get_settings

logger = get_logger(__name__)


class SearchServiceError(AIServiceError):
    """Raised when search service operations fail."""

    pass


class TavilySearchAdapter(SearchServiceInterface):
    """
    Tavily search service adapter.

    Wraps TavilySearchResults tool with timeout handling and error recovery.
    """

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 2):
        """
        Initialize Tavily search adapter.

        Args:
            api_key: Tavily API key (uses TAVILY_API_KEY env var if not provided)
            max_retries: Number of retries on transient failures
        """
        self._max_retries = max_retries

        # prefer explicit api_key argument, otherwise read environment or settings
        settings = get_settings()
        key = api_key or os.getenv("TAVILY_API_KEY") or settings.TAVILY_API_KEY

        if not key:
            logger.warning(
                "tavily_api_key_missing",
                reason="TAVILY_API_KEY not set; Tavily search disabled",
            )
            self._tool = None
            return

        try:
            # Tavily wrapper expects parameter named `tavily_api_key`
            self._tool = TavilySearchResults(
                max_results=10,
                tavily_api_key=key,
            )
        except Exception as e:
            logger.error("tavily_initialization_failed", error=str(e))
            raise SearchServiceError(
                message="Failed to initialize Tavily search service",
                details={"error": str(e)},
            ) from e

    async def search(
        self,
        query: str,
        max_results: int = 10,
        timeout_seconds: int = 10,
    ) -> SearchResponse:
        """
        Perform a web search using Tavily.

        Args:
            query: The search query
            max_results: Maximum results to return
            timeout_seconds: Timeout for the search

        Returns:
            SearchResponse with results

        Raises:
            SearchServiceError: If search fails after retries
        """
        if not query or not query.strip():
            raise SearchServiceError(
                message="Search query cannot be empty",
                details={"query": query},
            )

        if self._tool is None:
            # Tavily not configured — return empty results so workflows can continue
            logger.info(
                "tavily_search_unconfigured",
                query=(query[:100] if query else ""),
            )
            return SearchResponse(results=[], query=query, total_results=0)

        logger.info(
            "tavily_search_started",
            query=query[:100],
            max_results=max_results,
        )

        for attempt in range(self._max_retries):
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(self._tool.invoke, {"query": query}),
                    timeout=timeout_seconds,
                )

                if not result:
                    logger.warning(
                        "tavily_search_empty_results",
                        query=query[:100],
                        attempt=attempt + 1,
                    )
                    return SearchResponse(
                        results=[],
                        query=query,
                        total_results=0,
                    )

                parsed_results = self._parse_results(result, max_results)

                logger.info(
                    "tavily_search_success",
                    query=query[:100],
                    results_count=len(parsed_results),
                )

                return SearchResponse(
                    results=parsed_results,
                    query=query,
                    total_results=len(parsed_results),
                )

            except asyncio.TimeoutError:
                logger.warning(
                    "tavily_search_timeout",
                    query=query[:100],
                    timeout_seconds=timeout_seconds,
                    attempt=attempt + 1,
                )
                if attempt == self._max_retries - 1:
                    raise SearchServiceError(
                        message=f"Search timed out after {timeout_seconds}s",
                        details={"query": query, "timeout_seconds": timeout_seconds},
                    ) from None

            except Exception as e:
                logger.warning(
                    "tavily_search_error",
                    query=query[:100],
                    error=str(e),
                    attempt=attempt + 1,
                )
                if attempt == self._max_retries - 1:
                    raise SearchServiceError(
                        message="Search service failed after retries",
                        details={"query": query, "error": str(e)},
                    ) from e

        raise SearchServiceError(
            message="Search service failed after all retries",
            details={"query": query},
        )

    @staticmethod
    def _parse_results(raw_results: list, max_results: int) -> list[SearchResult]:
        """
        Parse Tavily results into SearchResult objects.

        Args:
            raw_results: Raw results from Tavily
            max_results: Maximum results to return

        Returns:
            List of SearchResult objects
        """
        parsed = []

        for item in raw_results[:max_results]:
            if not isinstance(item, dict):
                continue

            title = item.get("title", "")
            url = item.get("url", "")
            snippet = item.get("content", "")

            if not title or not url:
                continue

            parsed.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet[:500],
                    source=item.get("source"),
                )
            )

        return parsed

    async def health_check(self) -> bool:
        """
        Check if Tavily search service is available.

        Returns:
            True if service is healthy
        """
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._tool.invoke, {"query": "test"}),
                timeout=5,
            )
            return bool(result)
        except Exception as e:
            logger.error("tavily_health_check_failed", error=str(e))
            return False


_search_service_instance: Optional[TavilySearchAdapter] = None


def get_search_service() -> TavilySearchAdapter:
    """
    Get or create the global search service singleton.

    Returns:
        TavilySearchAdapter instance
    """
    global _search_service_instance
    if _search_service_instance is None:
        _search_service_instance = TavilySearchAdapter()
    return _search_service_instance
