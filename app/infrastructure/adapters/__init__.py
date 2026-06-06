"""
Concrete Adapter Implementations
"""

from app.infrastructure.adapters.file_service_adapter import (
    HTTPFileService,
    file_service,
)
from app.infrastructure.adapters.file_generation_adapter import (
    HTTPFileGenerationService,
    MockFileGenerationService,
    file_generation_service,
)
from app.infrastructure.adapters.llm_adapter import GeminiLLMService, get_llm_service
from app.infrastructure.adapters.prompt_adapter import (
    LegalPromptService,
    prompt_service,
)
from app.infrastructure.adapters.rag_adapter import QdrantRAGService, rag_service
from app.infrastructure.adapters.search_service import TavilySearchAdapter, get_search_service
from .cache_adapter import get_semantic_cache_service
__all__ = [
    # Files Adapter
    "HTTPFileService",
    "file_service",
    # File Generation Adapter
    "HTTPFileGenerationService",
    "MockFileGenerationService",
    "file_generation_service",
    # LLM Adapter
    "GeminiLLMService",
    "get_llm_service",
    # RAG Adapter
    "QdrantRAGService",
    "rag_service",
    # Prompt Adapter
    "LegalPromptService",
    "prompt_service",
    # Search Adapter
    "TavilySearchAdapter",
    "get_search_service",
    # Cache Adapter
    "get_semantic_cache_service",
]
