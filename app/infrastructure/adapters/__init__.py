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
from app.infrastructure.adapters.llm_adapter import GeminiLLMService, llm_service
from app.infrastructure.adapters.prompt_adapter import (
    LegalPromptService,
    prompt_service,
)
from app.infrastructure.adapters.rag_adapter import QdrantRAGService, rag_service

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
    "llm_service",
    # RAG Adapter
    "QdrantRAGService",
    "rag_service",
    # Prompt Adapter
    "LegalPromptService",
    "prompt_service",
]
