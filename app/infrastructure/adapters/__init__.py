"""
Concrete Adapter Implementations
"""

from app.infrastructure.adapters.llm_adapter import GeminiLLMService, llm_service
from app.infrastructure.adapters.prompt_adapter import (
    LegalPromptService,
    prompt_service,
)
from app.infrastructure.adapters.rag_adapter import QdrantRAGService, rag_service

__all__ = [
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
