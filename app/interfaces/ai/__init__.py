"""
AI Service Interfaces - Abstract ports for RAG, LLM, and Prompt services.
"""

from app.interfaces.ai.llm_service import (
    LLMRequest,
    LLMResponse,
    LLMServiceInterface,
)
from app.interfaces.ai.prompt_service import (
    AssembledPrompt,
    PromptServiceInterface,
    PromptTemplate,
)
from app.interfaces.ai.rag_service import (
    Document,
    RAGQuery,
    RAGResponse,
    RAGServiceInterface,
)

__all__ = [
    # LLM Service
    "LLMServiceInterface",
    "LLMRequest",
    "LLMResponse",
    # RAG Service
    "RAGServiceInterface",
    "RAGQuery",
    "RAGResponse",
    "Document",
    # Prompt Service
    "PromptServiceInterface",
    "PromptTemplate",
    "AssembledPrompt",
]
