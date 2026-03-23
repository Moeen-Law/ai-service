"""
Gemini LLM Service Adapter

Production implementation using Google Gemini via LangChain.
Supports both single-shot generation and async streaming (SSE).
"""

from typing import AsyncIterator, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from app.infrastructure.config.settings import get_settings
from app.infrastructure.logging.logger import get_logger
from app.interfaces.ai.llm_service import (
    LLMRequest,
    LLMResponse,
    LLMServiceInterface,
)

logger = get_logger(__name__)


class GeminiLLMService(LLMServiceInterface):
    """
    LLM service backed by Google Gemini (via ``langchain-google-genai``).

    Provides:
    - ``generate`` — single-shot text generation
    - ``generate_with_context`` — generation with additional context
    - ``astream`` — async token-level streaming for SSE
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model_name = settings.LLM_MODEL
        self._llm = ChatGoogleGenerativeAI(
            model=self._model_name,
            temperature=settings.LLM_TEMPERATURE,
            google_api_key=settings.GEMINI_API_KEY,
        )

    # ------------------------------------------------------------------
    # Single-shot generation
    # ------------------------------------------------------------------

    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            result = await self._llm.ainvoke(request.prompt)

            content = result.content if hasattr(result, "content") else str(result)
            
            # Ensure content is a string (handle list from Gemini)
            if isinstance(content, list):
                content = " ".join(str(item) for item in content)
            elif not isinstance(content, str):
                content = str(content)
            
            return LLMResponse(
                content=content,
                model=self._model_name,
                tokens_used=len(content.split()),
                finish_reason="stop",
            )
        except Exception as exc:
            logger.error("llm_generate_failed", error=str(exc))
            raise

    async def generate_with_context(
        self,
        prompt: str,
        context: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        combined = ""
        if system_prompt:
            combined += f"{system_prompt}\n\n"
        combined += f"السياق:\n{context}\n\nالسؤال:\n{prompt}"

        result = await self._llm.ainvoke(combined)
        content = result.content if hasattr(result, "content") else str(result)
        
        # Ensure content is a string (handle list from Gemini)
        if isinstance(content, list):
            content = " ".join(str(item) for item in content)
        elif not isinstance(content, str):
            content = str(content)
        
        return LLMResponse(
            content=content,
            model=self._model_name,
            tokens_used=len(content.split()),
            finish_reason="stop",
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        """Yield tokens one-by-one from Gemini's async stream."""
        async for chunk in self._llm.astream(prompt):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                yield token

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            result = await self._llm.ainvoke("ping")
            return bool(result)
        except Exception:
            return False


# Default instance for dependency injection
llm_service = GeminiLLMService()
