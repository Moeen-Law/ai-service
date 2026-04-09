"""
Gemini LLM Service Adapter

Production implementation using Google Gemini via LangChain.
Supports both single-shot generation and async streaming (SSE).
"""

import asyncio

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
        self._request_timeout_seconds = settings.LLM_REQUEST_TIMEOUT_SECONDS
        self._max_retries = settings.LLM_MAX_RETRIES
        self._retry_base_delay_seconds = settings.LLM_RETRY_BASE_DELAY_SECONDS
        self._llm = ChatGoogleGenerativeAI(
            model=self._model_name,
            temperature=settings.LLM_TEMPERATURE,
            google_api_key=settings.GEMINI_API_KEY,
        )

    # ------------------------------------------------------------------
    # Single-shot generation
    # ------------------------------------------------------------------

    async def generate(self, request: LLMRequest) -> LLMResponse:
        attempts = self._max_retries + 1
        last_error: Optional[Exception] = None

        for attempt in range(attempts):
            try:
                result = await asyncio.wait_for(
                    self._llm.ainvoke(request.prompt),
                    timeout=self._request_timeout_seconds,
                )

                content = result.content if hasattr(result, "content") else str(result)

                # Ensure content is a string (Gemini may return list chunks)
                if isinstance(content, list):
                    content = "".join(str(item) for item in content)
                elif not isinstance(content, str):
                    content = str(content)

                return LLMResponse(
                    content=content,
                    model=self._model_name,
                    tokens_used=len(content.split()),
                    finish_reason="stop",
                )
            except Exception as exc:
                last_error = exc
                is_last_attempt = attempt == attempts - 1
                logger.warning(
                    "llm_generate_attempt_failed",
                    attempt=attempt + 1,
                    max_attempts=attempts,
                    error=str(exc),
                )

                if is_last_attempt:
                    logger.error("llm_generate_failed", error=str(exc))
                    raise

                backoff_seconds = self._retry_base_delay_seconds * (2**attempt)
                await asyncio.sleep(backoff_seconds)

        if last_error is not None:
            raise last_error

        raise RuntimeError("LLM generation failed without an explicit error")

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

        return await self.generate(
            LLMRequest(
                prompt=combined,
                max_tokens=max_tokens,
            )
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
