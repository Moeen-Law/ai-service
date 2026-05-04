"""
Gemini LLM Service Adapter

Production implementation using Google Gemini via LangChain.
Supports both single-shot generation and async streaming (SSE).
"""

import asyncio
import json

from typing import Any, AsyncIterator, Optional, Sequence
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.infrastructure.config.settings import get_settings
from app.infrastructure.logging.logger import get_logger
from app.interfaces.ai.llm_service import (
    LLMRequest,
    LLMResponse,
    LLMServiceInterface,
)

logger = get_logger(__name__)


def _coerce_to_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(_coerce_to_text(item) for item in content)
    if isinstance(content, dict):
        if content.get("type") in {"tool_use", "tool_result", "image", "document"}:
            return ""
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
        return json.dumps(content, ensure_ascii=False)
    if hasattr(content, "text") and isinstance(getattr(content, "text"), str):
        return getattr(content, "text")
    if isinstance(content, str):
        return content
    return str(content)


def _extract_token_count(message: Any, content: str) -> int:
    usage_metadata = getattr(message, "usage_metadata", None)
    token_count = 0

    if usage_metadata is not None:
        if hasattr(usage_metadata, "total_token_count"):
            token_count = getattr(usage_metadata, "total_token_count") or 0
        elif isinstance(usage_metadata, dict):
            token_count = (
                usage_metadata.get("total_token_count")
                or usage_metadata.get("total_tokens")
                or 0
            )

    if token_count:
        return int(token_count)

    logger.warning("token_count_unavailable")
    return len(content.split())


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
        self._last_successful_generation_at: Optional[float] = None

    # ------------------------------------------------------------------
    # Single-shot generation
    # ------------------------------------------------------------------

    async def generate(self, request: LLMRequest) -> LLMResponse:
        attempts = self._max_retries + 1
        messages = []
        if request.system_prompt:
            messages.append(SystemMessage(content=request.system_prompt))
        messages.append(HumanMessage(content=request.prompt))

        for attempt in range(attempts):
            try:
                result = await asyncio.wait_for(
                    self._llm.ainvoke(messages),
                    timeout=self._request_timeout_seconds,
                )

                content = result.content if hasattr(result, "content") else str(result)
                content = _coerce_to_text(content)
                self._last_successful_generation_at = time.monotonic()

                return LLMResponse(
                    content=content,
                    model=self._model_name,
                    tokens_used=_extract_token_count(result, content),
                    finish_reason="stop",
                )
            except Exception as exc:
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

    async def generate_with_context(
        self,
        prompt: str,
        context: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        return await self.generate(
            LLMRequest(
                prompt=f"السياق:\n{context}\n\nالسؤال:\n{prompt}",
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def astream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Yield tokens one-by-one from Gemini's async stream."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        async for chunk in self._llm.astream(messages):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            token = _coerce_to_text(token)
            if token:
                yield token

    async def generate_with_tools(
        self,
        request: LLMRequest,
        tools: Sequence[Any],
        max_iterations: int = 6,
    ) -> LLMResponse:
        """Run a tool-calling loop and return the final assistant response."""
        if not tools:
            return await self.generate(request)

        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

        llm_with_tools = self._llm.bind_tools(list(tools))
        tool_map = {tool.name: tool for tool in tools}

        messages = []
        if request.system_prompt:
            messages.append(SystemMessage(content=request.system_prompt))
        messages.append(HumanMessage(content=request.prompt))

        tool_calls_count = 0
        tool_invocations: list[dict[str, Any]] = []

        for _ in range(max_iterations):
            ai_msg = await asyncio.wait_for(
                llm_with_tools.ainvoke(messages),
                timeout=self._request_timeout_seconds,
            )
            messages.append(ai_msg)

            tool_calls = getattr(ai_msg, "tool_calls", None) or []
            if not tool_calls:
                content = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)
                content = _coerce_to_text(content)
                self._last_successful_generation_at = time.monotonic()

                return LLMResponse(
                    content=content,
                    model=self._model_name,
                    tokens_used=_extract_token_count(ai_msg, content),
                    finish_reason="stop",
                    metadata={
                        "tool_calls_count": tool_calls_count,
                        "tool_invocations": tool_invocations,
                    },
                )

            for idx, call in enumerate(tool_calls):
                tool_name = call.get("name")
                tool_obj = tool_map.get(tool_name)
                if tool_obj is None:
                    raise ValueError(f"Model requested unknown tool: {tool_name}")

                tool_args = call.get("args", {})
                if not isinstance(tool_args, dict):
                    tool_args = {}

                tool_calls_count += 1
                tool_result = await tool_obj.ainvoke(tool_args)
                tool_invocations.append(
                    {
                        "name": tool_name,
                        "args": tool_args,
                        "result": tool_result,
                    }
                )
                if isinstance(tool_result, str):
                    tool_result_text = tool_result
                else:
                    try:
                        tool_result_text = json.dumps(tool_result, ensure_ascii=False)
                    except TypeError:
                        tool_result_text = str(tool_result)

                call_id = call.get("id") or f"tool_call_{tool_calls_count}_{idx}"
                messages.append(
                    ToolMessage(
                        content=tool_result_text,
                        tool_call_id=call_id,
                        name=tool_name,
                    )
                )

        raise RuntimeError(
            "Tool-calling loop exceeded max_iterations without final response"
        )

    async def stream_with_tools(
        self,
        request: LLMRequest,
        tools: Sequence[Any],
        max_iterations: int = 6,
    ) -> AsyncIterator[tuple[str, bool, Optional[dict[str, Any]]]]:
        """Run a tool-calling loop and return the final assistant response as a stream."""
        if not tools:
            async for chunk in self.astream(
                request.prompt,
                system_prompt=request.system_prompt,
            ):
                yield chunk, False, None
            self._last_successful_generation_at = time.monotonic()
            yield "", True, {"tool_calls_count": 0, "tool_invocations": []}
            return

        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

        llm_with_tools = self._llm.bind_tools(list(tools))
        tool_map = {tool.name: tool for tool in tools}

        messages = []
        if request.system_prompt:
            messages.append(SystemMessage(content=request.system_prompt))
        messages.append(HumanMessage(content=request.prompt))

        tool_calls_count = 0
        tool_invocations: list[dict[str, Any]] = []

        for _ in range(max_iterations):
            async def accumulate_turn() -> tuple[Any, list[str]]:
                full_msg = None
                buffer: list[str] = []

                async for chunk in llm_with_tools.astream(messages):
                    if full_msg is None:
                        full_msg = chunk
                    else:
                        full_msg += chunk

                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    content = _coerce_to_text(content)
                    if content:
                        buffer.append(content)

                return full_msg, buffer

            full_msg, buffer = await asyncio.wait_for(
                accumulate_turn(),
                timeout=self._request_timeout_seconds,
            )

            ai_msg = full_msg
            if ai_msg is None:
                ai_msg = AIMessage(content="")

            messages.append(ai_msg)

            tool_calls = getattr(ai_msg, "tool_calls", None) or []
            if not tool_calls:
                self._last_successful_generation_at = time.monotonic()
                for content in buffer:
                    yield content, False, None
                yield "", True, {
                    "tool_calls_count": tool_calls_count,
                    "tool_invocations": tool_invocations,
                }
                return

            for idx, call in enumerate(tool_calls):
                tool_name = call.get("name")
                tool_obj = tool_map.get(tool_name)
                if tool_obj is None:
                    raise ValueError(f"Model requested unknown tool: {tool_name}")

                tool_args = call.get("args", {})
                if not isinstance(tool_args, dict):
                    tool_args = {}

                tool_calls_count += 1
                tool_result = await tool_obj.ainvoke(tool_args)
                tool_invocations.append(
                    {
                        "name": tool_name,
                        "args": tool_args,
                        "result": tool_result,
                    }
                )
                if isinstance(tool_result, str):
                    tool_result_text = tool_result
                else:
                    try:
                        tool_result_text = json.dumps(tool_result, ensure_ascii=False)
                    except TypeError:
                        tool_result_text = str(tool_result)

                call_id = call.get("id") or f"tool_call_{tool_calls_count}_{idx}"
                messages.append(
                    ToolMessage(
                        content=tool_result_text,
                        tool_call_id=call_id,
                        name=tool_name,
                    )
                )

        raise RuntimeError(
            "Tool-calling loop exceeded max_iterations without final response"
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            _ = getattr(self._llm, "model", None) or getattr(
                self._llm, "model_name", None
            )

            if (
                self._last_successful_generation_at is not None
                and time.monotonic() - self._last_successful_generation_at <= 60
            ):
                return True

            # LangChain's Gemini chat wrapper does not expose a zero-cost ping;
            # use a short, cached fallback generation only when no recent call succeeded.
            result = await asyncio.wait_for(self._llm.ainvoke("ping"), timeout=3.0)
            if result:
                self._last_successful_generation_at = time.monotonic()
                return True
            return False
        except Exception:
            return False


_llm_service: Optional[GeminiLLMService] = None


def get_llm_service() -> GeminiLLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = GeminiLLMService()
    return _llm_service
