"""
Gemini LLM Service Adapter

Production implementation using Google Gemini via LangChain.
Supports both single-shot generation and async streaming (SSE) with robust failover.
"""

import asyncio
import json
from typing import Any, AsyncIterator, Optional, Sequence
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ServiceUnavailable, ResourceExhausted

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

    Provides auto-failover tracking across multi-key architecture configurations.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model_name = settings.LLM_MODEL
        self._request_timeout_seconds = settings.LLM_REQUEST_TIMEOUT_SECONDS
        self._max_retries = settings.LLM_MAX_RETRIES
        self._retry_base_delay_seconds = settings.LLM_RETRY_BASE_DELAY_SECONDS
        self._temperature = settings.LLM_TEMPERATURE

        raw_keys = getattr(settings, "GEMINI_API_KEYS", "") or settings.GEMINI_API_KEY
        self._api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

        if not self._api_keys:
            raise ValueError("No Gemini API keys configured in settings.")

        # Initialize tracking sliding windows and structural blockades
        self._keys_tracker = {
            key: {
                "requests_count": 0,
                "window_start": time.monotonic(),
                "benched_until": 0.0
            }
            for key in self._api_keys
        }

        self._lock = asyncio.Lock()
        self._last_successful_generation_at: Optional[float] = None

    async def _get_best_llm_instance(self) -> tuple[ChatGoogleGenerativeAI, str]:
        """Finds the most available, non-throttled API key under the current runtime constraints."""
        async with self._lock:
            now = time.monotonic()

            for key in self._api_keys:
                tracker = self._keys_tracker[key]
                if now - tracker["window_start"] >= 60.0:
                    tracker["requests_count"] = 0
                    tracker["window_start"] = now

            SAFE_LIMIT_PER_MINUTE = 5

            # Filter keys that are neither rate-saturated nor explicitly benched via 503 circuit-breaking
            available_keys = [
                k for k in self._api_keys
                if self._keys_tracker[k]["requests_count"] < SAFE_LIMIT_PER_MINUTE
                and now >= self._keys_tracker[k]["benched_until"]
            ]

            if not available_keys:
                # If all keys are saturated or cooling down, grab the least utilized key as an absolute fallback
                selected_key = min(self._api_keys, key=lambda k: self._keys_tracker[k]["requests_count"])
                logger.warning("all_keys_saturated_forcing_lowest_load", key_index=self._api_keys.index(selected_key))
            else:
                selected_key = min(available_keys, key=lambda k: self._keys_tracker[k]["requests_count"])

            self._keys_tracker[selected_key]["requests_count"] += 1
            key_index = self._api_keys.index(selected_key)
            masked_key = f"{selected_key[:4]}...{selected_key[-4:]}" if len(selected_key) > 8 else "invalid_key"

            logger.info(
                "api_key_selected",
                key_index=key_index,
                masked_key=masked_key,
                current_minute_load=self._keys_tracker[selected_key]["requests_count"]
            )

            llm_instance = ChatGoogleGenerativeAI(
                model=self._model_name,
                temperature=self._temperature,
                google_api_key=selected_key,
            )
            return llm_instance, selected_key

    async def _handle_key_failure(self, key: str, is_server_error: bool = False) -> None:
        """Helper to clear loads and temporarily bench keys hitting 503/429 limits."""
        async with self._lock:
            if key in self._keys_tracker:
                if self._keys_tracker[key]["requests_count"] > 0:
                    self._keys_tracker[key]["requests_count"] -= 1
                if is_server_error:
                    # Implement Circuit Breaking: Bench this key for 45 seconds so other operations bypass it
                    self._keys_tracker[key]["benched_until"] = time.monotonic() + 45.0
                    logger.warning("key_benched_due_to_throttling", key_index=self._api_keys.index(key), duration=45)

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
            llm, active_key = await self._get_best_llm_instance()
            try:
                result = await asyncio.wait_for(
                    llm.ainvoke(messages),
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
            except (ServiceUnavailable, ResourceExhausted) as throttling_exc:
                await self._handle_key_failure(active_key, is_server_error=True)
                if attempt == attempts - 1:
                    raise throttling_exc
                await asyncio.sleep(self._retry_base_delay_seconds * (2**attempt))
            except Exception as exc:
                await self._handle_key_failure(active_key, is_server_error=False)
                if attempt == attempts - 1:
                    logger.error("llm_generate_failed", error=str(exc))
                    raise
                await asyncio.sleep(self._retry_base_delay_seconds * (2**attempt))

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
        """Yield tokens one-by-one from Gemini's async stream with auto-key failover capabilities."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        # We allow streaming to look across up to 3 keys if consecutive keys are hitting immediate 503 limits
        max_stream_failover_attempts = min(3, len(self._api_keys))

        for attempt in range(max_stream_failover_attempts):
            llm, active_key = await self._get_best_llm_instance()
            try:
                # We consume chunks inside a try block to catch immediate 503/429 handshakes
                async for chunk in llm.astream(messages):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    token = _coerce_to_text(token)
                    if token:
                        yield token

                self._last_successful_generation_at = time.monotonic()
                return  # Stream finished successfully, break out completely

            except (ServiceUnavailable, ResourceExhausted) as throttling_exc:
                await self._handle_key_failure(active_key, is_server_error=True)
                logger.warning("astream_key_throttled_switching_key", attempt=attempt+1, key_index=self._api_keys.index(active_key))
                if attempt == max_stream_failover_attempts - 1:
                    raise throttling_exc
                continue
            except Exception as exc:
                await self._handle_key_failure(active_key, is_server_error=False)
                if attempt == max_stream_failover_attempts - 1:
                    logger.error("astream_failed_permanently", error=str(exc))
                    raise
                continue

    async def generate_with_tools(
        self,
        request: LLMRequest,
        tools: Sequence[Any],
        max_iterations: int = 6,
    ) -> LLMResponse:
        if not tools:
            return await self.generate(request)

        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

        llm, _ = await self._get_best_llm_instance()
        llm_with_tools = llm.bind_tools(list(tools))
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
        """Run a tool-calling loop and stream assistant responses in real time with Load Balancing."""
        if not tools:
            async for chunk in self.astream(
                    request.prompt,
                    system_prompt=request.system_prompt,
            ):
                yield chunk, False, None

            self._last_successful_generation_at = time.monotonic()
            yield "", True, {
                "tool_calls_count": 0,
                "tool_invocations": [],
            }
            return

        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

        llm, _ = await self._get_best_llm_instance()
        llm_with_tools = llm.bind_tools(list(tools))
        tool_map = {tool.name: tool for tool in tools}

        messages = []
        if request.system_prompt:
            messages.append(SystemMessage(content=request.system_prompt))
        messages.append(HumanMessage(content=request.prompt))

        tool_calls_count = 0
        tool_invocations: list[dict[str, Any]] = []

        for _ in range(max_iterations):
            full_msg = None
            try:
                stream = llm_with_tools.astream(messages)
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            anext(stream),
                            timeout=self._request_timeout_seconds,
                        )
                    except StopAsyncIteration:
                        break

                    if full_msg is None:
                        full_msg = chunk
                    else:
                        full_msg += chunk

                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    content = _coerce_to_text(content)

                    if content:
                        yield content, False, None
                        await asyncio.sleep(0)

            except asyncio.TimeoutError:
                raise TimeoutError(f"LLM streaming exceeded timeout ({self._request_timeout_seconds}s)")

            ai_msg = full_msg
            if ai_msg is None:
                ai_msg = AIMessage(content="")

            messages.append(ai_msg)
            tool_calls = getattr(ai_msg, "tool_calls", None) or []

            if not tool_calls:
                self._last_successful_generation_at = time.monotonic()
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
                    {"name": tool_name, "args": tool_args, "result": tool_result}
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
                    ToolMessage(content=tool_result_text, tool_call_id=call_id, name=tool_name)
                )

        raise RuntimeError("Tool-calling loop exceeded max_iterations without final response")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            llm, _ = await self._get_best_llm_instance()
            _ = getattr(llm, "model", None) or getattr(llm, "model_name", None)

            if (
                self._last_successful_generation_at is not None
                and time.monotonic() - self._last_successful_generation_at <= 60
            ):
                return True

            result = await asyncio.wait_for(llm.ainvoke("ping"), timeout=3.0)
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