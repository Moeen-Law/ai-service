"""
Legal Chat Workflow

Full-pipeline workflow for conversational legal queries.
Uses intent classification, hybrid RAG retrieval, LLM query-rewriting, and SSE streaming.

Graph runs for BOTH execute() and stream():
  - execute()  → graph.ainvoke()   → returns full Result
  - stream()   → graph.astream_events() → yields SSE tokens in real-time
"""
import asyncio
import json
import time
from dataclasses import replace
from typing import Any, AsyncIterator, Dict, List, Optional, TypedDict
from uuid import UUID

from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool

from app.core.services.file_text_extractor import FileTextExtractor
from app.core.domain.entities import Context, ExecutionOptions, Result, ResultMetadata
from app.core.domain.enums import Intent, TaskType
from app.core.validators.intent_classifier import (
    IntentClassificationResult,
    IntentClassifier,
)
from app.core.services.query_pipeline import QueryPipeline, PipelineResult
from app.core.workflows.base import BaseWorkflow
from app.infrastructure.logging.logger import get_logger
from app.interfaces.ai.llm_service import LLMRequest, LLMServiceInterface
from app.interfaces.ai.prompt_service import PromptServiceInterface
from app.interfaces.ai.rag_service import RAGServiceInterface
from app.interfaces.external.file_service import ExternalFileServiceInterface
from app.interfaces.external.cache_service import CacheServiceInterface
from app.shared.errors.exceptions import (
    FileExtractionError,
    FilesServiceError,
    PayloadValidationError,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

class LegalChatState(TypedDict, total=False):
    # ── inputs ──
    message: str
    files_ids: List[str]
    jurisdiction: str
    language: str
    retrieval_k: int
    conversation_history: Any
    include_sources: bool
    # ── intermediate ──
    intent_result: Optional[IntentClassificationResult]
    pipeline_result: Optional[PipelineResult]
    uploaded_files_context: Optional[str]
    assembled_prompt: Optional[str]
    system_prompt: Optional[str]
    tools: Optional[List[Any]]
    # ── outputs ──
    llm_response: Optional[Any]
    final_output: Optional[Dict[str, Any]]


# ─────────────────────────────────────────────
# Workflow
# ─────────────────────────────────────────────

class LegalChatWorkflow(BaseWorkflow):
    """
    Workflow for LEGAL_CHAT task type.

    Single LangGraph graph used by BOTH paths:
    - execute()       → ainvoke()        (non-streaming, returns Result)
    - stream()        → astream_events() (SSE token-by-token)

    Pipeline:
    1. Semantic cache check (stream path only, before graph)
    2. classify_intent  → route
       ├─ CHITCHAT / VAGUE / OUT_OF_SCOPE → personality_response → END
       └─ LEGAL_QUERY →
          fetch_files → run_rag_pipeline → merge_context →
          assemble_prompt → build_tools → llm_generate →
          filter_sources → END
    """

    def __init__(
        self,
        rag_service: RAGServiceInterface,
        llm_service: LLMServiceInterface,
        prompt_service: PromptServiceInterface,
        file_service: Optional[ExternalFileServiceInterface] = None,
        file_text_extractor: Optional[FileTextExtractor] = None,
        max_frontend_sources: int = 7,
        semantic_cache: Optional[CacheServiceInterface] = None,
    ) -> None:
        self._rag = rag_service
        self._llm = llm_service
        self._prompt = prompt_service
        self._intent_classifier = IntentClassifier(llm_service=llm_service)
        self._pipeline = QueryPipeline(
            rag_service=rag_service,
            llm_service=llm_service,
            max_frontend_sources=max_frontend_sources,
        )
        self._file_service = file_service
        self._file_text_extractor = file_text_extractor or FileTextExtractor()
        self._max_sources = max_frontend_sources
        self._semantic_cache = semantic_cache

        # ── build graph ──
        wf = StateGraph(LegalChatState)
        wf.add_node("classify_intent",    self._node_classify_intent)
        wf.add_node("fetch_files",        self._node_fetch_files)
        wf.add_node("run_rag_pipeline",   self._node_run_rag_pipeline)
        wf.add_node("merge_context",      self._node_merge_context)
        wf.add_node("assemble_prompt",    self._node_assemble_prompt)
        wf.add_node("build_tools",        self._node_build_tools)
        wf.add_node("llm_generate",       self._node_llm_generate)
        wf.add_node("filter_sources",     self._node_filter_output)
        wf.add_node("personality_response", self._node_personality_response)

        wf.add_edge(START, "classify_intent")
        wf.add_conditional_edges("classify_intent", self._route_by_intent)
        wf.add_edge("fetch_files",      "run_rag_pipeline")
        wf.add_edge("run_rag_pipeline", "merge_context")
        wf.add_edge("merge_context",    "assemble_prompt")
        wf.add_edge("assemble_prompt",  "build_tools")
        wf.add_edge("build_tools",      "llm_generate")
        wf.add_edge("llm_generate",     "filter_sources")
        wf.add_edge("filter_sources",     END)
        wf.add_edge("personality_response", END)

        self.graph = wf.compile()

    # ─────────────────────────────────────────
    # BaseWorkflow contract
    # ─────────────────────────────────────────

    @property
    def name(self) -> str:
        return "LegalChatWorkflow"

    async def execute(
        self,
        task_id: str,
        context: Context,
        payload: Dict[str, Any],
        options: Optional[ExecutionOptions] = None,
    ) -> Result:
        """Non-streaming path — runs the full graph and returns a Result."""
        start_time = time.time()

        files_ids = self._normalize_file_ids(payload.get("files_ids"))
        if files_ids and self._file_service is None:
            raise FilesServiceError(
                message="Files service integration is not configured",
                details={"field": "file_service"},
            )

        state = LegalChatState(
            message=payload.get("message", ""),
            files_ids=files_ids,
            jurisdiction=context.jurisdiction.value,
            language=context.language.value,
            retrieval_k=payload.get("retrieval_k", 4),
            conversation_history=payload.get("conversation_history", []),
            include_sources=options.include_sources if options is not None else True,
        )

        result_state = await self.graph.ainvoke(state)
        final_output = result_state.get("final_output", {})
        resp = result_state.get("llm_response")

        return Result.success(
            task_id=UUID(task_id),
            task_type=TaskType.LEGAL_CHAT,
            data=final_output,
            metadata=ResultMetadata(
                execution_time_ms=int((time.time() - start_time) * 1000),
                model_used=getattr(resp, "model", None),
                tokens_used=getattr(resp, "tokens_used", None),
            ),
        )

    # ─────────────────────────────────────────
    # SSE streaming path
    # ─────────────────────────────────────────

    async def stream(
        self,
        question: str,
        retrieval_k: int = 4,
        conversation_history: Optional[Any] = None,
        files_ids: Optional[List[str]] = None,
        jurisdiction: str = "egypt",
        language: str = "ar",
        include_sources: bool = True,
    ) -> AsyncIterator[str]:
        """Stream legal chat response via LangGraph astream_events."""
        t_start = time.perf_counter()
        normalized_files_ids = self._normalize_file_ids(files_ids)
        token_queue: asyncio.Queue[str] = asyncio.Queue()
        self._current_token_queue = token_queue

        logger.info(
            "legal_chat_stream_started",
            question=question[:100],
            files_count=len(normalized_files_ids),
        )

        try:
            if normalized_files_ids and self._file_service is None:
                raise FilesServiceError(
                    message="Files service integration is not configured",
                    details={"field": "file_service"},
                )

            # ── 1. Semantic cache (file-free queries only) ──
            if not normalized_files_ids and self._semantic_cache is not None:
                cached = await self._semantic_cache.get(question)
                if cached:
                    logger.info("legal_chat_semantic_cache_hit", question=question[:80])
                    async for event in self._stream_cached_response(cached, t_start):
                        yield event
                    return

            # ── 2. Run graph via astream_events ──
            state = LegalChatState(
                message=question,
                files_ids=normalized_files_ids,
                jurisdiction=jurisdiction,
                language=language,
                retrieval_k=retrieval_k,
                conversation_history=conversation_history or [],
                include_sources=include_sources,
            )

            complete_answer = ""
            final_output: Dict[str, Any] = {}
            first_token_logged = False
            llm_stream_start: Optional[float] = None
            t_retrieval_start: Optional[float] = None
            t_retrieval_end: Optional[float] = None

            graph_task = asyncio.create_task(
            self.graph.ainvoke(state)
        )

            while True:

                if graph_task.done() and token_queue.empty():
                    break

                try:
                    token = await asyncio.wait_for(
                        token_queue.get(),
                        timeout=0.1
                    )

                    if token == "__STREAM_DONE__":
                        break

                    complete_answer += token

                    yield self._sse(
                        "token",
                        {
                            "content": token
                        }
                    )

                except asyncio.TimeoutError:
                    continue

            result_state = await graph_task

            final_output = result_state.get(
                "final_output",
                {}
            )

            # ── post-graph: stream personality tokens word-by-word if chitchat ──
            is_personality = final_output.get("intent") not in (
                Intent.LEGAL_QUERY.value, None
            ) and not complete_answer
            if is_personality:
                reply_text = final_output.get("message", "")
                words = reply_text.split(" ")
                for i, word in enumerate(words):
                    chunk = word + (" " if i < len(words) - 1 else "")
                    complete_answer += chunk
                    yield self._sse("token", {"content": chunk})
                    await asyncio.sleep(0)

            # ── post-graph: emit sources + save cache ──
            t_done = time.perf_counter()
            timing = {
                "retrieval_ms": round(
                    ((t_retrieval_end or t_done) - (t_retrieval_start or t_done)) * 1000, 1
                ),
                "total_ms": round((t_done - t_start) * 1000, 1),
            }

            if llm_stream_start:
                logger.info(f"[LLM TIME] Total generation: {time.perf_counter() - llm_stream_start:.4f}s")

            intent_val = final_output.get("intent", Intent.LEGAL_QUERY.value)
            frontend_sources = final_output.get("sources", [])

            # save to semantic cache
            if not normalized_files_ids and self._semantic_cache and complete_answer:
                await self._semantic_cache.set(question, {
                    "answer":  complete_answer,
                    "sources": frontend_sources,
                })
            fallback = ""

            if not complete_answer:
                fallback = final_output.get(
                    "message",
                    ""
                )

            if fallback:
                yield self._sse(
                    "token",
                    {
                        "content": fallback
                    }
                )

            yield self._sse("sources", {
                "sources": frontend_sources,
                "timing":  timing,
                "intent":  intent_val,
            })

            logger.info("stream_ended_successfully", final_intent=intent_val)
            logger.info(
                "STREAM_RESULT",
                complete_answer_len=len(complete_answer),
                final_output=final_output
            )
            yield self._format_sse_data("[DONE]")

        except Exception as exc:
            error_message = self._stringify_exception(exc)
            logger.error("stream_error", error=error_message, error_type=type(exc).__name__)
            yield self._sse("error", {"content": error_message})
            yield self._format_sse_data("[DONE]")

    # ─────────────────────────────────────────
    # Graph nodes
    # ─────────────────────────────────────────

    async def _node_classify_intent(self, state: LegalChatState) -> Dict[str, Any]:
        files_ids = state.get("files_ids", [])
        question  = state.get("message", "")
        history   = self._format_conversation_history(state.get("conversation_history", []))

        intent_result = await self._intent_classifier.classify(question, history=history)

        if files_ids and intent_result.intent != Intent.LEGAL_QUERY:
            logger.info(
                "intent_overridden_due_to_files",
                original_intent=intent_result.intent.value,
            )
            intent_result = IntentClassificationResult(
                intent=Intent.LEGAL_QUERY,
                history=history,
                confidence=1.0,
                domain=intent_result.domain,
                keywords=intent_result.keywords,
                likely_articles=intent_result.likely_articles,
                reasoning="Overridden: uploaded files present",
            )

        logger.info(
            "intent_classification_result",
            intent=intent_result.intent.value,
            confidence=intent_result.confidence,
        )
        return {"intent_result": intent_result}

    def _route_by_intent(self, state: LegalChatState) -> str:
        intent_result = state.get("intent_result")
        if intent_result and intent_result.intent == Intent.LEGAL_QUERY:
            return "fetch_files"
        return "personality_response"

    async def _node_fetch_files(self, state: LegalChatState) -> Dict[str, Any]:
        files_ids = state.get("files_ids")
        uploaded_context = None
        if files_ids and self._file_service is not None:
            uploaded_context = await self._build_uploaded_files_context(files_ids)
        return {"uploaded_files_context": uploaded_context}

    async def _node_run_rag_pipeline(self, state: LegalChatState) -> Dict[str, Any]:
        intent_result = state.get("intent_result")
        pipeline_result = await self._pipeline.run(
            question=state.get("message", ""),
            retrieval_k=state.get("retrieval_k", 4),
            precomputed_domain=intent_result.domain if intent_result else None,
            precomputed_keywords=intent_result.keywords if intent_result else [],
            precomputed_articles=intent_result.likely_articles if intent_result else [],
            skip_rewrite=True,
        )
        return {"pipeline_result": pipeline_result}

    async def _node_merge_context(self, state: LegalChatState) -> Dict[str, Any]:
        pipeline_result   = state.get("pipeline_result")
        uploaded_context  = state.get("uploaded_files_context")
        if uploaded_context and pipeline_result:
            pipeline_result = replace(
                pipeline_result,
                context=self._merge_contexts(pipeline_result.context, uploaded_context),
            )
        return {"pipeline_result": pipeline_result}

    async def _node_assemble_prompt(self, state: LegalChatState) -> Dict[str, Any]:
        pipeline_result = state.get("pipeline_result")
        template = await self._prompt.get_template(
            task_type="LEGAL_CHAT",
            jurisdiction=state.get("jurisdiction", "egypt"),
            language=state.get("language", "ar"),
        )
        assembled = await self._prompt.assemble_prompt(
            template=template,
            variables={
                "question": state.get("message", ""),
                "context":  pipeline_result.context if pipeline_result else "",
                "conversation_history": self._format_conversation_history(
                    state.get("conversation_history", [])
                ),
            },
        )
        return {
            "assembled_prompt": assembled.prompt,
            "system_prompt":    assembled.system_prompt,
        }

    async def _node_build_tools(self, state: LegalChatState) -> Dict[str, Any]:
        files_ids         = state.get("files_ids", [])
        base_system       = state.get("system_prompt", "")
        tools             = self._build_agent_tools(files_ids=files_ids)
        agent_instructions = self._build_agent_system_prompt(files_ids=files_ids)

        # Only append tool instructions when tools are actually present
        if tools:
            combined = f"{base_system}\n\n=== تعليمات استخدام الأدوات ===\n{agent_instructions}"
        else:
            combined = base_system

        return {"tools": tools, "system_prompt": combined}

    async def _node_llm_generate(
    self,
    state: LegalChatState,
    token_queue: asyncio.Queue | None = None,
    ) -> Dict[str, Any]:
        """
        Run stream_with_tools and collect the full response.
        astream_events() intercepts the underlying on_chat_model_stream events
        automatically, so tokens flow to the SSE layer without extra work here.
        """
        complete_content = ""
        final_meta: Dict[str, Any] = {}

        async for chunk, is_final, metadata in self._llm.stream_with_tools(
            LLMRequest(
                prompt=state.get("assembled_prompt", ""),
                system_prompt=state.get("system_prompt", ""),
            ),
            tools=state.get("tools") or [],
        ):
            if chunk:
                logger.info(
                    "NODE_RECEIVED_TOKEN",
                    preview=chunk[:30]
                )

                complete_content += chunk

            queue = getattr(self, "_current_token_queue", None)

            if queue and chunk:
                await queue.put(chunk)

            if is_final:
                final_meta = metadata or {}

        queue = getattr(self, "_current_token_queue", None)

        if queue:
            await queue.put("__STREAM_DONE__")

        from app.shared.utils.token_calculator import TokenCostCalculator
        input_tokens  = TokenCostCalculator.estimate_tokens(
            (state.get("assembled_prompt") or "") + (state.get("system_prompt") or "")
        )
        output_tokens = TokenCostCalculator.estimate_tokens(complete_content)
        cost_per_1k   = TokenCostCalculator.calculate_cost_per_1k(input_tokens, output_tokens)
        logger.info(f"[TOKEN USAGE] Input: {input_tokens} | Output: {output_tokens} | Total: {input_tokens + output_tokens}")
        logger.info(f"[COST ESTIMATE] Cost/1k queries: ${cost_per_1k:.4f}")

        class _Resp:
            content      = complete_content
            metadata     = final_meta
            model        = final_meta.get("model", "gemini")
            tokens_used  = final_meta.get("tokens_used", input_tokens + output_tokens)
        logger.info(
        "LLM_FINAL_RESPONSE",
        response_length=len(complete_content),
        response_preview=complete_content[:500]
)
        return {"llm_response": _Resp()}

    async def _node_filter_output(self, state: LegalChatState) -> Dict[str, Any]:
        resp            = state.get("llm_response")
        answer          = resp.content if resp else ""
        intent_result   = state.get("intent_result")
        intent_val      = intent_result.intent.value if intent_result else Intent.LEGAL_QUERY.value
        pipeline_result = state.get("pipeline_result")
        include_sources = state.get("include_sources", True)

        cited = (
            self._pipeline.filter_cited_sources(answer, pipeline_result.sources)
            if pipeline_result else []
        )
        frontend_sources = cited[: self._max_sources] if include_sources else []
        result_data = {
            "message": answer,
            "sources": frontend_sources,
            "intent":  intent_val,
        }

        return {"final_output": result_data}

    async def _node_personality_response(self, state: LegalChatState) -> Dict[str, Any]:
        intent_result = state.get("intent_result")
        intent_val    = intent_result.intent.value if intent_result else Intent.CHITCHAT.value
        reply_text    = getattr(intent_result, "reply", None) if intent_result else None
        if not reply_text:
            reply_text = "أهلاً بك! أنا معين، كيف يمكنني مساعدتك في المسائل القانونية اليوم؟"

        class _Resp:
            content     = reply_text
            metadata    = {}
            model       = "precomputed"
            tokens_used = 0

        return {
            "llm_response": _Resp(),
            "final_output": {
                "message": reply_text,
                "sources": [],
                "intent":  intent_val,
            },
        }

    # ─────────────────────────────────────────
    # Cache helpers
    # ─────────────────────────────────────────

    async def _stream_cached_response(
        self, cached: Dict[str, Any], t_start: float
    ) -> AsyncIterator[str]:
        cached_answer  = cached.get("answer", "")
        cached_sources = cached.get("sources", [])

        words = cached_answer.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield self._sse("token", {"content": chunk})
            await asyncio.sleep(0)

        t_done = time.perf_counter()
        yield self._sse("sources", {
            "sources": cached_sources,
            "timing":  {"retrieval_ms": 0, "total_ms": round((t_done - t_start) * 1000, 1)},
            "intent":  Intent.LEGAL_QUERY.value,
        })

        yield self._format_sse_data("[DONE]")

    # ─────────────────────────────────────────
    # Tool builders
    # ─────────────────────────────────────────

    def _build_agent_tools(self, files_ids: List[str]) -> List[Any]:
        if not files_ids or self._file_service is None:
            return []

        allowed_set = set(files_ids)

        @tool("get_uploaded_files_content")
        async def get_uploaded_files_content(
            file_ids: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            """Fetch uploaded files text by IDs and return merged content."""
            target = file_ids or list(allowed_set)
            invalid = [fid for fid in target if fid not in allowed_set]
            if invalid:
                raise PayloadValidationError(
                    message="One or more file IDs are not allowed",
                    task_type=TaskType.LEGAL_CHAT.value,
                    details={"invalid_files_ids": invalid},
                )
            merged = await self._build_uploaded_files_context(target)
            return {"files_ids": target, "files_context": merged, "total_files": len(target)}

        return [get_uploaded_files_content]

    @staticmethod
    def _build_agent_system_prompt(files_ids: List[str]) -> str:
        return (
            "أنت مساعد قانوني مصري.\n"
            "مهمتك فقط:\n\n"
            "1. الإجابة على الأسئلة القانونية.\n"
            "2. تحليل الملفات المرفوعة والإجابة بناءً عليها.\n\n"
            "لا تنشئ عقوداً أو مذكرات أو صحف دعاوى أو مستندات قانونية.\n"
            "إذا طلب المستخدم إنشاء مستند قانوني فأخبره أن خدمة إنشاء المستندات متاحة في قسم مستقل."
        )

    # Static helpers
    # ─────────────────────────────────────────

    @staticmethod
    def _format_sse_data(data: str) -> str:
        return f"data: {data}\n\n"

    def _sse(self, event_type: str, payload: Dict[str, Any]) -> str:
        return self._format_sse_data(
            json.dumps({"type": event_type, **payload}, ensure_ascii=False)
        )

    @staticmethod
    def _stringify_exception(exc: BaseException) -> str:
        msg = str(exc).strip()
        return msg or repr(exc).strip() or f"{type(exc).__name__} with empty message"

    async def _build_uploaded_files_context(self, files_ids: List[str]) -> str:
        if not files_ids:
            return ""
        if self._file_service is None:
            raise FilesServiceError(
                message="Files service integration is not configured",
                details={"field": "file_service"},
            )
        try:
            files          = await self._file_service.fetch_files(files_ids)
            extracted_files = self._file_text_extractor.extract_many(files)
            logger.info(
                "files_processed_successfully",
                files_count=len(extracted_files),
                truncated_count=sum(1 for f in extracted_files if f.truncated),
            )
        except FileExtractionError as exc:
            raise PayloadValidationError(
                message="Unable to process one or more uploaded files",
                task_type="LEGAL_CHAT",
                details=exc.details,
            ) from exc

        chunks: List[str] = []
        for extracted in extracted_files:
            truncation_note = " [TRUNCATED]" if extracted.truncated else ""
            chunks.append(
                f"File ID: {extracted.file_id}\n"
                f"Filename: {extracted.filename}\n"
                f"Content Type: {extracted.content_type or 'unknown'}{truncation_note}\n"
                f"Content:\n{extracted.text}"
            )
        return "\n\n---\n\n".join(chunks)

    @staticmethod
    def _merge_contexts(legal_context: str, uploaded_files_context: str) -> str:
        legal_context           = (legal_context or "").strip()
        uploaded_files_context  = (uploaded_files_context or "").strip()
        if legal_context and uploaded_files_context:
            return (
                f"المواد القانونية ذات الصلة:\n{legal_context}\n\n"
                f"محتوى الملفات المرفوعة من المستخدم:\n{uploaded_files_context}"
            )
        return uploaded_files_context or legal_context

    @staticmethod
    def _normalize_file_ids(raw_value: Any) -> List[str]:
        if not isinstance(raw_value, list):
            return []
        seen: set = set()
        result: List[str] = []
        for item in raw_value:
            if not isinstance(item, str):
                continue
            value = item.strip()
            if value and value not in seen:
                seen.add(value)
                result.append(value)
        return result

    @staticmethod
    def _format_conversation_history(history: Any, limit: int = 8) -> str:
        if not isinstance(history, list) or not history:
            return "لا يوجد سجل محادثة سابق."
        lines: List[str] = []
        for item in history[-limit:]:
            if not isinstance(item, dict):
                continue
            role    = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            lines.append(f"{'المستخدم' if role == 'user' else 'المساعد'}: {content}")
        return "\n".join(lines) if lines else "لا يوجد سجل محادثة سابق."
