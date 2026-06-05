"""
Legal Chat Workflow

Full-pipeline workflow for conversational legal queries.
Uses intent classification, hybrid RAG retrieval, LLM query-rewriting, and SSE streaming.
"""
import asyncio
import json
import re
import time
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator, Dict, List, Optional, TypedDict
from uuid import UUID

from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool

from app.core.services.file_text_extractor import FileTextExtractor
from app.core.domain.entities import Context, ExecutionOptions, Result, ResultMetadata
from app.core.domain.enums import Intent, TaskStatus, TaskType
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
from app.interfaces.external.file_generation_service import (
    FileGenerationRequest,
    FileGenerationServiceInterface,
)
from app.interfaces.external.file_service import ExternalFileServiceInterface
from app.shared.errors.exceptions import (
    FileExtractionError,
    FileGenerationError,
    FilesServiceError,
    PayloadValidationError,
)
logger = get_logger(__name__)


@dataclass
class PreparedRequest:
    intent_result: IntentClassificationResult
    pipeline_result: Optional[PipelineResult] = None
    assembled_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    tools: Optional[List[Any]] = None


class LegalChatState(TypedDict, total=False):
    message: str
    files_ids: List[str]
    jurisdiction: str
    language: str
    retrieval_k: int
    conversation_history: Any
    include_sources: bool
    intent_result: Optional[IntentClassificationResult]
    pipeline_result: Optional[PipelineResult]
    uploaded_files_context: Optional[str]
    assembled_prompt: Optional[str]
    system_prompt: Optional[str]
    tools: Optional[List[Any]]
    llm_response: Optional[Any]
    final_output: Optional[Dict[str, Any]]


class LegalChatWorkflow(BaseWorkflow):
    """
    Workflow for LEGAL_CHAT task type.

    Pipeline:
    1. Intent Classification — determine if CHITCHAT, LEGAL_QUERY, VAGUE, or OUT_OF_SCOPE
    2. Route based on intent:
       - CHITCHAT/OUT_OF_SCOPE/VAGUE → Quick response
       - LEGAL_QUERY → Full pipeline:
          a. QueryPipeline.run() — rewrite → hybrid retrieve → inject → rerank → filter
          b. Assemble prompt with system instructions
          c. LLM generate (non-streaming) OR stream (streaming path)
          d. Filter cited sources
    """

    def __init__(
        self,
        rag_service: RAGServiceInterface,
        llm_service: LLMServiceInterface,
        prompt_service: PromptServiceInterface,
        file_service: Optional[ExternalFileServiceInterface] = None,
        file_text_extractor: Optional[FileTextExtractor] = None,
        file_generation_service: Optional[FileGenerationServiceInterface] = None,
        max_frontend_sources: int = 7,
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
        self._file_generation_service = file_generation_service
        self._max_sources = max_frontend_sources

        workflow = StateGraph(LegalChatState)
        workflow.add_node("classify_intent", self.classify_intent_node)
        workflow.add_node("fetch_files", self.fetch_files_node)
        workflow.add_node("run_rag_pipeline", self.run_rag_pipeline_node)
        workflow.add_node("merge_context", self.merge_context_node)
        workflow.add_node("assemble_prompt", self.assemble_prompt_node)
        workflow.add_node("build_tools", self.build_tools_node)
        workflow.add_node("llm_generate", self.llm_generate_node)
        workflow.add_node("filter_sources", self.filter_output_node)
        workflow.add_node("generation_output", self.filter_output_node)
        workflow.add_node("personality_response", self.personality_response_node)

        workflow.add_edge(START, "classify_intent")
        workflow.add_conditional_edges("classify_intent", self.route_by_intent)
        workflow.add_edge("fetch_files", "run_rag_pipeline")
        workflow.add_edge("run_rag_pipeline", "merge_context")
        workflow.add_edge("merge_context", "assemble_prompt")
        workflow.add_edge("assemble_prompt", "build_tools")
        workflow.add_edge("build_tools", "llm_generate")
        workflow.add_conditional_edges("llm_generate", self.route_output)
        workflow.add_edge("generation_output", END)
        workflow.add_edge("filter_sources", END)
        workflow.add_edge("personality_response", END)

        self.graph = workflow.compile()

    @property
    def name(self) -> str:
        return "LegalChatWorkflow"

    async def classify_intent_node(self, state: LegalChatState) -> Dict[str, Any]:
        files_ids = state.get("files_ids", [])
        question = state.get("message", "")
        conversation_history= state.get("conversation_history", [])
        formatted_history=self._format_conversation_history(conversation_history)
        intent_result = await self._intent_classifier.classify(question,history=formatted_history)
        if files_ids and intent_result.intent != Intent.LEGAL_QUERY:
            logger.info(
                "intent_overridden_due_to_files",
                original_intent=intent_result.intent.value,
                files_count=len(files_ids),
            )
            intent_result = IntentClassificationResult(
                intent=Intent.LEGAL_QUERY,
                history=formatted_history,
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
            reasoning=intent_result.reasoning,
        )
        return {"intent_result": intent_result}

    def route_by_intent(self, state: LegalChatState) -> str:
        intent_result = state.get("intent_result")
        if intent_result and intent_result.intent == Intent.LEGAL_QUERY:
            return "fetch_files"
        return "personality_response"

    async def fetch_files_node(self, state: LegalChatState) -> Dict[str, Any]:
        files_ids = state.get("files_ids")
        uploaded_context = None
        if files_ids and self._file_service is not None:
            uploaded_context = await self._build_uploaded_files_context(files_ids)
        return {"uploaded_files_context": uploaded_context}

    async def run_rag_pipeline_node(self, state: LegalChatState) -> Dict[str, Any]:
        question = state.get("message", "")
        retrieval_k = state.get("retrieval_k", 4)
        intent_result = state.get("intent_result")

        pipeline_result = await self._pipeline.run(
            question=question,
            retrieval_k=retrieval_k,
            precomputed_domain=intent_result.domain if intent_result else None,
            precomputed_keywords=intent_result.keywords if intent_result else [],
            precomputed_articles=intent_result.likely_articles if intent_result else [],
            skip_rewrite=True,
        )
        return {"pipeline_result": pipeline_result}

    async def merge_context_node(self, state: LegalChatState) -> Dict[str, Any]:
        pipeline_result = state.get("pipeline_result")
        uploaded_context = state.get("uploaded_files_context")

        if uploaded_context and pipeline_result:
            pipeline_result = replace(
                pipeline_result,
                context=self._merge_contexts(pipeline_result.context, uploaded_context)
            )
        return {"pipeline_result": pipeline_result}

    async def assemble_prompt_node(self, state: LegalChatState) -> Dict[str, Any]:
        question = state.get("message", "")
        jurisdiction = state.get("jurisdiction", "egypt")
        language = state.get("language", "ar")
        pipeline_result = state.get("pipeline_result")
        conversation_history = state.get("conversation_history", [])

        template = await self._prompt.get_template(
            task_type="LEGAL_CHAT",
            jurisdiction=jurisdiction,
            language=language,
        )
        assembled = await self._prompt.assemble_prompt(
            template=template,
            variables={
                "question": question,
                "context": pipeline_result.context if pipeline_result else "",
                "conversation_history": self._format_conversation_history(
                    conversation_history
                ),
            },
        )
        return {"assembled_prompt": assembled.prompt,"system_prompt": assembled.system_prompt}

    async def build_tools_node(self, state: LegalChatState) -> Dict[str, Any]:
        files_ids = state.get("files_ids", [])
        question = state.get("message", "")
        base_system_prompt = state.get("system_prompt", "")
        tools = self._build_agent_tools(files_ids=files_ids, source_prompt=question)
        agent_system_prompt = self._build_agent_system_prompt(files_ids=files_ids)
        combined_system_prompt = f"{base_system_prompt}\n\n=== تعليمات استخدام الأدوات ===\n{agent_system_prompt}"

        return {
            "tools": tools,
            "system_prompt": combined_system_prompt
        }

    async def llm_generate_node(self, state: LegalChatState) -> Dict[str, Any]:
        assembled_prompt = state.get("assembled_prompt", "")
        system_prompt = state.get("system_prompt", "")
        tools = state.get("tools", [])

        complete_content = ""
        final_meta = {}

        async for chunk, is_final, metadata in self._llm.stream_with_tools(
            LLMRequest(
                prompt=assembled_prompt,
                system_prompt=system_prompt,
            ),
            tools=tools,
        ):
            if chunk:
                complete_content += chunk
            if is_final:
                final_meta = metadata or {}

        class _MockResp:
            content = complete_content
            metadata = final_meta
            model = final_meta.get("model", "streaming")
            tokens_used = final_meta.get("tokens_used", 0)

        return {"llm_response": _MockResp()}

    def route_output(self, state: LegalChatState) -> str:
        resp = state.get("llm_response")
        generation_payload = self._extract_generation_tool_payload(resp.metadata) if resp else None
        if generation_payload is not None:
            return "generation_output"
        return "filter_sources"

    async def filter_output_node(self, state: LegalChatState) -> Dict[str, Any]:
        resp = state.get("llm_response")
        answer = resp.content if resp else ""
        intent_result = state.get("intent_result")
        intent_val = intent_result.intent.value if intent_result else Intent.LEGAL_QUERY.value
        pipeline_result = state.get("pipeline_result")
        include_sources = state.get("include_sources", True)

        generation_payload = self._extract_generation_tool_payload(resp.metadata) if resp else None
        if generation_payload is not None:
            frontend_sources = (
                pipeline_result.sources[: self._max_sources] if pipeline_result and include_sources else []
            )
            result_data = {
                "message": generation_payload.get("document_content") or answer,
                "document_content": generation_payload.get("document_content") or answer,
                "format": generation_payload.get("format", "docx"),
                "files_ids": generation_payload.get("files_ids", []),
                "generated_file": {
                    "file_id": generation_payload.get("file_id"),
                    "filename": generation_payload.get("filename"),
                    "content_type": generation_payload.get("content_type"),
                    "size_bytes": generation_payload.get("size_bytes"),
                },
                "sources": frontend_sources,
                "intent": "document_generation",
            }
        else:
            cited = self._pipeline.filter_cited_sources(answer, pipeline_result.sources) if pipeline_result else []
            frontend_sources = cited[: self._max_sources] if include_sources else []
            result_data = {
                "message": answer,
                "sources": frontend_sources,
                "intent": intent_val if not frontend_sources else Intent.LEGAL_QUERY.value,
            }

        return {"final_output": result_data}

    async def personality_response_node(self, state: LegalChatState) -> Dict[str, Any]:
        intent_result = state.get("intent_result")
        intent_val = intent_result.intent.value if intent_result else Intent.CHITCHAT.value

        reply_text = getattr(intent_result, 'reply', None) if intent_result else None
        if not reply_text:
            reply_text = "أهلاً بك! أنا معين، كيف يمكنني مساعدتك في المسائل القانونية اليوم؟"

        class _MockResp:
            content = reply_text
            metadata = {}
            model = "precomputed"
            tokens_used = 0

        result_data = {
            "message": reply_text,
            "sources": [],
            "intent": intent_val,
        }

        return {
            "llm_response": _MockResp(),
            "final_output": result_data
        }

    async def execute(
        self,
        task_id: str,
        context: Context,
        payload: Dict[str, Any],
        options: Optional[ExecutionOptions] = None,
    ) -> Result:
        """Non-streaming execution path using model-driven tool selection."""
        start_time = time.time()

        message = payload.get("message", "")
        retrieval_k = payload.get("retrieval_k", 4)
        conversation_history = payload.get("conversation_history", [])
        files_ids = self._normalize_file_ids(payload.get("files_ids"))

        if files_ids:
            logger.info(
                "processing_uploaded_files",
                files_ids=files_ids,
                files_count=len(files_ids),
            )

        if files_ids and self._file_service is None:
            raise FilesServiceError(
                message="Files service integration is not configured",
                details={"field": "file_service"},
            )

        include_sources = options.include_sources if options is not None else True

        state = LegalChatState(
            message=message,
            files_ids=files_ids,
            jurisdiction=context.jurisdiction.value,
            language=context.language.value,
            retrieval_k=retrieval_k,
            conversation_history=conversation_history,
            include_sources=include_sources,
        )

        result_state = await self.graph.ainvoke(state)
        final_output = result_state.get("final_output", {})
        resp = result_state.get("llm_response")

        execution_time_ms = int((time.time() - start_time) * 1000)

        return Result.success(
            task_id=UUID(task_id),
            task_type=TaskType.LEGAL_CHAT,
            data=final_output,
            metadata=ResultMetadata(
                execution_time_ms=execution_time_ms,
                model_used=resp.model if resp else None,
                tokens_used=resp.tokens_used if resp else None,
            ),
        )

    def _build_agent_tools(self, files_ids: List[str], source_prompt: str) -> List[Any]:
        """Build model-callable tools for LEGAL_CHAT analysis/generation."""
        tools: List[Any] = []

        if files_ids and self._file_service is not None:
            allowed_file_ids = list(files_ids)
            allowed_file_ids_set = set(allowed_file_ids)

            @tool("get_uploaded_files_content")
            async def get_uploaded_files_content(
                file_ids: Optional[List[str]] = None,
            ) -> Dict[str, Any]:
                """Fetch uploaded files text by IDs from payload.files_ids and return merged content."""
                target_file_ids = file_ids or allowed_file_ids
                invalid_file_ids = [
                    file_id
                    for file_id in target_file_ids
                    if file_id not in allowed_file_ids_set
                ]
                if invalid_file_ids:
                    raise PayloadValidationError(
                        message="One or more file IDs are not allowed for this request",
                        task_type=TaskType.LEGAL_CHAT.value,
                        details={"invalid_files_ids": invalid_file_ids},
                    )

                merged_context = await self._build_uploaded_files_context(
                    target_file_ids
                )
                return {
                    "files_ids": target_file_ids,
                    "files_context": merged_context,
                    "total_files": len(target_file_ids),
                }

            tools.append(get_uploaded_files_content)

        if self._file_generation_service is not None:

            @tool("generate_docx")
            async def generate_docx(
                document_content: str,
                filename: Optional[str] = None,
            ) -> Dict[str, Any]:
                """Generate DOCX output from provided legal document content."""
                clean_content = (document_content or "").strip()
                if not clean_content:
                    raise FileGenerationError(
                        message="document_content must not be empty",
                        details={"field": "document_content"},
                    )

                final_filename = (
                    filename or self._build_generated_filename(source_prompt)
                ).strip()
                if not final_filename:
                    final_filename = self._build_generated_filename(source_prompt)

                generated_file = await self._file_generation_service.generate_docx(
                    FileGenerationRequest(
                        source_prompt=source_prompt,
                        content=clean_content,
                        filename=final_filename,
                        metadata={
                            "task_type": TaskType.LEGAL_CHAT.value,
                            "mode": "generation",
                        },
                    )
                )

                return {
                    "document_content": clean_content,
                    "format": "docx",
                    "files_ids": [generated_file.file_id],
                    "file_id": generated_file.file_id,
                    "filename": generated_file.filename,
                    "content_type": generated_file.content_type,
                    "size_bytes": generated_file.size_bytes,
                }

            tools.append(generate_docx)

        return tools

    @staticmethod
    def _build_agent_system_prompt(files_ids: List[str]) -> str:
        files_hint = ", ".join(files_ids) if files_ids else "(no uploaded files)"
        if files_ids:
            files_section = (
                f"الملفات المرفوعة (File IDs): {files_hint}\n"
                "- محتوى هذه الملفات متاح بالفعل في الـ context أمامك.\n"
                "- لا تستدعي get_uploaded_files_content إلا إذا طلب المستخدم صراحةً "
                "تحليل جزء محدد أو مقارنة بين ملفات.\n"
            )
        else:
            files_section = "- لا توجد ملفات مرفوعة in هذا الطلب.\n"
        return (
            "أنت مساعد قانوني محترف. اتخذ قرار استخدام الأدوات بناءً على طلب المستخدم فقط.\n\n"
            f"{files_section}\n"
            "الأدوات المتاحة:\n"
            "1) get_uploaded_files_content: لجلب محتوى ملفات إضافية عند الحاجة.\n"
            "2) generate_docx: لإنشاء ملف DOCX عندما يطلب المستخدم صياغة مستند.\n\n"
            "قواعد التشغيل:\n"
            "- لا تخترع File IDs غير الموجودة في القائمة المتاحة.\n"
            "- إذا طُلب إنشاء مستند، أنشئ المحتوى أولاً ثم استخدم generate_docx.\n"
            "- أعد إجابة نهائية واضحة للمستخدم بعد أي استدعاءات أدوات.\n"
        )

    @staticmethod
    def _extract_generation_tool_payload(
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not metadata:
            return None

        invocations = metadata.get("tool_invocations")
        if not isinstance(invocations, list):
            return None

        for invocation in reversed(invocations):
            if not isinstance(invocation, dict):
                continue
            if invocation.get("name") != "generate_docx":
                continue
            result = invocation.get("result")
            if isinstance(result, dict):
                return result

        return None

    @staticmethod
    def _format_sse_data(data: str) -> str:
        return f"data: {data}\n\n"

    @staticmethod
    def _stringify_exception(exc: BaseException) -> str:
        message = str(exc).strip()
        if message:
            return message
        repr_value = repr(exc).strip()
        if repr_value:
            return repr_value
        return f"{type(exc).__name__} with empty message"

    @staticmethod
    def _build_generated_filename(message: str) -> str:
        """Build a deterministic filename from prompt text and timestamp."""
        clean = re.sub(r"[^\w\s-]", "", (message or "").strip().lower())
        tokens = [token for token in clean.split() if token]
        slug = "_".join(tokens[:6]) if tokens else "generated_document"
        slug = slug[:40] if slug else "generated_document"
        timestamp = int(time.time())
        return f"{slug}_{timestamp}.docx"

    async def _build_uploaded_files_context(self, files_ids: List[str]) -> str:
        """Fetch uploaded files by IDs and convert them into prompt-ready text."""
        if not files_ids:
            return ""

        if self._file_service is None:
            raise FilesServiceError(
                message="Files service integration is not configured",
                details={"field": "file_service"},
            )

        try:
            logger.debug(
                "fetching_uploaded_files",
                files_ids=files_ids,
            )
            files = await self._file_service.fetch_files(files_ids)

            logger.debug(
                "extracting_file_text",
                files_count=len(files),
                total_size_bytes=sum(f.size_bytes for f in files),
            )
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
                (
                    f"File ID: {extracted.file_id}\n"
                    f"Filename: {extracted.filename}\n"
                    f"Content Type: {extracted.content_type or 'unknown'}{truncation_note}\n"
                    f"Content:\n{extracted.text}"
                )
            )

        return "\n\n---\n\n".join(chunks)

    @staticmethod
    def _merge_contexts(legal_context: str, uploaded_files_context: str) -> str:
        """Combine legal-RAG context with uploaded-file context for prompt assembly."""
        legal_context = (legal_context or "").strip()
        uploaded_files_context = (uploaded_files_context or "").strip()

        if legal_context and uploaded_files_context:
            return (
                f"المواد القانونية ذات الصلة:\n{legal_context}\n\n"
                "محتوى الملفات المرفوعة من المستخدم:\n"
                f"{uploaded_files_context}"
            )

        if uploaded_files_context:
            return f"محتوى الملفات المرفوعة من المستخدم:\n{uploaded_files_context}"

        return legal_context

    @staticmethod
    def _normalize_file_ids(raw_value: Any) -> List[str]:
        """Normalize and de-duplicate file IDs while preserving input order."""
        if not isinstance(raw_value, list):
            return []

        normalized: List[str] = []
        seen = set()
        for item in raw_value:
            if not isinstance(item, str):
                continue
            value = item.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    # ------------------------------------------------------------------
    # SSE streaming path (called from the streaming route, not execute)
    # ------------------------------------------------------------------

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
        """Stream legal chat response directly without graph overhead."""
        logger.info(
            "legal_chat_stream_started",
            question=question[:100],
            files_ids=files_ids,
            files_count=len(files_ids) if files_ids else 0,
        )
        t_start = time.perf_counter()
        normalized_files_ids = self._normalize_file_ids(files_ids)

        try:
            if normalized_files_ids and self._file_service is None:
                raise FilesServiceError(
                    message="Files service integration is not configured",
                    details={"field": "file_service"},
                )

            # 1) Intent classification
            formatted_history = self._format_conversation_history(conversation_history or [])
            intent_result = await self._intent_classifier.classify(question, history=formatted_history)
            logger.info(f"Intent classification result: {intent_result.intent.value} (confidence: {intent_result.confidence}, reasoning: {intent_result.reasoning})")
            if normalized_files_ids and intent_result.intent != Intent.LEGAL_QUERY:
                intent_result = IntentClassificationResult(
                    intent=Intent.LEGAL_QUERY,
                    confidence=1.0,
                    domain=intent_result.domain,
                    keywords=intent_result.keywords,
                    likely_articles=intent_result.likely_articles,
                    reasoning="Overridden: uploaded files present",
                )

            if intent_result.intent != Intent.LEGAL_QUERY:
                reply_text = getattr(intent_result, 'reply', None)
                if not reply_text:
                    reply_text = "أهلاً بك! أنا معين، كيف يمكنني مساعدتك في المسائل القانونية اليوم؟"

                logger.info(f"Full LLM Reply Before Streaming: {reply_text}")

                async def mock_gemini_stream(text: str):
                    words = text.split(" ")
                    for i, word in enumerate(words):
                        chunk = word + " " if i < len(words) - 1 else word
                        yield chunk
                        await asyncio.sleep(0.05)

                async_stream = mock_gemini_stream(reply_text)
                async for chunk in async_stream:
                    if chunk:
                        yield self._format_sse_data(
                            json.dumps({"type": "token", "content": chunk}, ensure_ascii=False)
                        )

                t_done = time.perf_counter()
                timing = {
                    "retrieval_ms": 0,
                    "total_ms": round((t_done - t_start) * 1000, 1),
                }
                yield self._format_sse_data(
                    json.dumps({
                        "type": "sources",
                        "sources": [],
                        "timing": timing,
                        "intent": intent_result.intent.value,
                    }, ensure_ascii=False)
                )
                yield self._format_sse_data("[DONE]")
                return

            # 2) Full Pipeline for LEGAL_QUERY
            # a) File Extraction
            uploaded_context = None
            if normalized_files_ids:
                uploaded_context = await self._build_uploaded_files_context(normalized_files_ids)

            # b) RAG Pipeline
            pipeline_result = await self._pipeline.run(
                question=question,
                retrieval_k=retrieval_k,
                precomputed_domain=intent_result.domain,
                precomputed_keywords=intent_result.keywords,
                precomputed_articles=intent_result.likely_articles,
                skip_rewrite=True,
            )
            t_retrieval = time.perf_counter()

            # c) Merge Contexts
            merged_context = ""
            if pipeline_result:
                merged_context = self._merge_contexts(pipeline_result.context, uploaded_context or "")
            else:
                merged_context = uploaded_context or ""

            # d) Assemble Prompt
            template = await self._prompt.get_template(
                task_type="LEGAL_CHAT",
                jurisdiction=jurisdiction,
                language=language,
            )
            assembled = await self._prompt.assemble_prompt(
                template=template,
                variables={
                    "question": question,
                    "context": merged_context,
                    "conversation_history": self._format_conversation_history(conversation_history or []),
                },
            )

            # e) Tools Setup
            if normalized_files_ids:
                tools = self._build_agent_tools(files_ids=normalized_files_ids, source_prompt=question)
            else:
                tools=[]
            agent_instructions = self._build_agent_system_prompt(files_ids=normalized_files_ids)
            base_system_prompt = assembled.system_prompt or ""

            combined_system_prompt = f"{base_system_prompt}\n\n=== تعليمات استخدام الأدوات ===\n{agent_instructions}"

            # f) Stream Response with tools integration
            final_meta = {}
            complete_answer = ""

            llm_stream_start = time.perf_counter()
            first_token_logged = False

            async for chunk, is_final, metadata in self._llm.stream_with_tools(
                LLMRequest(
                    prompt=assembled.prompt,
                    system_prompt=combined_system_prompt,
                ),
                tools=tools,
            ):
                if chunk:
                    if not first_token_logged:
                        logger.info(
                            f"[LLM TIME] Time to FIRST TOKEN: {time.perf_counter() - llm_stream_start:.4f} seconds")
                        first_token_logged = True

                    complete_answer += chunk
                    yield self._format_sse_data(
                        json.dumps({"type": "token", "content": chunk}, ensure_ascii=False)
                    )
                if is_final:
                    final_meta = metadata or {}
            logger.info(
                    f"[LLM TIME] Total LLM Generation Time: {time.perf_counter() - llm_stream_start:.4f} seconds")
            from app.shared.utils.token_calculator import TokenCostCalculator
            input_tokens = TokenCostCalculator.estimate_tokens((assembled.prompt or "") + (combined_system_prompt or ""))
            output_tokens = TokenCostCalculator.estimate_tokens(complete_answer)
            cost_per_1k = TokenCostCalculator.calculate_cost_per_1k(input_tokens, output_tokens)

            logger.info(
                f"[TOKEN USAGE] Input: {input_tokens} tokens | Output: {output_tokens} tokens | Total: {input_tokens + output_tokens} tokens"
            )
            logger.info(f"[COST ESTIMATE - Gemini 2.5 Flash] Cost for 1,000 queries: ${cost_per_1k:.4f}")
            # Parse tool response if any (document generation check)
            t_done = time.perf_counter()
            generation_payload = self._extract_generation_tool_payload(final_meta)
            frontend_sources = []
            final_intent = intent_result.intent.value

            # g) Send Sources and final resolution
            if generation_payload is not None:
                final_intent = "document_generation"
                generation_event = {
                    "type": "generation",
                    "intent": "document_generation",
                    "format": generation_payload.get("format", "docx"),
                    "files_ids": generation_payload.get("files_ids", []),
                    "generated_file": {
                        "file_id": generation_payload.get("file_id"),
                        "filename": generation_payload.get("filename"),
                        "content_type": generation_payload.get("content_type"),
                        "size_bytes": generation_payload.get("size_bytes"),
                    },
                }
                yield self._format_sse_data(
                    json.dumps(generation_event, ensure_ascii=False)
                )
            else:
                if pipeline_result and include_sources:
                    cited = self._pipeline.filter_cited_sources(complete_answer, pipeline_result.sources)
                    frontend_sources = cited[: self._max_sources]
                if not frontend_sources:
                    final_intent = Intent.LEGAL_QUERY.value

            timing = {
                "retrieval_ms": round((t_retrieval - t_start) * 1000, 1),
                "total_ms": round((t_done - t_start) * 1000, 1),
            }

            yield self._format_sse_data(
                json.dumps({
                    "type": "sources",
                    "sources": frontend_sources,
                    "timing": timing,
                    "intent": final_intent,
                }, ensure_ascii=False)
            )

            logger.info("stream_ended_successfully", final_intent=final_intent)
            yield self._format_sse_data("[DONE]")

        except Exception as exc:
            error_message = self._stringify_exception(exc)
            logger.error(
                "stream_error",
                error=error_message,
                error_type=type(exc).__name__,
            )
            yield self._format_sse_data(
                json.dumps(
                    {"type": "error", "content": error_message},
                    ensure_ascii=False,
                )
            )
            yield self._format_sse_data("[DONE]")

    @staticmethod
    def _format_conversation_history(history: Any, limit: int = 8) -> str:
        if not isinstance(history, list) or not history:
            return "لا يوجد سجل محادثة سابق."

        recent = history[-limit:]
        lines: list[str] = []
        for item in recent:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            role_label = "المستخدم" if role == "user" else "المساعد"
            lines.append(f"{role_label}: {content}")

        if not lines:
            return "لا يوجد سجل محادثة سابق."

        return "\n".join(lines)