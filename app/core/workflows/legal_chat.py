"""
Legal Chat Workflow

Full-pipeline workflow for conversational legal queries.
Uses intent classification, hybrid RAG retrieval, LLM query-rewriting, and SSE streaming.
"""

import json
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import UUID

from langchain_core.tools import tool

from app.core.services.file_text_extractor import FileTextExtractor
from app.core.domain.entities import Context, ExecutionOptions, Result, ResultMetadata
from app.core.domain.enums import Intent, TaskStatus, TaskType
from app.core.services.query_pipeline import QueryPipeline
from app.core.validators.intent_classifier import IntentClassifier
from app.core.workflows.base import BaseWorkflow
from app.infrastructure.logging.logger import get_logger
from app.interfaces.ai.llm_service import LLMServiceInterface
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
        # Keep legacy patterns as fallback
        self._social_only_patterns = [
            r"^\s*(شكرا|شكرًا|متشكر|تسلم|تمام|اوك|أوك|موافق|ماشي|اهلا|أهلا|سلام|باي|مع السلامة|thanks|ok)\s*[!.؟?]*\s*$",
        ]
        self._legal_cue_patterns = [
            r"قانون",
            r"مادة",
            r"مواد",
            r"عقوبة",
            r"جريمة",
            r"دعوى",
            r"محكمة",
            r"عقد",
            r"طلاق",
            r"نفقة",
            r"ميراث",
            r"جنحة",
            r"جناية",
        ]

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
        """Non-streaming execution path using model-driven tool selection."""
        start_time = time.time()

        message = payload.get("message", "")
        retrieval_k = payload.get("retrieval_k", 4)
        conversation_history = payload.get("conversation_history", [])
        files_ids = self._normalize_file_ids(payload.get("files_ids"))

        if files_ids and self._file_service is None:
            raise FilesServiceError(
                message="Files service integration is not configured",
                details={"field": "file_service"},
            )

        pipeline_result = await self._pipeline.run(
            question=message,
            retrieval_k=retrieval_k,
        )

        template = await self._prompt.get_template(
            task_type="LEGAL_CHAT",
            jurisdiction=context.jurisdiction.value,
            language=context.language.value,
        )
        assembled = await self._prompt.assemble_prompt(
            template=template,
            variables={
                "question": message,
                "context": pipeline_result.context,
                "conversation_history": self._format_conversation_history(
                    conversation_history
                ),
            },
        )

        from app.interfaces.ai.llm_service import LLMRequest

        tools = self._build_agent_tools(files_ids=files_ids, source_prompt=message)
        include_sources = options.include_sources if options is not None else True

        resp = await self._llm.generate_with_tools(
            LLMRequest(
                prompt=assembled.prompt,
                system_prompt=self._build_agent_system_prompt(files_ids=files_ids),
            ),
            tools=tools,
        )
        answer = resp.content

        generation_payload = self._extract_generation_tool_payload(resp.metadata)
        if generation_payload is not None:
            frontend_sources = (
                pipeline_result.sources[: self._max_sources] if include_sources else []
            )
            result_data = {
                "message": generation_payload.get("document_content") or answer,
                "document_content": generation_payload.get("document_content")
                or answer,
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
            cited = self._pipeline.filter_cited_sources(answer, pipeline_result.sources)
            frontend_sources = cited[: self._max_sources] if include_sources else []
            result_data = {
                "message": answer,
                "sources": frontend_sources,
                "intent": Intent.LEGAL_QUERY.value,
            }

        execution_time_ms = int((time.time() - start_time) * 1000)

        return Result.success(
            task_id=UUID(task_id),
            task_type=TaskType.LEGAL_CHAT,
            data=result_data,
            metadata=ResultMetadata(
                execution_time_ms=execution_time_ms,
                model_used=resp.model,
                tokens_used=resp.tokens_used,
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
        return (
            "أنت مساعد قانوني محترف. اتخذ قرار استخدام الأدوات بناء على طلب المستخدم فقط.\n"
            "الأدوات المتاحة:\n"
            "1) get_uploaded_files_content: لتحليل الملفات المرفوعة.\n"
            "2) generate_docx: لإنشاء ملف DOCX عندما يطلب المستخدم صياغة مستند.\n"
            f"File IDs المتاحة من payload.files_ids: {files_hint}\n"
            "قواعد التشغيل:\n"
            "- إذا طُلب منك تحليل ملفات مرفوعة، استخدم get_uploaded_files_content قبل الإجابة.\n"
            "- لا تخترع File IDs غير الموجودة في القائمة المتاحة.\n"
            "- إذا طُلب إنشاء مستند، أنشئ المحتوى أولا ثم استخدم generate_docx.\n"
            "- أعد إجابة نهائية واضحة للمستخدم بعد أي استدعاءات أدوات."
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
    def _iter_stream_tokens(text: str) -> List[str]:
        return [match.group(0) for match in re.finditer(r"\S+|\s+", text or "")]

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
            files = await self._file_service.fetch_files(files_ids)
            extracted_files = self._file_text_extractor.extract_many(files)
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
        """
        Async generator that yields SSE-formatted events:
          - ``{type: "token", content: "..."}`` for each token
          - ``{type: "sources", sources: [...], timing: {...}}`` at the end
          - ``{type: "generation", files_ids: [...], ...}`` for generation mode
          - ``[DONE]`` sentinel
        """
        t_start = time.perf_counter()
        normalized_files_ids = self._normalize_file_ids(files_ids)

        try:
            if normalized_files_ids and self._file_service is None:
                raise FilesServiceError(
                    message="Files service integration is not configured",
                    details={"field": "file_service"},
                )

            pipeline_result = await self._pipeline.run(
                question=question,
                retrieval_k=retrieval_k,
            )
            t_retrieval = time.perf_counter()

            template = await self._prompt.get_template(
                "LEGAL_CHAT", jurisdiction, language
            )
            assembled = await self._prompt.assemble_prompt(
                template=template,
                variables={
                    "question": question,
                    "context": pipeline_result.context,
                    "conversation_history": self._format_conversation_history(
                        conversation_history
                    ),
                },
            )

            from app.interfaces.ai.llm_service import LLMRequest

            tools = self._build_agent_tools(
                files_ids=normalized_files_ids,
                source_prompt=question,
            )
            resp = await self._llm.generate_with_tools(
                LLMRequest(
                    prompt=assembled.prompt,
                    system_prompt=self._build_agent_system_prompt(
                        files_ids=normalized_files_ids
                    ),
                ),
                tools=tools,
            )

            complete_answer = (resp.content or "").strip()
            if not complete_answer:
                complete_answer = "عذرا، لم أتمكن من توليد إجابة مناسبة الآن."
            generation_payload = self._extract_generation_tool_payload(resp.metadata)
            stream_content = complete_answer
            if generation_payload is not None:
                doc_content = generation_payload.get("document_content")
                if isinstance(doc_content, str) and doc_content.strip():
                    stream_content = doc_content

            for token in self._iter_stream_tokens(stream_content):
                yield self._format_sse_data(
                    json.dumps({"type": "token", "content": token}, ensure_ascii=False)
                )

            t_done = time.perf_counter()
            if generation_payload is not None:
                frontend_sources = (
                    pipeline_result.sources[: self._max_sources]
                    if include_sources
                    else []
                )
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
                intent_value = "document_generation"
            else:
                cited = self._pipeline.filter_cited_sources(
                    complete_answer, pipeline_result.sources
                )
                frontend_sources = cited[: self._max_sources] if include_sources else []
                intent_value = Intent.LEGAL_QUERY.value

            timing = {
                "retrieval_ms": round((t_retrieval - t_start) * 1000, 1),
                "total_ms": round((t_done - t_start) * 1000, 1),
            }

            yield self._format_sse_data(
                json.dumps(
                    {
                        "type": "sources",
                        "sources": frontend_sources,
                        "timing": timing,
                        "intent": intent_value,
                    },
                    ensure_ascii=False,
                )
            )
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

    def _is_social_only_message(self, message: str) -> bool:
        """Legacy fallback pattern matching."""
        text = (message or "").strip().lower()
        if not text:
            return False

        has_legal_cue = any(
            re.search(pattern, text) for pattern in self._legal_cue_patterns
        )
        if has_legal_cue:
            return False

        return any(re.match(pattern, text) for pattern in self._social_only_patterns)

    def _build_chitchat_response(self, message: str) -> str:
        """Generate a friendly response for chitchat messages."""
        text = (message or "").strip().lower()
        if re.search(r"شكرا|شكرًا|متشكر|thanks|تسلم", text):
            return (
                "العفو، تحت أمرك في أي وقت. لو حابب نكمل في أي نقطة قانونية أنا معاك."
            )
        if re.search(r"اهلا|أهلا", text):
            return "أهلا بيك، منور. احكي لي سؤالك القانوني وأنا أساعدك خطوة بخطوة."
        if re.search(r"باي|مع السلامة|سلام", text):
            return "مع السلامة، وفي أي وقت تحتاج استشارة قانونية أنا موجود."
        return "تمام، أنا معاك. ابعت سؤالك القانوني أو التفاصيل اللي تحب نكمل عليها."

    def _build_out_of_scope_response(self) -> str:
        """Generate a response for out-of-scope messages."""
        return (
            "عذراً، أنا متخصص في الاستشارات القانونية والقوانين المصرية. "
            "لو عندك سؤال قانوني أو استفسار عن حقوقك، أنا هنا لمساعدتك."
        )

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
