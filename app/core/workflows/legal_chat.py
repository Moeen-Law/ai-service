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

from app.core.services.llm_json import extract_first_json_object
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
        """Non-streaming execution path — returns a complete answer."""
        start_time = time.time()

        message = payload.get("message", "")
        retrieval_k = payload.get("retrieval_k", 4)
        conversation_history = payload.get("conversation_history", [])
        files_ids = self._normalize_file_ids(payload.get("files_ids"))

        # Prompt-only file generation path (Task 2)
        if self._is_generation_prompt(message) and not files_ids:
            return await self._generate_docx_from_prompt(
                task_id=task_id,
                context=context,
                message=message,
                start_time=start_time,
                options=options,
            )

        # Step 1: Classify intent
        classification = await self._intent_classifier.classify(message)
        logger.debug(
            f"Intent: {classification.intent.value} "
            f"(confidence: {classification.confidence})"
        )

        # Step 2: Route based on intent
        if classification.intent == Intent.CHITCHAT and not files_ids:
            return Result.success(
                task_id=UUID(task_id),
                task_type=TaskType.LEGAL_CHAT,
                data={
                    "message": self._build_chitchat_response(message),
                    "sources": [],
                    "intent": Intent.CHITCHAT.value,
                },
                metadata=ResultMetadata(
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    model_used="rule-based",
                ),
            )

        if classification.intent == Intent.OUT_OF_SCOPE and not files_ids:
            return Result.success(
                task_id=UUID(task_id),
                task_type=TaskType.LEGAL_CHAT,
                data={
                    "message": self._build_out_of_scope_response(),
                    "sources": [],
                    "intent": Intent.OUT_OF_SCOPE.value,
                },
                metadata=ResultMetadata(
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    model_used="rule-based",
                ),
            )

        if classification.intent == Intent.VAGUE and not files_ids:
            clarification = (
                classification.suggested_clarification
                or "تقصد بخصوص قانون العمل، أم قانون مدني، أم غيره؟"
            )
            return Result.success(
                task_id=UUID(task_id),
                task_type=TaskType.LEGAL_CHAT,
                data={
                    "message": f"السؤال غير واضح تماماً. {clarification}",
                    "sources": [],
                    "intent": Intent.VAGUE.value,
                },
                metadata=ResultMetadata(
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    model_used="rule-based",
                ),
            )

        # Step 3: Full legal query processing
        uploaded_files_context = await self._build_uploaded_files_context(files_ids)

        # If intent is legal_query, proceed with full pipeline
        # 1. Retrieval pipeline (pass precomputed rewrite data to avoid duplicate LLM call)
        pipeline_result = await self._pipeline.run(
            question=message,
            retrieval_k=retrieval_k,
            precomputed_domain=classification.domain,
            precomputed_keywords=classification.keywords,
            precomputed_articles=classification.likely_articles,
        )

        combined_context = self._merge_contexts(
            legal_context=pipeline_result.context,
            uploaded_files_context=uploaded_files_context,
        )

        if not combined_context:
            return Result.success(
                task_id=UUID(task_id),
                task_type=TaskType.LEGAL_CHAT,
                data={
                    "message": (
                        "عذراً، لم أتمكن من العثور على مواد قانونية ذات صلة بسؤالك. "
                        "يرجى إعادة صياغة السؤال أو التأكد من صحة المصطلحات المستخدمة."
                    ),
                    "sources": [],
                    "intent": Intent.LEGAL_QUERY.value,
                },
                metadata=ResultMetadata(
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    model_used="none",
                ),
            )

        # 2. Assemble prompt
        template = await self._prompt.get_template(
            task_type="LEGAL_CHAT",
            jurisdiction=context.jurisdiction.value,
            language=context.language.value,
        )
        assembled = await self._prompt.assemble_prompt(
            template=template,
            variables={
                "question": message,
                "context": combined_context,
                "conversation_history": self._format_conversation_history(
                    conversation_history
                ),
            },
        )

        # 3. Generate answer
        from app.interfaces.ai.llm_service import LLMRequest

        resp = await self._llm.generate(LLMRequest(prompt=assembled.prompt))
        answer = resp.content

        # 4. Filter cited sources
        cited = self._pipeline.filter_cited_sources(answer, pipeline_result.sources)
        frontend_sources = cited[: self._max_sources]

        execution_time_ms = int((time.time() - start_time) * 1000)

        return Result.success(
            task_id=UUID(task_id),
            task_type=TaskType.LEGAL_CHAT,
            data={
                "message": answer,
                "sources": frontend_sources,
                "intent": Intent.LEGAL_QUERY.value,
            },
            metadata=ResultMetadata(
                execution_time_ms=execution_time_ms,
                model_used=resp.model,
                tokens_used=resp.tokens_used,
            ),
        )

    async def _generate_docx_from_prompt(
        self,
        task_id: str,
        context: Context,
        message: str,
        start_time: float,
        options: Optional[ExecutionOptions],
    ) -> Result:
        """Generate document content from prompt and return mocked file IDs."""
        if self._file_generation_service is None:
            raise FileGenerationError(
                message="File generation service is not configured",
                details={"field": "file_generation_service"},
            )

        # Retrieve legal context to ground the generated draft.
        pipeline_result = await self._pipeline.run(
            question=f"صياغة مستند قانوني: {message}",
            retrieval_k=6,
        )

        template = await self._prompt.get_template(
            task_type="DOCUMENT_GENERATION",
            jurisdiction=context.jurisdiction.value,
            language=context.language.value,
        )
        assembled = await self._prompt.assemble_prompt(
            template=template,
            variables={
                "document_type": "docx",
                "parameters": message,
                "context": pipeline_result.context or "لا توجد مواد قانونية متاحة",
            },
        )

        from app.interfaces.ai.llm_service import LLMRequest

        structured_prompt = (
            f"{assembled.prompt}\n\n"
            "أعد النتيجة كـ JSON صالح فقط بدون أي نص إضافي وفق هذا الشكل:\n"
            '{"document_content":"..."}'
        )
        resp = await self._llm.generate(LLMRequest(prompt=structured_prompt))
        parsed = extract_first_json_object(resp.content)

        generated_content = str(resp.content).strip()
        if parsed:
            generated_content = (
                str(parsed.get("document_content", "")).strip() or generated_content
            )

        filename = self._build_generated_filename(message)
        generated_file = await self._file_generation_service.generate_docx(
            FileGenerationRequest(
                source_prompt=message,
                content=generated_content,
                filename=filename,
                metadata={"task_type": TaskType.LEGAL_CHAT.value, "mode": "generation"},
            )
        )

        include_sources = options.include_sources if options is not None else True
        frontend_sources = (
            pipeline_result.sources[: self._max_sources] if include_sources else []
        )

        execution_time_ms = int((time.time() - start_time) * 1000)
        return Result.success(
            task_id=UUID(task_id),
            task_type=TaskType.LEGAL_CHAT,
            data={
                "message": generated_content,
                "document_content": generated_content,
                "format": "docx",
                "files_ids": [generated_file.file_id],
                "generated_file": {
                    "file_id": generated_file.file_id,
                    "filename": generated_file.filename,
                    "content_type": generated_file.content_type,
                    "size_bytes": generated_file.size_bytes,
                },
                "sources": frontend_sources,
                "intent": "document_generation",
            },
            metadata=ResultMetadata(
                execution_time_ms=execution_time_ms,
                model_used=resp.model,
                tokens_used=resp.tokens_used,
            ),
        )

    @staticmethod
    def _is_generation_prompt(message: str) -> bool:
        """Heuristic intent detection for prompt-based document generation."""
        text = (message or "").strip().lower()
        if not text:
            return False

        action_patterns = [
            r"\bgenerate\b",
            r"\bcreate\b",
            r"\bdraft\b",
            r"\bwrite\b",
            r"انش",
            r"أنش",
            r"اكتب",
            r"صياغ",
            r"حرر",
            r"جهز",
            r"ولد",
        ]
        doc_patterns = [
            r"\bcontract\b",
            r"\bagreement\b",
            r"\bdocument\b",
            r"\bdocx\b",
            r"عقد",
            r"اتفاق",
            r"مستند",
            r"مذكرة",
            r"خطاب",
            r"لائحة",
        ]

        has_action = any(re.search(pattern, text) for pattern in action_patterns)
        has_doc_keyword = any(re.search(pattern, text) for pattern in doc_patterns)
        return has_action and has_doc_keyword

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
            # Prompt-only file generation path (mirrors non-streaming behavior)
            if self._is_generation_prompt(question) and not normalized_files_ids:
                if self._file_generation_service is None:
                    raise FileGenerationError(
                        message="File generation service is not configured",
                        details={"field": "file_generation_service"},
                    )

                pipeline_result = await self._pipeline.run(
                    question=f"صياغة مستند قانوني: {question}",
                    retrieval_k=6,
                )
                t_retrieval = time.perf_counter()

                template = await self._prompt.get_template(
                    task_type="DOCUMENT_GENERATION",
                    jurisdiction=jurisdiction,
                    language=language,
                )
                assembled = await self._prompt.assemble_prompt(
                    template=template,
                    variables={
                        "document_type": "docx",
                        "parameters": question,
                        "context": pipeline_result.context
                        or "لا توجد مواد قانونية متاحة",
                    },
                )

                from app.interfaces.ai.llm_service import LLMRequest

                structured_prompt = (
                    f"{assembled.prompt}\n\n"
                    "أعد النتيجة كـ JSON صالح فقط بدون أي نص إضافي وفق هذا الشكل:\n"
                    '{"document_content":"..."}'
                )
                resp = await self._llm.generate(LLMRequest(prompt=structured_prompt))
                parsed = extract_first_json_object(resp.content)

                generated_content = str(resp.content).strip()
                if parsed:
                    generated_content = (
                        str(parsed.get("document_content", "")).strip()
                        or generated_content
                    )

                filename = self._build_generated_filename(question)
                generated_file = await self._file_generation_service.generate_docx(
                    FileGenerationRequest(
                        source_prompt=question,
                        content=generated_content,
                        filename=filename,
                        metadata={
                            "task_type": TaskType.LEGAL_CHAT.value,
                            "mode": "generation",
                        },
                    )
                )
                t_done = time.perf_counter()

                frontend_sources = (
                    pipeline_result.sources[: self._max_sources]
                    if include_sources
                    else []
                )

                yield f"data: {json.dumps({'type': 'token', 'content': generated_content}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'generation', 'intent': 'document_generation', 'format': 'docx', 'files_ids': [generated_file.file_id], 'generated_file': {'file_id': generated_file.file_id, 'filename': generated_file.filename, 'content_type': generated_file.content_type, 'size_bytes': generated_file.size_bytes}}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'sources', 'sources': frontend_sources, 'timing': {'retrieval_ms': round((t_retrieval - t_start) * 1000, 1), 'total_ms': round((t_done - t_start) * 1000, 1)}, 'intent': 'document_generation'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Step 1: Classify intent
            classification = await self._intent_classifier.classify(question)
            logger.debug(
                f"Intent: {classification.intent.value} "
                f"(confidence: {classification.confidence})"
            )

            # Step 2: Route based on intent
            if (
                classification.intent
                in (Intent.CHITCHAT, Intent.OUT_OF_SCOPE, Intent.VAGUE)
                and not normalized_files_ids
            ):
                if classification.intent == Intent.CHITCHAT:
                    quick_reply = self._build_chitchat_response(question)
                elif classification.intent == Intent.OUT_OF_SCOPE:
                    quick_reply = self._build_out_of_scope_response()
                else:  # VAGUE
                    clarification = (
                        classification.suggested_clarification
                        or "تقصد بخصوص قانون العمل، أم قانون مدني، أم غيره؟"
                    )
                    quick_reply = f"السؤال غير واضح تماماً. {clarification}"

                yield f"data: {json.dumps({'type': 'token', 'content': quick_reply}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'intent': classification.intent.value}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Step 3: Full legal query processing
            uploaded_files_context = await self._build_uploaded_files_context(
                normalized_files_ids
            )

            pipeline_result = await self._pipeline.run(
                question=question,
                retrieval_k=retrieval_k,
                precomputed_domain=classification.domain,
                precomputed_keywords=classification.keywords,
                precomputed_articles=classification.likely_articles,
            )
            t_retrieval = time.perf_counter()

            combined_context = self._merge_contexts(
                legal_context=pipeline_result.context,
                uploaded_files_context=uploaded_files_context,
            )

            if not combined_context:
                no_result_msg = (
                    "عذراً، لم أتمكن من العثور على مواد قانونية ذات صلة بسؤالك. "
                    "يرجى إعادة صياغة السؤال أو التأكد من صحة المصطلحات المستخدمة."
                )
                yield f"data: {json.dumps({'type': 'token', 'content': no_result_msg}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'intent': Intent.LEGAL_QUERY.value}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            template = await self._prompt.get_template(
                "LEGAL_CHAT", jurisdiction, language
            )
            assembled = await self._prompt.assemble_prompt(
                template=template,
                variables={
                    "question": question,
                    "context": combined_context,
                    "conversation_history": self._format_conversation_history(
                        conversation_history
                    ),
                },
            )

            # Stream answer
            full_answer: list[str] = []
            t_stream_start = time.perf_counter()
            async for token in self._llm.astream(assembled.prompt):
                full_answer.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

            t_done = time.perf_counter()
            complete_answer = "".join(full_answer)

            cited = self._pipeline.filter_cited_sources(
                complete_answer, pipeline_result.sources
            )
            frontend_sources = cited[: self._max_sources] if include_sources else []

            timing = {
                "retrieval_ms": round((t_retrieval - t_start) * 1000, 1),
                "streaming_ms": round((t_done - t_stream_start) * 1000, 1),
                "total_ms": round((t_done - t_start) * 1000, 1),
            }

            yield f"data: {json.dumps({'type': 'sources', 'sources': frontend_sources, 'timing': timing, 'intent': Intent.LEGAL_QUERY.value}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as exc:
            logger.error("stream_error", error=str(exc))
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

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
