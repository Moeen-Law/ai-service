"""
Legal Chat Workflow

Full-pipeline workflow for conversational legal queries.
Uses intent classification, hybrid RAG retrieval, LLM query-rewriting, and SSE streaming.
"""

import json
import re
import time
from typing import Any, AsyncIterator, Dict, Optional
from uuid import UUID

from app.core.domain.entities import Context, ExecutionOptions, Result, ResultMetadata
from app.core.domain.enums import Intent, TaskStatus, TaskType
from app.core.services.query_pipeline import QueryPipeline
from app.core.validators.intent_classifier import IntentClassifier
from app.core.workflows.base import BaseWorkflow
from app.infrastructure.logging.logger import get_logger
from app.interfaces.ai.llm_service import LLMServiceInterface
from app.interfaces.ai.prompt_service import PromptServiceInterface
from app.interfaces.ai.rag_service import RAGServiceInterface

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

        # Step 1: Classify intent
        classification = await self._intent_classifier.classify(message)
        logger.debug(
            f"Intent: {classification.intent.value} "
            f"(confidence: {classification.confidence})"
        )

        # Step 2: Route based on intent
        if classification.intent == Intent.CHITCHAT:
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

        if classification.intent == Intent.OUT_OF_SCOPE:
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

        if classification.intent == Intent.VAGUE:
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
        # If intent is legal_query, proceed with full pipeline
        # 1. Retrieval pipeline (pass precomputed rewrite data to avoid duplicate LLM call)
        pipeline_result = await self._pipeline.run(
            question=message,
            retrieval_k=retrieval_k,
            precomputed_domain=classification.domain,
            precomputed_keywords=classification.keywords,
            precomputed_articles=classification.likely_articles,
        )

        if not pipeline_result.context:
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
                "context": pipeline_result.context,
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

    # ------------------------------------------------------------------
    # SSE streaming path (called from the streaming route, not execute)
    # ------------------------------------------------------------------

    async def stream(
        self,
        question: str,
        retrieval_k: int = 4,
        conversation_history: Optional[Any] = None,
    ) -> AsyncIterator[str]:
        """
        Async generator that yields SSE-formatted events:
          - ``{type: "token", content: "..."}`` for each token
          - ``{type: "sources", sources: [...], timing: {...}}`` at the end
          - ``[DONE]`` sentinel
        """
        t_start = time.perf_counter()

        # Step 1: Classify intent
        classification = await self._intent_classifier.classify(question)
        logger.debug(
            f"Intent: {classification.intent.value} "
            f"(confidence: {classification.confidence})"
        )

        # Step 2: Route based on intent
        if classification.intent in (Intent.CHITCHAT, Intent.OUT_OF_SCOPE, Intent.VAGUE):
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
        # Retrieval (pass precomputed rewrite data to avoid duplicate LLM call)
        pipeline_result = await self._pipeline.run(
            question=question,
            retrieval_k=retrieval_k,
            precomputed_domain=classification.domain,
            precomputed_keywords=classification.keywords,
            precomputed_articles=classification.likely_articles,
        )
        t_retrieval = time.perf_counter()

        if not pipeline_result.context:
            no_result_msg = (
                "عذراً، لم أتمكن من العثور على مواد قانونية ذات صلة بسؤالك. "
                "يرجى إعادة صياغة السؤال أو التأكد من صحة المصطلحات المستخدمة."
            )
            yield f"data: {json.dumps({'type': 'token', 'content': no_result_msg}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'intent': Intent.LEGAL_QUERY.value}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Assemble prompt
        template = await self._prompt.get_template("LEGAL_CHAT", "egypt", "ar")
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

        # Stream answer
        full_answer: list[str] = []
        t_stream_start = time.perf_counter()
        try:
            async for token in self._llm.astream(assembled.prompt):
                full_answer.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.error("stream_error", error=str(exc))
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        t_done = time.perf_counter()
        complete_answer = "".join(full_answer)

        # Filter cited sources
        cited = self._pipeline.filter_cited_sources(
            complete_answer, pipeline_result.sources
        )
        frontend_sources = cited[: self._max_sources]

        timing = {
            "retrieval_ms": round((t_retrieval - t_start) * 1000, 1),
            "streaming_ms": round((t_done - t_stream_start) * 1000, 1),
            "total_ms": round((t_done - t_start) * 1000, 1),
        }

        yield f"data: {json.dumps({'type': 'sources', 'sources': frontend_sources, 'timing': timing, 'intent': Intent.LEGAL_QUERY.value}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    def _is_social_only_message(self, message: str) -> bool:
        """Legacy fallback pattern matching."""
        text = (message or "").strip().lower()
        if not text:
            return False

        has_legal_cue = any(re.search(pattern, text) for pattern in self._legal_cue_patterns)
        if has_legal_cue:
            return False

        return any(re.match(pattern, text) for pattern in self._social_only_patterns)

    def _build_chitchat_response(self, message: str) -> str:
        """Generate a friendly response for chitchat messages."""
        text = (message or "").strip().lower()
        if re.search(r"شكرا|شكرًا|متشكر|thanks|تسلم", text):
            return "العفو، تحت أمرك في أي وقت. لو حابب نكمل في أي نقطة قانونية أنا معاك."
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
