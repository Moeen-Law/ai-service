"""
Legal Chat Workflow

Full-pipeline workflow for conversational legal queries.
Uses hybrid RAG retrieval, LLM query-rewriting, and SSE streaming.
"""

import json
import re
import time
from typing import Any, AsyncIterator, Dict, Optional
from uuid import UUID

from app.core.domain.entities import Context, ExecutionOptions, Result, ResultMetadata
from app.core.domain.enums import TaskStatus, TaskType
from app.core.services.query_pipeline import QueryPipeline
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
    1. QueryPipeline.run() — rewrite → hybrid retrieve → inject → rerank → filter
    2. Assemble prompt with مُعين system instructions
    3. LLM generate (non-streaming path) OR stream (streaming path)
    4. Filter cited sources
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
        self._pipeline = QueryPipeline(
            rag_service=rag_service,
            llm_service=llm_service,
            max_frontend_sources=max_frontend_sources,
        )
        self._max_sources = max_frontend_sources
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

        if self._is_social_only_message(message):
            return Result.success(
                task_id=UUID(task_id),
                task_type=TaskType.LEGAL_CHAT,
                data={
                    "message": self._build_social_response(message),
                    "sources": [],
                },
                metadata=ResultMetadata(
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    model_used="rule-based",
                ),
            )

        # 1. Retrieval pipeline
        pipeline_result = await self._pipeline.run(
            question=message,
            retrieval_k=retrieval_k,
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

        if self._is_social_only_message(question):
            quick_reply = self._build_social_response(question)
            yield f"data: {json.dumps({'type': 'token', 'content': quick_reply}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': []}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Retrieval
        pipeline_result = await self._pipeline.run(
            question=question,
            retrieval_k=retrieval_k,
        )
        t_retrieval = time.perf_counter()

        if not pipeline_result.context:
            no_result_msg = (
                "عذراً، لم أتمكن من العثور على مواد قانونية ذات صلة بسؤالك. "
                "يرجى إعادة صياغة السؤال أو التأكد من صحة المصطلحات المستخدمة."
            )
            yield f"data: {json.dumps({'type': 'token', 'content': no_result_msg}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': []}, ensure_ascii=False)}\n\n"
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

        yield f"data: {json.dumps({'type': 'sources', 'sources': frontend_sources, 'timing': timing}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    def _is_social_only_message(self, message: str) -> bool:
        text = (message or "").strip().lower()
        if not text:
            return False

        has_legal_cue = any(re.search(pattern, text) for pattern in self._legal_cue_patterns)
        if has_legal_cue:
            return False

        return any(re.match(pattern, text) for pattern in self._social_only_patterns)

    def _build_social_response(self, message: str) -> str:
        text = (message or "").strip().lower()
        if re.search(r"شكرا|شكرًا|متشكر|thanks|تسلم", text):
            return "العفو، تحت أمرك في أي وقت. لو حابب نكمل في أي نقطة قانونية أنا معاك."
        if re.search(r"اهلا|أهلا", text):
            return "أهلا بيك، منور. احكي لي سؤالك القانوني وأنا أساعدك خطوة بخطوة."
        if re.search(r"باي|مع السلامة|سلام", text):
            return "مع السلامة، وفي أي وقت تحتاج استشارة قانونية أنا موجود."
        return "تمام، أنا معاك. ابعت سؤالك القانوني أو التفاصيل اللي تحب نكمل عليها."

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
