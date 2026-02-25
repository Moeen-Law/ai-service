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
            variables={"question": message, "context": pipeline_result.context},
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
    ) -> AsyncIterator[str]:
        """
        Async generator that yields SSE-formatted events:
          - ``{type: "token", content: "..."}`` for each token
          - ``{type: "sources", sources: [...], timing: {...}}`` at the end
          - ``[DONE]`` sentinel
        """
        t_start = time.perf_counter()

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
            variables={"question": question, "context": pipeline_result.context},
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
