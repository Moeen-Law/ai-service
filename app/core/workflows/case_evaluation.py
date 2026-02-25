"""
Case Evaluation Workflow

Evaluates legal cases using RAG-retrieved legal articles and LLM analysis.
"""

import json
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.core.domain.entities import Context, ExecutionOptions, Result, ResultMetadata
from app.core.domain.enums import TaskType
from app.core.services.query_pipeline import QueryPipeline
from app.core.workflows.base import BaseWorkflow
from app.infrastructure.logging.logger import get_logger
from app.interfaces.ai.llm_service import LLMRequest, LLMServiceInterface
from app.interfaces.ai.prompt_service import PromptServiceInterface
from app.interfaces.ai.rag_service import RAGServiceInterface

logger = get_logger(__name__)


class CaseEvaluationWorkflow(BaseWorkflow):
    """
    Workflow for CASE_EVALUATION task type.

    Pipeline:
    1. Retrieve relevant legal articles via QueryPipeline
    2. Assemble a structured evaluation prompt
    3. LLM generates evaluation with strengths/weaknesses/recommendation
    4. Parse structured output
    """

    def __init__(
        self,
        rag_service: RAGServiceInterface,
        llm_service: LLMServiceInterface,
        prompt_service: PromptServiceInterface,
    ) -> None:
        self._rag = rag_service
        self._llm = llm_service
        self._prompt = prompt_service
        self._pipeline = QueryPipeline(
            rag_service=rag_service,
            llm_service=llm_service,
        )

    @property
    def name(self) -> str:
        return "CaseEvaluationWorkflow"

    async def execute(
        self,
        task_id: str,
        context: Context,
        payload: Dict[str, Any],
        options: Optional[ExecutionOptions] = None,
    ) -> Result:
        start_time = time.time()

        case_description = payload.get("case_description", "")
        case_type = payload.get("case_type")
        evidence_summary = payload.get("evidence_summary")

        # 1. Retrieve relevant articles
        search_query = case_description[:500]
        if case_type:
            search_query = f"{case_type}: {search_query}"
        pipeline_result = await self._pipeline.run(question=search_query, retrieval_k=6)

        # 2. Assemble prompt
        template = await self._prompt.get_template(
            task_type="CASE_EVALUATION",
            jurisdiction=context.jurisdiction.value,
            language=context.language.value,
        )
        variables: Dict[str, Any] = {
            "case_description": case_description,
            "context": pipeline_result.context or "لا توجد مواد قانونية متاحة",
        }
        if evidence_summary:
            variables["case_description"] += f"\n\nملخص الأدلة:\n{evidence_summary}"

        assembled = await self._prompt.assemble_prompt(
            template=template,
            variables=variables,
        )

        # 3. LLM generation
        resp = await self._llm.generate(LLMRequest(prompt=assembled.prompt))
        answer = resp.content

        # 4. Parse into structured output
        response_data = self._parse_evaluation(answer, pipeline_result.sources)

        execution_time_ms = int((time.time() - start_time) * 1000)

        return Result.success(
            task_id=UUID(task_id),
            task_type=TaskType.CASE_EVALUATION,
            data=response_data,
            metadata=ResultMetadata(
                execution_time_ms=execution_time_ms,
                model_used=resp.model,
                tokens_used=resp.tokens_used,
            ),
        )

    @staticmethod
    def _parse_evaluation(answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Best-effort parse of the LLM answer into structured fields."""
        # Try to extract sections by Arabic headers
        strengths: List[str] = []
        weaknesses: List[str] = []
        recommendation = ""

        sections = answer.split("\n")
        current: Optional[str] = None
        for line in sections:
            stripped = line.strip()
            if not stripped:
                continue
            if "نقاط القوة" in stripped or "القوة" in stripped:
                current = "strengths"
                continue
            elif "نقاط الضعف" in stripped or "الضعف" in stripped:
                current = "weaknesses"
                continue
            elif "التوصية" in stripped or "توصية" in stripped:
                current = "recommendation"
                continue

            if current == "strengths" and stripped.startswith(
                ("-", "•", "١", "٢", "٣", "٤", "٥")
            ):
                strengths.append(stripped.lstrip("-•١٢٣٤٥٦٧٨٩٠. "))
            elif current == "weaknesses" and stripped.startswith(
                ("-", "•", "١", "٢", "٣", "٤", "٥")
            ):
                weaknesses.append(stripped.lstrip("-•١٢٣٤٥٦٧٨٩٠. "))
            elif current == "recommendation":
                recommendation += stripped + " "

        return {
            "evaluation": answer,
            "strengths": strengths or ["راجع التقييم أعلاه"],
            "weaknesses": weaknesses or ["راجع التقييم أعلاه"],
            "recommendation": recommendation.strip() or answer[-300:],
            "sources": sources[:7],
        }
