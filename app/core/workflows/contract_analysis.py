"""
Contract Analysis Workflow

Analyses contracts using RAG-retrieved legal articles and LLM risk assessment.
"""

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


class ContractAnalysisWorkflow(BaseWorkflow):
    """
    Workflow for CONTRACT_ANALYSIS task type.

    Pipeline:
    1. Retrieve relevant legal articles via QueryPipeline
    2. Assemble a contract-analysis prompt
    3. LLM generates risk analysis
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
        return "ContractAnalysisWorkflow"

    async def execute(
        self,
        task_id: str,
        context: Context,
        payload: Dict[str, Any],
        options: Optional[ExecutionOptions] = None,
    ) -> Result:
        start_time = time.time()

        contract_text = payload.get("contract_text", "")
        analysis_type = payload.get("analysis_type", "risk_assessment")

        # 1. Retrieve relevant articles based on contract content
        search_query = f"تحليل عقد: {contract_text[:400]}"
        pipeline_result = await self._pipeline.run(question=search_query, retrieval_k=6)

        # 2. Assemble prompt
        template = await self._prompt.get_template(
            task_type="CONTRACT_ANALYSIS",
            jurisdiction=context.jurisdiction.value,
            language=context.language.value,
        )
        assembled = await self._prompt.assemble_prompt(
            template=template,
            variables={
                "contract_text": contract_text,
                "analysis_type": analysis_type,
                "context": pipeline_result.context or "لا توجد مواد قانونية متاحة",
            },
        )

        # 3. Generate
        resp = await self._llm.generate(LLMRequest(prompt=assembled.prompt))
        answer = resp.content

        # 4. Build response
        response_data = {
            "risks": self._extract_risks(answer),
            "summary": answer,
            "recommendations": self._extract_recommendations(answer),
            "sources": pipeline_result.sources[:7],
        }

        execution_time_ms = int((time.time() - start_time) * 1000)

        return Result.success(
            task_id=UUID(task_id),
            task_type=TaskType.CONTRACT_ANALYSIS,
            data=response_data,
            metadata=ResultMetadata(
                execution_time_ms=execution_time_ms,
                model_used=resp.model,
                tokens_used=resp.tokens_used,
            ),
        )

    @staticmethod
    def _extract_risks(answer: str) -> List[Dict[str, str]]:
        """Best-effort extraction of risk items from the LLM answer."""
        risks: List[Dict[str, str]] = []
        lines = answer.split("\n")
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            risk_level = "medium"
            if "عالي" in stripped or "خطير" in stripped:
                risk_level = "high"
            elif "منخفض" in stripped or "بسيط" in stripped:
                risk_level = "low"

            if stripped.startswith(("-", "•", "١", "٢", "٣", "٤", "٥")) and (
                "خطر" in stripped or "مخاطر" in stripped or "risk" in stripped.lower()
            ):
                risks.append(
                    {
                        "clause": "",
                        "risk_level": risk_level,
                        "description": stripped.lstrip("-•١٢٣٤٥٦٧٨٩٠. "),
                    }
                )
        return risks

    @staticmethod
    def _extract_recommendations(answer: str) -> List[str]:
        """Best-effort extraction of recommendations from the LLM answer."""
        recs: List[str] = []
        in_recs = False
        for line in answer.split("\n"):
            stripped = line.strip()
            if "توصي" in stripped or "التوصيات" in stripped:
                in_recs = True
                continue
            if in_recs and stripped.startswith(("-", "•", "١", "٢", "٣", "٤", "٥")):
                recs.append(stripped.lstrip("-•١٢٣٤٥٦٧٨٩٠. "))
        return recs
