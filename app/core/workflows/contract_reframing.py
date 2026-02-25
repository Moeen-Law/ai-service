"""
Contract Reframing Workflow

Reframes contract clauses using RAG-retrieved legal articles and LLM generation.
"""

import time
from typing import Any, Dict, Optional
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


class ContractReframingWorkflow(BaseWorkflow):
    """
    Workflow for CONTRACT_REFRAMING task type.

    Pipeline:
    1. Retrieve relevant legal articles via QueryPipeline
    2. Assemble reframing prompt with perspective
    3. LLM generates reframed clause
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
        return "ContractReframingWorkflow"

    async def execute(
        self,
        task_id: str,
        context: Context,
        payload: Dict[str, Any],
        options: Optional[ExecutionOptions] = None,
    ) -> Result:
        start_time = time.time()

        clause_text = payload.get("clause_text", "")
        target_perspective = payload.get("target_perspective", "balanced")

        # 1. Retrieve relevant legal articles
        search_query = f"إعادة صياغة بند عقد: {clause_text[:400]}"
        pipeline_result = await self._pipeline.run(question=search_query, retrieval_k=6)

        # 2. Assemble prompt
        template = await self._prompt.get_template(
            task_type="CONTRACT_REFRAMING",
            jurisdiction=context.jurisdiction.value,
            language=context.language.value,
        )
        assembled = await self._prompt.assemble_prompt(
            template=template,
            variables={
                "clause_text": clause_text,
                "target_perspective": target_perspective,
                "context": pipeline_result.context or "لا توجد مواد قانونية متاحة",
            },
        )

        # 3. Generate
        resp = await self._llm.generate(LLMRequest(prompt=assembled.prompt))
        answer = resp.content

        # 4. Build response
        response_data = {
            "reframed_clause": self._extract_reframed(answer),
            "changes_summary": self._extract_changes_summary(answer),
            "sources": pipeline_result.sources[:7],
        }

        execution_time_ms = int((time.time() - start_time) * 1000)

        return Result.success(
            task_id=UUID(task_id),
            task_type=TaskType.CONTRACT_REFRAMING,
            data=response_data,
            metadata=ResultMetadata(
                execution_time_ms=execution_time_ms,
                model_used=resp.model,
                tokens_used=resp.tokens_used,
            ),
        )

    @staticmethod
    def _extract_reframed(answer: str) -> str:
        """Extract the reframed clause portion from the LLM answer."""
        # Try to find a section labelled as the reframed clause
        markers = [
            "البند المعدل",
            "النص المعدل",
            "الصياغة الجديدة",
            "البند بعد التعديل",
        ]
        for marker in markers:
            idx = answer.find(marker)
            if idx != -1:
                # Return from marker to end of that section (next heading or end)
                section = answer[idx:]
                for end_marker in ["\n\n##", "\nالتغييرات", "\nملخص"]:
                    end_idx = section.find(end_marker)
                    if end_idx != -1:
                        return section[:end_idx].strip()
                return section.strip()
        # Fallback: return full answer
        return answer.strip()

    @staticmethod
    def _extract_changes_summary(answer: str) -> str:
        """Extract the changes summary portion from the LLM answer."""
        markers = ["التغييرات", "ملخص التعديلات", "ما تم تغييره"]
        for marker in markers:
            idx = answer.find(marker)
            if idx != -1:
                return answer[idx:].strip()
        # Fallback: brief description
        return "تم إعادة صياغة البند وفقاً للمواد القانونية ذات الصلة."
