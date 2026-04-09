"""
Document Generation Workflow

Generates legal documents using RAG-retrieved templates/articles and LLM.
"""

import time
from typing import Any, Dict, Optional
from uuid import UUID

from app.core.domain.entities import Context, ExecutionOptions, Result, ResultMetadata
from app.core.domain.enums import TaskType
from app.core.services.llm_json import extract_first_json_object
from app.core.services.query_pipeline import QueryPipeline
from app.core.workflows.base import BaseWorkflow
from app.infrastructure.logging.logger import get_logger
from app.interfaces.ai.llm_service import LLMRequest, LLMServiceInterface
from app.interfaces.ai.prompt_service import PromptServiceInterface
from app.interfaces.ai.rag_service import RAGServiceInterface

logger = get_logger(__name__)


class DocumentGenerationWorkflow(BaseWorkflow):
    """
    Workflow for DOCUMENT_GENERATION task type.

    Pipeline:
    1. Retrieve relevant legal articles via QueryPipeline
    2. Assemble document-generation prompt
    3. LLM generates the document
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
        return "DocumentGenerationWorkflow"

    async def execute(
        self,
        task_id: str,
        context: Context,
        payload: Dict[str, Any],
        options: Optional[ExecutionOptions] = None,
    ) -> Result:
        start_time = time.time()

        document_type = payload.get("document_type", "document")
        parameters = payload.get("parameters", {})

        # 1. Retrieve relevant articles for the document type
        search_query = f"إنشاء مستند قانوني: {document_type}"
        if parameters.get("subject"):
            search_query += f" - {parameters['subject']}"
        pipeline_result = await self._pipeline.run(question=search_query, retrieval_k=6)

        # 2. Assemble prompt
        template = await self._prompt.get_template(
            task_type="DOCUMENT_GENERATION",
            jurisdiction=context.jurisdiction.value,
            language=context.language.value,
        )
        assembled = await self._prompt.assemble_prompt(
            template=template,
            variables={
                "document_type": document_type,
                "parameters": str(parameters),
                "context": pipeline_result.context or "لا توجد مواد قانونية متاحة",
            },
        )

        # 3. Generate structured output (fallback to plain text when parsing fails)
        structured_prompt = (
            f"{assembled.prompt}\n\n"
            "أعد النتيجة كـ JSON صالح فقط بدون أي نص إضافي وفق هذا الشكل:\n"
            '{"document_content":"...","format":"markdown"}'
        )
        resp = await self._llm.generate(LLMRequest(prompt=structured_prompt))
        parsed = extract_first_json_object(resp.content)

        content = str(resp.content).strip()
        document_format = "markdown"
        if parsed:
            content = str(parsed.get("document_content", "")).strip() or content
            document_format = (
                str(parsed.get("format", "markdown")).strip() or "markdown"
            )

        include_sources = options.include_sources if options is not None else True
        frontend_sources = pipeline_result.sources[:7] if include_sources else []

        # 4. Build response
        response_data = {
            "document_content": content,
            "format": document_format,
            "sources": frontend_sources,
        }

        execution_time_ms = int((time.time() - start_time) * 1000)

        return Result.success(
            task_id=UUID(task_id),
            task_type=TaskType.DOCUMENT_GENERATION,
            data=response_data,
            metadata=ResultMetadata(
                execution_time_ms=execution_time_ms,
                model_used=resp.model,
                tokens_used=resp.tokens_used,
            ),
        )
