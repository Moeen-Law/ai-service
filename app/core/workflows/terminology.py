"""
Terminology Workflow

Workflow for explaining legal terminology with examples, utilizing LLM and optional RAG context.
"""

import json
import time
from typing import Any, Dict, Optional
from uuid import UUID

from app.core.domain.entities import Context, ExecutionOptions, Result, ResultMetadata
from app.core.domain.enums import TaskType
from app.core.workflows.base import BaseWorkflow
from app.infrastructure.logging.logger import get_logger
from app.interfaces.ai.llm_service import LLMRequest, LLMServiceInterface
from app.interfaces.ai.rag_service import RAGQuery, RAGServiceInterface
from app.shared.utils.exceptions import _stringify_exception

logger = get_logger(__name__)

class TerminologyWorkflow(BaseWorkflow):
    """
    Workflow for the LEGAL_TERMINOLOGY task type.
    """

    def __init__(
        self,
        llm_service: LLMServiceInterface,
        rag_service: RAGServiceInterface,
    ) -> None:
        self._llm_service = llm_service
        self._rag_service = rag_service

    @property
    def name(self) -> str:
        return "terminology_explanation_workflow"

    async def execute(
        self,
        task_id: str,
        context: Context,
        payload: Dict[str, Any],
        options: Optional[ExecutionOptions] = None,
    ) -> Result:
        start_time = time.monotonic()
        term = payload.get("terminology", "")
        use_rag = payload.get("use_rag", True)
        
        sources = []
        retrieved_text = ""
        rag_used = False
        raw_response = None
        
        try:
            # 1. Option to use RAG context
            if use_rag:
                rag_query = RAGQuery(
                    query=term,
                    jurisdiction=context.jurisdiction.value,
                    language=context.language.value,
                    top_k=3,
                    threshold=0.6,
                    domain=context.domain.value if context.domain else None
                )
                
                try:
                    rag_response = await self._rag_service.retrieve(rag_query)
                    if rag_response and rag_response.documents:
                        rag_used = True
                        for doc in rag_response.documents:
                            sources.append({
                                "id": doc.id,
                                "title": doc.source,
                                "excerpt": doc.content[:300],  # truncated excerpt
                                "score": doc.score
                            })
                            retrieved_text += f"\n- {doc.content}"
                except Exception as e:
                    logger.warning(f"RAG retrieval failed for terminology '{term}': {{str(e)}}")

            # 2. Prepare Prompt
            rag_instruction = f"Informational context (if relevant):\n{{retrieved_text}}" if rag_used else ""
            
            prompt = f"""
أنت مساعد قانوني مصري. اشرح المصطلح القانوني التالي بوضوح وإيجاز.
المصطلح: "{term}"

المطلوب إرجاعه هو مقطع JSON فقط بالصيغة التالية بدون أي تنسيقات مثل markdown:
{{
    "brief_explanation": "شرح مبسط للمصطلح في جملتين حد أقصى",
    "examples": ["مثال أول من عقود أو قضايا", "مثال ثاني"]
}}

{rag_instruction}
"""
            system_prompt = "You are a legal assistant. Always format your output strictly as a valid JSON object without markdown formatting blocks."
            
            # 3. LLM Call
            llm_req = LLMRequest(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=options.max_tokens if options else 500,
                temperature=options.temperature if options else 0.3
            )
            
            llm_resp = await self._llm_service.generate(llm_req)
            raw_response = llm_resp.content
            tokens_used = llm_resp.tokens_used
            model_used = llm_resp.model
            
            # 4. Parse output
            clean_json = raw_response.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json.replace("```json", "", 1)
            if clean_json.startswith("```"):
                clean_json = clean_json.replace("```", "", 1)
            if clean_json.endswith("```"):
                clean_json = clean_json[::-1].replace("```", "", 1)[::-1]
            clean_json = clean_json.strip()

            parsed = json.loads(clean_json)
            
            brief_explanation = parsed.get("brief_explanation", "لم نتمكن من صياغة الشرح.")
            examples = parsed.get("examples", [])
            
            execution_time_ms = int((time.monotonic() - start_time) * 1000)
            metadata = ResultMetadata(
                execution_time_ms=execution_time_ms,
                model_used=model_used,
                tokens_used=tokens_used
            )
            
            data = {
                "term": term,
                "brief_explanation": brief_explanation,
                "examples": examples,
                "sources": sources,
                "rag_used": rag_used,
                "raw_response": raw_response
            }
            
            return Result.success(
                task_id=UUID(task_id) if isinstance(task_id, str) else task_id,
                task_type=TaskType.LEGAL_TERMINOLOGY,
                data=data,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error("terminology_workflow_failed", error=_stringify_exception(e))
            execution_time_ms = int((time.monotonic() - start_time) * 1000)
            metadata = ResultMetadata(
                execution_time_ms=execution_time_ms,
                model_used="unknown",
                tokens_used=0
            )
            return Result.failure(
                task_id=UUID(task_id) if isinstance(task_id, str) else task_id,
                task_type=TaskType.LEGAL_TERMINOLOGY,
                errors=["فشل استخراج شرح المصطلح: " + _stringify_exception(e)],
                metadata=metadata
            )
