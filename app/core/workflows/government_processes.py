"""
Government Processes Workflow

Workflow for searching and extracting structured information about
Egyptian government procedures using web search and LLM extraction.
"""

import json
import time
from typing import Any, Dict, Optional
from uuid import UUID


from app.core.domain.entities import Context, ExecutionOptions, Result, ResultMetadata
from app.core.domain.enums import TaskType
from app.core.services.llm_json import extract_first_json_object
from app.core.workflows.base import BaseWorkflow
from app.infrastructure.logging.logger import get_logger
from app.interfaces.ai.llm_service import LLMRequest, LLMServiceInterface
from app.interfaces.external.search_service import SearchServiceInterface


def _stringify_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    if message:
        return message
    repr_value = repr(exc).strip()
    if repr_value:
        return repr_value
    return f"{type(exc).__name__} with empty message"


logger = get_logger(__name__)


class GovernmentProcessesWorkflow(BaseWorkflow):
    """
    Workflow for the GOVERNMENT_PROCESSES task type.

    Pipeline:
    1. Build optimized Arabic search query
    2. Perform web search using TavilySearchResults
    3. Extract search results
    4. Pass to Gemini with strict JSON extraction prompt
    5. Parse and validate JSON response
    6. Return structured data with sources
    """

    def __init__(
        self,
        search_service: SearchServiceInterface,
        llm_service: LLMServiceInterface,
    ) -> None:
        self._search = search_service
        self._llm = llm_service

    @property
    def name(self) -> str:
        return "government_processes_workflow"

    async def execute(
        self,
        task_id: str,
        context: Context,
        payload: Dict[str, Any],
        options: Optional[ExecutionOptions] = None,
    ) -> Result:
        start_time = time.monotonic()
        query = payload.get("query", "").strip()

        if not query:
            return Result.failure(
                task_id=UUID(task_id) if isinstance(task_id, str) else task_id,
                task_type=TaskType.GOVERNMENT_PROCESSES,
                errors=["Query cannot be empty"],
                metadata=ResultMetadata(
                    execution_time_ms=0,
                    model_used="unknown",
                    tokens_used=0,
                ),
            )

        try:
            logger.info(
                "government_processes_started",
                query=query[:100],
                task_id=task_id,
            )

            # 1. Build optimized search query
            search_query = self._build_search_query(query)
            logger.debug("search_query_built", search_query=search_query)

            # 2. Perform web search
            search_response = await self._search.search(
                query=search_query,
                max_results=8,
                timeout_seconds=10,
            )

            if not search_response.results:
                logger.warning(
                    "government_processes_no_results",
                    query=query,
                    search_query=search_query,
                )
                return self._build_fallback_response(
                    task_id=task_id,
                    query=query,
                    start_time=start_time,
                    reason="no_search_results",
                )

            logger.info(
                "search_results_retrieved",
                results_count=len(search_response.results),
            )

            # 3. Format search results for LLM
            formatted_results = self._format_search_results(search_response.results)

            # 4. Extract structured data using LLM
            extraction_prompt = self._build_extraction_prompt(query, formatted_results)
            system_prompt = (
                "You are a legal assistant specializing in Egyptian government procedures. "
                "Always respond with ONLY valid JSON, no markdown formatting or extra text. "
                "Ensure strings in JSON are enclosed in double quotes."
            )

            llm_req = LLMRequest(
                prompt=extraction_prompt,
                system_prompt=system_prompt,
                max_tokens=options.max_tokens if options else 1000,
                temperature=0.2,
            )

            llm_resp = await self._llm.generate(llm_req)
            raw_response = llm_resp.content
            tokens_used = llm_resp.tokens_used
            model_used = llm_resp.model

            logger.debug("llm_response_received", response_length=len(raw_response))

            # 5. Parse and validate JSON
            parsed_data = self._parse_json_response(raw_response)

            if not parsed_data:
                logger.warning(
                    "json_parsing_failed",
                    raw_response=raw_response[:200],
                )
                return self._build_fallback_response(
                    task_id=task_id,
                    query=query,
                    start_time=start_time,
                    reason="json_parsing_failed",
                )

            # 6. Enrich with sources
            sources = [
                {"title": result.title, "url": result.url}
                for result in search_response.results[:5]
            ]
            parsed_data["sources"] = sources

            execution_time_ms = int((time.monotonic() - start_time) * 1000)
            metadata = ResultMetadata(
                execution_time_ms=execution_time_ms,
                model_used=model_used,
                tokens_used=tokens_used,
            )

            logger.info(
                "government_processes_success",
                query=query,
                execution_time_ms=execution_time_ms,
            )

            return Result.success(
                task_id=UUID(task_id) if isinstance(task_id, str) else task_id,
                task_type=TaskType.GOVERNMENT_PROCESSES,
                data=parsed_data,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(
                "government_processes_failed",
                error=_stringify_exception(e),
                error_type=type(e).__name__,
            )
            execution_time_ms = int((time.monotonic() - start_time) * 1000)
            return Result.failure(
                task_id=UUID(task_id) if isinstance(task_id, str) else task_id,
                task_type=TaskType.GOVERNMENT_PROCESSES,
                errors=[
                    f"Failed to process government procedures: {_stringify_exception(e)}"
                ],
                metadata=ResultMetadata(
                    execution_time_ms=execution_time_ms,
                    model_used="unknown",
                    tokens_used=0,
                ),
            )

    @staticmethod
    def _build_search_query(user_query: str) -> str:
        """
        Build optimized Arabic search query for Egyptian government procedures.

        Args:
            user_query: User's original query

        Returns:
            Optimized search query
        """
        return f"إجراءات {user_query} في مصر 2026 الأوراق والرسوم"

    @staticmethod
    def _format_search_results(results: list) -> str:
        """
        Format search results into a readable string for LLM.

        Args:
            results: List of SearchResult objects

        Returns:
            Formatted string of search results
        """
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"{i}. العنوان: {result.title}\n"
                f"   الرابط: {result.url}\n"
                f"   الملخص: {result.snippet}\n"
            )
        return "\n".join(formatted)

    @staticmethod
    def _build_extraction_prompt(user_query: str, search_results: str) -> str:
        """
        Build the extraction prompt for LLM.

        Args:
            user_query: User's original query
            search_results: Formatted search results

        Returns:
            Extraction prompt
        """
        return f"""
استخرج المعلومات المهمة عن إجراءات: "{user_query}"

نتائج البحث:
{search_results}

استخرج المعلومات التالية وأرجعها كـ JSON فقط بدون أي نص إضافي:

{{
    "summary": "ملخص شامل للإجراء في 2-3 جمل",
    "structured_data": {{
        "required_docs": ["المستند الأول المطلوب", "المستند الثاني"],
        "estimated_fees": "تقدير الرسوم أو 'مجاني' إن لم تكن هناك رسوم",
        "authority": "الجهة الحكومية المسؤولة",
        "steps": ["الخطوة الأولى", "الخطوة الثانية", "الخطوة الثالثة"]
    }}
}}

تأكد من أن الرد هو JSON صحيح فقط بدون markdown أو نصوص إضافية.
"""

    @staticmethod
    def _parse_json_response(response: str) -> Optional[Dict[str, Any]]:
        """
        Parse and validate JSON response from LLM.

        Args:
            response: Raw LLM response

        Returns:
            Parsed JSON dict or None if parsing fails
        """
        try:
            parsed = extract_first_json_object(response)

            if not parsed:
                logger.warning(
                    "json_decode_error",
                    response=response[:200],
                )
                return None

            required_keys = {"summary", "structured_data"}
            if not all(key in parsed for key in required_keys):
                logger.warning(
                    "json_missing_required_keys",
                    provided_keys=list(parsed.keys()),
                    required_keys=list(required_keys),
                )
                return None

            structured = parsed.get("structured_data", {})
            required_struct_keys = {
                "required_docs",
                "estimated_fees",
                "authority",
                "steps",
            }
            if not all(key in structured for key in required_struct_keys):
                logger.warning(
                    "json_missing_structured_keys",
                    provided_keys=list(structured.keys()),
                    required_keys=list(required_struct_keys),
                )
                return None

            return parsed

        except json.JSONDecodeError as e:
            logger.warning("json_decode_error", error=str(e), response=response[:200])
            return None
        except Exception as e:
            logger.warning("json_parsing_exception", error=str(e))
            return None

    @staticmethod
    def _build_fallback_response(
        task_id: str,
        query: str,
        start_time: float,
        reason: str,
    ) -> Result:
        """
        Build a graceful fallback response when extraction fails.

        Args:
            task_id: Task ID
            query: Original query
            start_time: Start time for execution timing
            reason: Reason for fallback

        Returns:
            Result with fallback data
        """
        execution_time_ms = int((time.monotonic() - start_time) * 1000)

        fallback_data = {
            "summary": f"لم نتمكن من استخراج معلومات مفصلة عن '{query}'. يرجى المحاولة مرة أخرى أو تحسين صيغة السؤال.",
            "structured_data": {
                "required_docs": ["يرجى البحث مباشرة على موقع الجهة الحكومية"],
                "estimated_fees": "غير متوفر",
                "authority": "جهة حكومية مصرية",
                "steps": ["تواصل مع الجهة الحكومية المختصة للحصول على معلومات دقيقة"],
            },
            "sources": [],
            "fallback": True,
            "fallback_reason": reason,
        }

        logger.info(
            "fallback_response_returned",
            reason=reason,
            query=query,
        )

        return Result.success(
            task_id=UUID(task_id) if isinstance(task_id, str) else task_id,
            task_type=TaskType.GOVERNMENT_PROCESSES,
            data=fallback_data,
            metadata=ResultMetadata(
                execution_time_ms=execution_time_ms,
                model_used="unknown",
                tokens_used=0,
            ),
        )
