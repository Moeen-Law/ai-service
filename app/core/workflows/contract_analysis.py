"""
Contract Analysis Workflow

Analyses contracts using RAG-retrieved legal articles and LLM risk assessment.
"""

import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.core.domain.entities import Context, ExecutionOptions, Result, ResultMetadata
from app.core.domain.enums import TaskType
from app.core.services.llm_json import ensure_string_list, extract_first_json_object
from app.core.services.query_pipeline import QueryPipeline
from app.core.workflows.base import BaseWorkflow
from app.infrastructure.logging.logger import get_logger
from app.interfaces.ai.llm_service import LLMRequest, LLMServiceInterface
from app.interfaces.ai.prompt_service import PromptServiceInterface
from app.interfaces.ai.rag_service import RAGServiceInterface
from app.interfaces.external.file_service import ExternalFileServiceInterface
from app.core.services.file_text_extractor import FileTextExtractor
from app.shared.errors.exceptions import (
    FileExtractionError,
    FilesServiceError,
    PayloadValidationError,
)

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
        file_service: Optional[ExternalFileServiceInterface] = None,
        file_text_extractor: Optional[FileTextExtractor] = None,
        max_frontend_sources: int = 7,
    ) -> None:
        self._rag = rag_service
        self._llm = llm_service
        self._prompt = prompt_service
        self._pipeline = QueryPipeline(
            rag_service=rag_service,
            llm_service=llm_service,
        )
        self._file_service = file_service
        self._file_text_extractor = file_text_extractor or FileTextExtractor()
        self._max_sources = max_frontend_sources

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

        # Accept either inline contract text or uploaded files (files_ids)
        contract_text = payload.get("contract_text", "")
        files_ids_raw = payload.get("files_ids") or payload.get("file_ids")
        files_ids = self._normalize_file_ids(files_ids_raw)

        # 1. If files provided, build uploaded files context
        uploaded_context = None
        if files_ids:
            if self._file_service is None:
                raise FilesServiceError(
                    message="Files service integration is not configured",
                    details={"field": "file_service"},
                )
            uploaded_context = await self._build_uploaded_files_context(files_ids)

        # 2. Retrieve relevant articles based on contract content or uploaded files
        search_seed = contract_text or (uploaded_context or "")
        search_query = f"تحليل عقد: {str(search_seed)[:400]}"
        pipeline_result = await self._pipeline.run(question=search_query, retrieval_k=6)

        # 2. Assemble prompt
        template = await self._prompt.get_template(
            task_type="CONTRACT_ANALYSIS",
            jurisdiction=context.jurisdiction.value,
            language=context.language.value,
        )
        # Merge RAG context with uploaded files content (if any)
        merged_context = pipeline_result.context or ""
        if uploaded_context:
            merged_context = self._merge_contexts(merged_context, uploaded_context)

        assembled = await self._prompt.assemble_prompt(
            template=template,
            variables={
                "contract_text": contract_text or (uploaded_context or ""),
                "context": merged_context or "لا توجد مواد قانونية متاحة",
            },
        )

        # Use a system prompt tailored to file analysis if available
        try:
            system_prompt = await self._prompt.get_system_prompt(
                task_type="CONTRACT_ANALYSIS",
                jurisdiction=context.jurisdiction.value,
                language=context.language.value,
            )
        except Exception:
            system_prompt = (
                "أنت مساعد قانوني متخصص في تحليل العقود. أعد تحليلاً مفصلاً بالـ Markdown "
                "يغطي تقييم المخاطر، استخراج البنود المهمة، التحقق من الامتثال، واكتشاف العيوب المحتملة."
            )

        # 3. Generate structured output (fallback to heuristic extraction)
        structured_prompt = (
            f"{assembled.prompt}\n\n"
            "أعد النتيجة كـ JSON صالح فقط بدون أي نص إضافي وفق الشكل التالي:\n"
            '{"risks":[{"clause":"...","risk_level":"high|medium|low","description":"..."}],'
            '"summary":"...","recommendations":["..."]}'
        )
        resp = await self._llm.generate(LLMRequest(prompt=structured_prompt))
        answer = resp.content

        include_sources = options.include_sources if options is not None else True
        frontend_sources = (
            pipeline_result.sources[: self._max_sources] if include_sources else []
        )
        parsed = extract_first_json_object(answer)

        # 4. Build response data (keep structured data) and also render a
        # LEGAL_CHAT-style Markdown message so the API can return the same
        # shape as `LegalChatWorkflow` (a `message` string + `sources`).
        if parsed:
            structured = {
                "risks": self._normalize_risks(parsed.get("risks", [])),
                "summary": str(parsed.get("summary", "")).strip() or answer,
                "recommendations": ensure_string_list(parsed.get("recommendations")),
            }
        else:
            structured = {
                "risks": self._extract_risks(answer),
                "summary": answer,
                "recommendations": self._extract_recommendations(answer),
            }

        response_data = {**structured, "sources": frontend_sources}

        # Render Markdown message matching LEGAL_CHAT final output style.
        message = self._render_markdown_analysis(structured)

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Return in the same shape as LegalChat: a `message` string plus
        # `sources` and an `intent` value so clients get a consistent format.
        final_output = {
            "message": message,
            "sources": frontend_sources,
            "intent": TaskType.CONTRACT_ANALYSIS.value,
        }

        # Also keep the structured data for downstream consumers inside
        # a `document` key so nothing is lost.
        final_output["document"] = response_data

        return Result.success(
            task_id=UUID(task_id),
            task_type=TaskType.CONTRACT_ANALYSIS,
            data=final_output,
            metadata=ResultMetadata(
                execution_time_ms=execution_time_ms,
                model_used=resp.model,
                tokens_used=resp.tokens_used,
            ),
        )

    async def _build_uploaded_files_context(self, files_ids: List[str]) -> str:
        """Fetch uploaded files by IDs and convert them into prompt-ready text."""
        if not files_ids:
            return ""

        try:
            files = await self._file_service.fetch_files(files_ids)
            extracted_files = self._file_text_extractor.extract_many(files)
        except FileExtractionError as exc:
            raise PayloadValidationError(
                message="Unable to process one or more uploaded files",
                task_type="CONTRACT_ANALYSIS",
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

    @staticmethod
    def _normalize_risks(value: Any) -> List[Dict[str, str]]:
        """Normalize model-produced risk list into API-compatible risk items."""
        if not isinstance(value, list):
            return []

        normalized: List[Dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                continue

            description = str(item.get("description", "")).strip()
            if not description:
                continue

            normalized.append(
                {
                    "clause": str(item.get("clause", "")).strip(),
                    "risk_level": ContractAnalysisWorkflow._normalize_risk_level(
                        str(item.get("risk_level", "medium"))
                    ),
                    "description": description,
                }
            )

        return normalized

    @staticmethod
    def _normalize_risk_level(value: str) -> str:
        """Map Arabic/English risk levels to high|medium|low."""
        mapping = {
            "high": "high",
            "medium": "medium",
            "low": "low",
            "عالي": "high",
            "مرتفع": "high",
            "متوسط": "medium",
            "منخفض": "low",
            "بسيط": "low",
        }
        return mapping.get(value.strip().lower(), "medium")

    @staticmethod
    def _render_markdown_analysis(structured: dict) -> str:
        """Render a Markdown summary similar to LEGAL_CHAT's message output.

        `structured` is expected to contain `summary`, `risks` (list), and
        `recommendations` (list).
        """
        parts: List[str] = []
        summary = structured.get("summary", "").strip()
        parts.append("## تحليل العقد — ملخّص وتنبيهات\n")
        parts.append("**الملخّص:**  \\n+" + (summary or "لا توجد ملخص متاح") + "\n")
        parts.append("---\n\n")

        risks = structured.get("risks") or []
        if risks:
            parts.append("### المخاطر الرئيسية\n")
            for idx, r in enumerate(risks, start=1):
                clause = r.get("clause", "").strip()
                level = r.get("risk_level", "medium")
                desc = r.get("description", "").strip()
                parts.append(
                    f"{idx}. **{clause or 'بند غير محدد'} — مستوى الخطر: {level}**\n"
                )
                parts.append(f"   {desc}\n")
            parts.append("\n")

        recs = structured.get("recommendations") or []
        if recs:
            parts.append("### التوصيات العملية\n")
            for r in recs:
                parts.append(f"- {r}\n")
            parts.append("\n")

        parts.append("### المصادر المقتبسة\n")
        parts.append("- راجع المراجع القانونية المرفقة في الحقول المهيكلة.")

        return "\n".join(parts)
