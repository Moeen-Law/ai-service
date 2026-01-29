"""
Case Evaluation Workflow

Stubbed workflow for evaluating legal cases.
Returns deterministic mock responses for testing.
"""

import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.core.domain.entities import Context, ExecutionOptions, Result, ResultMetadata
from app.core.domain.enums import TaskType
from app.core.workflows.base import BaseWorkflow


class CaseEvaluationWorkflow(BaseWorkflow):
    """
    Workflow for CASE_EVALUATION task type.

    Evaluates legal cases and provides strategic recommendations.
    Currently returns stubbed responses for testing.
    """

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
        """
        Execute the case evaluation workflow.

        In the real implementation, this would:
        1. Parse case description and evidence
        2. Call RAG service for relevant precedents
        3. Call LLM service for analysis
        4. Structure evaluation with strengths/weaknesses
        """
        start_time = time.time()

        case_description = payload.get("case_description", "")
        case_type = payload.get("case_type")
        evidence_summary = payload.get("evidence_summary")

        response_data = self._generate_stub_evaluation(
            case_description, case_type, evidence_summary, context
        )

        execution_time_ms = int((time.time() - start_time) * 1000)

        return Result.success(
            task_id=UUID(task_id),
            task_type=TaskType.CASE_EVALUATION,
            data=response_data,
            metadata=ResultMetadata(
                execution_time_ms=execution_time_ms,
                model_used="stub-model",
                tokens_used=len(case_description.split()) * 2,
            ),
        )

    def _generate_stub_evaluation(
        self,
        case_description: str,
        case_type: Optional[str],
        evidence_summary: Optional[str],
        context: Context,
    ) -> Dict[str, Any]:
        """Generate a stubbed evaluation based on input."""
        domain = context.domain.value if context.domain else "general"

        if context.language.value == "ar":
            return {
                "evaluation": (
                    f"تقييم أولي للقضية في مجال {domain}:\n\n"
                    f"بناءً على وصف القضية المقدم، يتضح أن هذه قضية "
                    f"تتطلب دراسة معمقة للوقائع والأدلة المتاحة. "
                    f"المختصر: {case_description[:150]}...\n\n"
                    "تحليل الموقف القانوني:\n"
                    "يتوقف نجاح الدعوى على عدة عوامل رئيسية تتضمن "
                    "قوة الأدلة المقدمة وتطبيق القانون على الوقائع.\n\n"
                    "[هذا تقييم تجريبي]"
                ),
                "strengths": [
                    "وضوح الوقائع المعروضة",
                    "توفر أساس قانوني واضح للدعوى",
                    "الأدلة الأولية تدعم الموقف",
                ],
                "weaknesses": [
                    "قد تحتاج الأدلة إلى تعزيز إضافي",
                    "احتمال وجود دفوع مضادة قوية",
                    "التقادم قد يكون عاملاً مؤثراً",
                ],
                "recommendation": (
                    "ننصح بجمع المزيد من الأدلة الداعمة قبل المضي "
                    "في الإجراءات القانونية. كما ننصح بالتشاور مع "
                    "محامٍ متخصص في هذا المجال للحصول على تقييم شامل. "
                    "[توصية تجريبية]"
                ),
            }

        return {
            "evaluation": (
                f"Preliminary case evaluation in {domain} domain:\n\n"
                f"Based on the case description provided, this case "
                f"requires in-depth study of the facts and available evidence. "
                f"Summary: {case_description[:150]}...\n\n"
                "Legal position analysis:\n"
                "The success of the claim depends on several key factors including "
                "the strength of the evidence presented and application of law to facts.\n\n"
                "[This is a stub evaluation]"
            ),
            "strengths": [
                "Clear presentation of facts",
                "Valid legal basis for the claim",
                "Preliminary evidence supports the position",
            ],
            "weaknesses": [
                "Evidence may need additional strengthening",
                "Possibility of strong counter-defenses",
                "Statute of limitations may be a factor",
            ],
            "recommendation": (
                "We recommend gathering more supporting evidence before proceeding "
                "with legal action. We also advise consulting with a specialist "
                "attorney in this field for a comprehensive evaluation. "
                "[Stub recommendation]"
            ),
        }
