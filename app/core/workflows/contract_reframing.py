"""
Contract Reframing Workflow

Stubbed workflow for reframing contract clauses.
Returns deterministic mock responses for testing.
"""

import time
from typing import Any, Dict, Optional
from uuid import UUID

from app.core.domain.entities import Context, ExecutionOptions, Result, ResultMetadata
from app.core.domain.enums import TaskType
from app.core.workflows.base import BaseWorkflow


class ContractReframingWorkflow(BaseWorkflow):
    """
    Workflow for CONTRACT_REFRAMING task type.

    Reframes contract clauses from different perspectives.
    Currently returns stubbed responses for testing.
    """

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
        """
        Execute the contract reframing workflow.

        In the real implementation, this would:
        1. Analyze the original clause
        2. Call Prompt service with target perspective
        3. Call LLM service to reframe
        4. Validate the reframed version maintains intent
        """
        start_time = time.time()

        clause_text = payload.get("clause_text", "")
        target_perspective = payload.get("target_perspective", "balanced")
        preserve_intent = payload.get("preserve_intent", True)

        response_data = self._generate_stub_reframing(
            clause_text, target_perspective, context
        )

        execution_time_ms = int((time.time() - start_time) * 1000)

        return Result.success(
            task_id=UUID(task_id),
            task_type=TaskType.CONTRACT_REFRAMING,
            data=response_data,
            metadata=ResultMetadata(
                execution_time_ms=execution_time_ms,
                model_used="stub-model",
                tokens_used=len(clause_text.split()) * 3,
            ),
        )

    def _generate_stub_reframing(
        self,
        clause_text: str,
        target_perspective: str,
        context: Context,
    ) -> Dict[str, Any]:
        """Generate a stubbed reframing based on input."""
        perspective_map = {
            "balanced": "متوازن" if context.language.value == "ar" else "balanced",
            "party_a_favorable": (
                "لصالح الطرف الأول"
                if context.language.value == "ar"
                else "favorable to Party A"
            ),
            "party_b_favorable": (
                "لصالح الطرف الثاني"
                if context.language.value == "ar"
                else "favorable to Party B"
            ),
        }

        perspective_text = perspective_map.get(target_perspective, target_perspective)

        if context.language.value == "ar":
            return {
                "reframed_clause": (
                    f"[نص معاد صياغته بمنظور {perspective_text}]\n\n"
                    f"البند الأصلي: {clause_text[:200]}...\n\n"
                    "البند المعدل: يتعهد الطرفان بالالتزام بأحكام هذا البند وفقاً "
                    "للقواعد المتفق عليها، مع مراعاة مصالح جميع الأطراف "
                    "وضمان تحقيق العدالة التعاقدية.\n\n"
                    "[هذا نص تجريبي - سيتم توليد النص الفعلي في النسخة النهائية]"
                ),
                "changes_summary": (
                    f"تمت إعادة صياغة البند بمنظور {perspective_text}. "
                    "التغييرات الرئيسية:\n"
                    "1. توضيح الالتزامات المتبادلة\n"
                    "2. إضافة ضمانات للحقوق\n"
                    "3. تحسين صياغة الشروط\n"
                    "[ملخص تجريبي]"
                ),
            }

        return {
            "reframed_clause": (
                f"[Reframed with {perspective_text} perspective]\n\n"
                f"Original clause: {clause_text[:200]}...\n\n"
                "Modified clause: Both parties hereby undertake to comply with "
                "the provisions of this clause in accordance with the agreed rules, "
                "taking into account the interests of all parties and ensuring "
                "contractual fairness.\n\n"
                "[This is stub text - actual text will be generated in final version]"
            ),
            "changes_summary": (
                f"Clause reframed with {perspective_text} perspective. "
                "Key changes:\n"
                "1. Clarified mutual obligations\n"
                "2. Added rights guarantees\n"
                "3. Improved terms wording\n"
                "[Stub summary]"
            ),
        }
