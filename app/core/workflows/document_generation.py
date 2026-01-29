"""
Document Generation Workflow

Stubbed workflow for generating legal documents.
Returns deterministic mock responses for testing.
"""

import time
from typing import Any, Dict, Optional
from uuid import UUID

from app.core.domain.entities import Context, ExecutionOptions, Result, ResultMetadata
from app.core.domain.enums import TaskType
from app.core.workflows.base import BaseWorkflow


class DocumentGenerationWorkflow(BaseWorkflow):
    """
    Workflow for DOCUMENT_GENERATION task type.

    Generates legal documents from templates and parameters.
    Currently returns stubbed responses for testing.
    """

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
        """
        Execute the document generation workflow.

        In the real implementation, this would:
        1. Call RAG service to retrieve document template
        2. Call Prompt service to assemble generation instructions
        3. Call LLM service to populate template
        4. Format and return generated document
        """
        start_time = time.time()

        document_type = payload.get("document_type", "document")
        parameters = payload.get("parameters", {})

        response_data = self._generate_stub_document(document_type, parameters, context)

        execution_time_ms = int((time.time() - start_time) * 1000)

        return Result.success(
            task_id=UUID(task_id),
            task_type=TaskType.DOCUMENT_GENERATION,
            data=response_data,
            metadata=ResultMetadata(
                execution_time_ms=execution_time_ms,
                model_used="stub-model",
                tokens_used=500,
            ),
        )

    def _generate_stub_document(
        self,
        document_type: str,
        parameters: Dict[str, Any],
        context: Context,
    ) -> Dict[str, Any]:
        """Generate a stubbed document based on input."""
        party_a = parameters.get("party_a", "الطرف الأول")
        party_b = parameters.get("party_b", "الطرف الثاني")
        subject = parameters.get("subject", "موضوع العقد")

        if context.language.value == "ar":
            content = f"""# {document_type.upper()}

## عقد {subject}

**الأطراف:**
- الطرف الأول: {party_a}
- الطرف الثاني: {party_b}

**البنود:**

### البند الأول: موضوع العقد
يتفق الطرفان على {subject} وفقاً للشروط المحددة أدناه.

### البند الثاني: مدة العقد
يسري هذا العقد من تاريخ التوقيع ولمدة عام كامل.

### البند الثالث: الالتزامات
[محتوى تجريبي - سيتم استبداله في النسخة النهائية]

---
*هذا مستند تجريبي. سيتم توليد المحتوى الفعلي في النسخة النهائية.*
"""
        else:
            content = f"""# {document_type.upper()}

## {subject} Agreement

**Parties:**
- First Party: {party_a}
- Second Party: {party_b}

**Terms:**

### Article 1: Subject Matter
Both parties agree to {subject} according to the terms specified below.

### Article 2: Duration
This contract is effective from the signing date for a period of one year.

### Article 3: Obligations
[Stub content - will be replaced in final version]

---
*This is a stub document. Actual content will be generated in the final version.*
"""

        return {
            "document_content": content,
            "format": "markdown",
        }
