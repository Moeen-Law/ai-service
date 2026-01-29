"""
Legal Chat Workflow

Stubbed workflow for handling conversational legal queries.
Returns deterministic mock responses for testing.
"""

import time
from typing import Any, Dict, Optional
from uuid import UUID

from app.core.domain.entities import Context, ExecutionOptions, Result, ResultMetadata
from app.core.domain.enums import TaskStatus, TaskType
from app.core.workflows.base import BaseWorkflow


class LegalChatWorkflow(BaseWorkflow):
    """
    Workflow for LEGAL_CHAT task type.

    Handles interactive legal assistance queries.
    Currently returns stubbed responses for testing.
    """

    @property
    def name(self) -> str:
        return "LegalChatWorkflow"

    async def execute(
        self,
        task_id: str,
        context: Context,
        payload: Dict[str, Any],
        options: Optional[ExecutionOptions] = None,
    ) -> Result:
        """
        Execute the legal chat workflow.

        In the real implementation, this would:
        1. Call RAG service to retrieve relevant legal documents
        2. Call Prompt service to assemble context + query + history
        3. Call LLM service to generate response
        4. Structure response with sources

        Currently returns a stubbed response.
        """
        start_time = time.time()

        # Extract payload data
        message = payload.get("message", "")
        conversation_history = payload.get("conversation_history", [])

        # Generate stubbed response based on context
        response_data = self._generate_stub_response(message, context)

        execution_time_ms = int((time.time() - start_time) * 1000)

        return Result.success(
            task_id=UUID(task_id),
            task_type=TaskType.LEGAL_CHAT,
            data=response_data,
            metadata=ResultMetadata(
                execution_time_ms=execution_time_ms,
                model_used="stub-model",
                tokens_used=len(message.split()) * 10,  # Simulated token count
            ),
        )

    def _generate_stub_response(
        self,
        message: str,
        context: Context,
    ) -> Dict[str, Any]:
        """Generate a stubbed response based on the input."""
        # Arabic response for Arabic language
        if context.language.value == "ar":
            return {
                "message": (
                    "هذه استجابة تجريبية من نظام المساعدة القانونية. "
                    "في النسخة النهائية، سيتم توفير إجابة قانونية دقيقة "
                    f"بناءً على قوانين {context.jurisdiction.value}. "
                    f"سؤالك: {message[:100]}..."
                ),
                "sources": [
                    "القانون المدني - المادة 89",
                    "قانون العقود - المادة 123",
                ],
            }

        # English response
        return {
            "message": (
                "This is a stubbed response from the legal assistance system. "
                "In the final version, a precise legal answer will be provided "
                f"based on {context.jurisdiction.value} laws. "
                f"Your question: {message[:100]}..."
            ),
            "sources": [
                "Civil Code - Article 89",
                "Contract Law - Article 123",
            ],
        }
