"""
Stub LLM Service Adapter

Fake implementation of LLM service for testing and development.
Returns canned responses without calling any real LLM.
"""

from typing import Optional

from app.interfaces.ai.llm_service import (
    LLMRequest,
    LLMResponse,
    LLMServiceInterface,
)


class StubLLMService(LLMServiceInterface):
    """
    Stub implementation of LLM service.

    Returns deterministic canned responses for testing.
    Can be easily replaced with real implementation later.
    """

    def __init__(self, model_name: str = "stub-llm-v1") -> None:
        self._model_name = model_name

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a stubbed response."""
        # Generate a simple response based on prompt length
        prompt_length = len(request.prompt)
        response_content = self._generate_stub_content(request.prompt)

        return LLMResponse(
            content=response_content,
            model=self._model_name,
            tokens_used=prompt_length + len(response_content.split()),
            finish_reason="stop",
            metadata={"stub": True},
        )

    async def generate_with_context(
        self,
        prompt: str,
        context: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate a stubbed response with context."""
        combined_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"
        response_content = self._generate_stub_content(combined_prompt)

        return LLMResponse(
            content=response_content,
            model=self._model_name,
            tokens_used=len(combined_prompt.split()) + len(response_content.split()),
            finish_reason="stop",
            metadata={"stub": True, "has_context": True},
        )

    async def health_check(self) -> bool:
        """Always returns True for stub."""
        return True

    def _generate_stub_content(self, prompt: str) -> str:
        """Generate stub content based on prompt."""
        # Check for Arabic content
        if any("\u0600" <= char <= "\u06ff" for char in prompt):
            return (
                "هذه استجابة تجريبية من نموذج اللغة. "
                "في النسخة النهائية، سيتم توليد استجابة حقيقية "
                "بناءً على السياق والسؤال المقدم. "
                "[نص تجريبي - LLM Stub]"
            )

        return (
            "This is a stub response from the language model. "
            "In the final version, a real response will be generated "
            "based on the context and question provided. "
            "[Stub text - LLM Stub]"
        )


# Default instance for dependency injection
stub_llm_service = StubLLMService()
