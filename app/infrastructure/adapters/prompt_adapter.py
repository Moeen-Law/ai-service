"""
Stub Prompt Service Adapter

Fake implementation of Prompt service for testing and development.
Returns canned prompt templates without calling any real prompt store.
"""

from typing import Any, Dict, List, Optional

from app.interfaces.ai.prompt_service import (
    AssembledPrompt,
    PromptServiceInterface,
    PromptTemplate,
)


class StubPromptService(PromptServiceInterface):
    """
    Stub implementation of Prompt service.

    Returns deterministic canned prompt templates for testing.
    Can be easily replaced with real implementation later.
    """

    # Sample templates for different task types
    _TEMPLATES: Dict[str, PromptTemplate] = {
        "LEGAL_CHAT": PromptTemplate(
            id="legal-chat-v1",
            name="Legal Chat Template",
            template=(
                "You are a legal assistant specializing in {jurisdiction} law.\n"
                "Language: {language}\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Provide a helpful legal response."
            ),
            variables=["jurisdiction", "language", "context", "question"],
            description="Template for legal chat interactions",
        ),
        "DOCUMENT_GENERATION": PromptTemplate(
            id="doc-gen-v1",
            name="Document Generation Template",
            template=(
                "Generate a {document_type} document for {jurisdiction}.\n"
                "Language: {language}\n\n"
                "Parameters:\n{parameters}\n\n"
                "Generate the document content."
            ),
            variables=["document_type", "jurisdiction", "language", "parameters"],
            description="Template for document generation",
        ),
        "CONTRACT_ANALYSIS": PromptTemplate(
            id="contract-analysis-v1",
            name="Contract Analysis Template",
            template=(
                "Analyze the following contract under {jurisdiction} law.\n"
                "Analysis type: {analysis_type}\n"
                "Language: {language}\n\n"
                "Contract:\n{contract_text}\n\n"
                "Provide detailed analysis."
            ),
            variables=["jurisdiction", "analysis_type", "language", "contract_text"],
            description="Template for contract analysis",
        ),
        "CONTRACT_REFRAMING": PromptTemplate(
            id="contract-reframe-v1",
            name="Contract Reframing Template",
            template=(
                "Reframe the following clause with {perspective} perspective.\n"
                "Jurisdiction: {jurisdiction}\n"
                "Language: {language}\n\n"
                "Original clause:\n{clause_text}\n\n"
                "Provide reframed version."
            ),
            variables=["perspective", "jurisdiction", "language", "clause_text"],
            description="Template for contract reframing",
        ),
        "CASE_EVALUATION": PromptTemplate(
            id="case-eval-v1",
            name="Case Evaluation Template",
            template=(
                "Evaluate the following case under {jurisdiction} law.\n"
                "Domain: {domain}\n"
                "Language: {language}\n\n"
                "Case description:\n{case_description}\n\n"
                "Provide evaluation with strengths and weaknesses."
            ),
            variables=["jurisdiction", "domain", "language", "case_description"],
            description="Template for case evaluation",
        ),
    }

    _SYSTEM_PROMPTS: Dict[str, str] = {
        "ar": (
            "أنت مساعد قانوني متخصص. قدم معلومات دقيقة ومفيدة "
            "مع الإشارة إلى المصادر القانونية ذات الصلة."
        ),
        "en": (
            "You are a specialized legal assistant. Provide accurate and helpful "
            "information with references to relevant legal sources."
        ),
    }

    async def get_template(
        self,
        task_type: str,
        jurisdiction: str,
        language: str,
    ) -> PromptTemplate:
        """Get the appropriate prompt template."""
        template = self._TEMPLATES.get(task_type)
        if template is None:
            # Return a generic template
            return PromptTemplate(
                id="generic-v1",
                name="Generic Template",
                template="Process the following request:\n{input}",
                variables=["input"],
            )
        return template

    async def assemble_prompt(
        self,
        template: PromptTemplate,
        variables: Dict[str, Any],
        context_documents: Optional[List[str]] = None,
    ) -> AssembledPrompt:
        """Assemble a complete prompt from template and variables."""
        # Add context documents to variables if provided
        if context_documents:
            variables["context"] = "\n\n".join(context_documents)

        # Simple variable substitution
        prompt = template.template
        for var, value in variables.items():
            placeholder = "{" + var + "}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, str(value))

        language = variables.get("language", "en")
        system_prompt = self._SYSTEM_PROMPTS.get(language, self._SYSTEM_PROMPTS["en"])

        return AssembledPrompt(
            prompt=prompt,
            system_prompt=system_prompt,
            template_id=template.id,
            template_version=template.version,
            metadata={"stub": True},
        )

    async def get_system_prompt(
        self,
        task_type: str,
        jurisdiction: str,
        language: str,
    ) -> str:
        """Get the system prompt for a task type."""
        return self._SYSTEM_PROMPTS.get(language, self._SYSTEM_PROMPTS["en"])

    async def health_check(self) -> bool:
        """Always returns True for stub."""
        return True


# Default instance for dependency injection
stub_prompt_service = StubPromptService()
