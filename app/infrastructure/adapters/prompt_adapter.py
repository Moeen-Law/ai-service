"""
Legal Prompt Service Adapter

Production prompt service for Egyptian legal AI assistant (مُعين).
Contains the real Arabic legal prompt templates and the LLM query-rewriting prompt.
"""

from typing import Any, Dict, List, Optional

from app.interfaces.ai.prompt_service import (
    AssembledPrompt,
    PromptServiceInterface,
    PromptTemplate,
)

# ---------------------------------------------------------------------------
# Core system prompt — shared across all legal workflows
# ---------------------------------------------------------------------------
_MOEIN_SYSTEM_PROMPT = (
    'أنت "مُعين"، مساعد قانوني مصري ذكي متخصص في القانون المصري.\n\n'
    "أولًا: تفاعل بأسلوب بشري مهذب وواضح، ثم قدم المحتوى القانوني بدقة.\n\n"
    "استخدم المواد القانونية التالية للإجابة على سؤال المستخدم بدقة ووضوح:\n\n"
    "**تعليمات هامة:**\n"
    "1. إذا وجدت مواد قانونية تجيب على السؤال مباشرة، استخدمها بالكامل\n"
    '2. إذا كانت المواد تشير إلى مواد أخرى (مثل "تطبق أحكام المواد 240 و241 و242")، اذكر ذلك بوضوح\n'
    "3. قدم الإجابة بشكل واضح مع ذكر رقم المادة واسم القانون بالكامل\n"
    "4. إذا لم تجد إجابة مباشرة، لا تخترع معلومات - قل ذلك بوضوح\n"
    "5. ركز على المواد من قانون العقوبات إذا كان السؤال عن جرائم أو عقوبات\n"
    "6. إذا كانت رسالة المستخدم اجتماعية فقط (مثل: شكرا/أهلا/تمام)، رد برد اجتماعي قصير ومهذب بدون افتعال إجابة قانونية\n"
    "7. إذا كان الطلب غير واضح، اسأل سؤال توضيحي واحد مباشر قبل التوسع\n"
    "8. استخدم لغة عربية بسيطة مفهومة، ثم أضف التفاصيل القانونية عند الحاجة\n"
)

# ---------------------------------------------------------------------------
# Prompt templates per task type
# ---------------------------------------------------------------------------
_TEMPLATES: Dict[str, PromptTemplate] = {
    "LEGAL_CHAT": PromptTemplate(
        id="legal-chat-v2",
        name="Legal Chat — مُعين",
        template=(
            "{system_prompt}\n\n"
            "سجل المحادثة السابق (قد يكون فارغاً):\n{conversation_history}\n\n"
            "المواد القانونية المتوفرة:\n{context}\n\n"
            "السؤال:\n{question}\n\n"
            "الإجابة:\n"
        ),
        variables=["system_prompt", "conversation_history", "context", "question"],
        description="Full legal chat prompt with context and system instructions",
    ),
    "CASE_EVALUATION": PromptTemplate(
        id="case-eval-v2",
        name="Case Evaluation — مُعين",
        template=(
            "{system_prompt}\n\n"
            "المواد القانونية ذات الصلة:\n{context}\n\n"
            "وصف القضية:\n{case_description}\n\n"
            "المطلوب: قم بتقييم القضية من الناحية القانونية وحدد:\n"
            "1. نقاط القوة\n"
            "2. نقاط الضعف\n"
            "3. التوصية القانونية\n\n"
            "التقييم:\n"
        ),
        variables=["system_prompt", "context", "case_description"],
        description="Case evaluation with strengths/weaknesses analysis",
    ),
    "CONTRACT_ANALYSIS": PromptTemplate(
        id="contract-analysis-v2",
        name="Contract Analysis — مُعين",
        template=(
            "{system_prompt}\n\n"
            "المواد القانونية ذات الصلة:\n{context}\n\n"
            "نص العقد المراد تحليله:\n{contract_text}\n\n"
            "نوع التحليل: {analysis_type}\n\n"
            "المطلوب: حلل العقد وحدد:\n"
            "1. المخاطر القانونية مع مستوى كل خطر (عالي/متوسط/منخفض)\n"
            "2. ملخص التحليل\n"
            "3. التوصيات\n\n"
            "التحليل:\n"
        ),
        variables=["system_prompt", "context", "contract_text", "analysis_type"],
        description="Contract risk analysis prompt",
    ),
    "CONTRACT_REFRAMING": PromptTemplate(
        id="contract-reframe-v2",
        name="Contract Reframing — مُعين",
        template=(
            "{system_prompt}\n\n"
            "المواد القانونية ذات الصلة:\n{context}\n\n"
            "البند الأصلي:\n{clause_text}\n\n"
            "المنظور المطلوب: {target_perspective}\n\n"
            "المطلوب: أعد صياغة البند مع:\n"
            "1. الحفاظ على المعنى القانوني الأساسي\n"
            "2. تعديل الصياغة لتناسب المنظور المطلوب\n"
            "3. ملخص التغييرات\n\n"
            "البند المعاد صياغته:\n"
        ),
        variables=["system_prompt", "context", "clause_text", "target_perspective"],
        description="Contract clause reframing prompt",
    ),
    "DOCUMENT_GENERATION": PromptTemplate(
        id="doc-gen-v2",
        name="Document Generation — مُعين",
        template=(
            "{system_prompt}\n\n"
            "المواد القانونية ذات الصلة:\n{context}\n\n"
            "نوع المستند: {document_type}\n"
            "المعلومات:\n{parameters}\n\n"
            "المطلوب: قم بإنشاء المستند القانوني بصياغة مهنية ودقيقة.\n\n"
            "المستند:\n"
        ),
        variables=["system_prompt", "context", "document_type", "parameters"],
        description="Legal document generation prompt",
    ),
}

# ---------------------------------------------------------------------------
# LLM query-rewriting prompt (used by the query pipeline)
# ---------------------------------------------------------------------------
QUERY_REWRITE_PROMPT = (
    "You are an Egyptian legal expert. Analyze the following question "
    "and respond ONLY with valid JSON matching this exact schema:\n\n"
    "{\n"
    '  "domain": "<one of: penal | civil | labor | constitution | commercial | criminal_procedure | null>",\n'
    '  "keywords": ["<formal Arabic legal keyword>", "..."],\n'
    '  "likely_articles": ["<article number string>", "..."]\n'
    "}\n\n"
    "Rules:\n"
    "- domain: single best Egyptian law domain for this question; use null if unclear.\n"
    "- keywords: 8-15 formal Arabic legal terms / concepts relevant to the question. "
    "Do NOT include numbers representing durations or monetary amounts.\n"
    "- likely_articles: article numbers you are confident directly address this "
    "situation (e.g. '163', '240'). Use empty list if uncertain.\n\n"
    "Question: {question}\n\n"
    "JSON:"
)


class LegalPromptService(PromptServiceInterface):
    """
    Production prompt service for Egyptian legal workflows.

    Provides the real Arabic مُعين prompts and supports
    query rewriting via the QUERY_REWRITE_PROMPT.
    """

    async def get_template(
        self,
        task_type: str,
        jurisdiction: str,
        language: str,
    ) -> PromptTemplate:
        template = _TEMPLATES.get(task_type)
        if template is None:
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
        # Inject context documents
        if context_documents:
            variables["context"] = "\n\n".join(context_documents)

        # Always inject the system prompt
        variables.setdefault("system_prompt", _MOEIN_SYSTEM_PROMPT)
        variables.setdefault("conversation_history", "لا يوجد سجل محادثة سابق.")

        # Variable substitution
        prompt = template.template
        for var, value in variables.items():
            placeholder = "{" + var + "}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, str(value))

        return AssembledPrompt(
            prompt=prompt,
            system_prompt=_MOEIN_SYSTEM_PROMPT,
            template_id=template.id,
            template_version=template.version,
        )

    async def get_system_prompt(
        self,
        task_type: str,
        jurisdiction: str,
        language: str,
    ) -> str:
        return _MOEIN_SYSTEM_PROMPT

    async def health_check(self) -> bool:
        return True


# Default instance for dependency injection
prompt_service = LegalPromptService()
