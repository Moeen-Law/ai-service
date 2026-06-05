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
    "أنت 'مُعين'، مساعد قانوني مصري ذكي، رصين، ومحترف.\n"
    "مهمتك الأساسية هي تقديم استشارات قانونية مبنية **فقط** على المواد المتاحة في (Context). يمنع تماماً الاختلاق (Zero Hallucination).\n\n"
    
    "قواعد صارمة للرد:\n"
    "1. لا تخترع مواد أو أرقام غير موجودة في السياق إطلاقاً.\n"
    "2. إذا كان السياق غير كافٍ، توقف واكتب: 'عذراً، لا تتوفر لدي نصوص قانونية دقيقة للإجابة على هذا السؤال، يُنصح باستشارة محامٍ متخصص'.\n"
    "3. لا تكرر ديباجة اسم القانون مع كل مادة (اذكرها مرة واحدة فقط).\n"
    "4. لا ترحب بالمستخدم في بداية الرد إلا إذا كان [سجل المحادثة] فارغاً.\n"
    "5. تجنب تماماً النصائح الطبية أو النفسية.\n"
    "6. لا تضمّن أي فواصل أو حدود رسومية (مثل ═، ─) في الإجابة النهائية.\n"
    "7. التزم حرفياً بالهيكل الموجود في نهاية الطلب لترتيب إجابتك."
)

# ---------------------------------------------------------------------------
# Prompt templates per task type
# ---------------------------------------------------------------------------
_TEMPLATES: Dict[str, PromptTemplate] = {
    "LEGAL_CHAT": PromptTemplate(
        id="legal-chat-v9",
        name="Legal Chat — مُعين (Forced Format)",
        template=(
            "=== السياق القانوني (Ground Truth) ===\n"
            "استخرج الإجابة من هذه المواد فقط:\n"
            "{context}\n\n"
            "=== سجل المحادثة ===\n"
            "{conversation_history}\n\n"
            "=== سؤال المستخدم ===\n"
            "{question}\n\n"
            "أنت مجبر على الرد باستخدام هذا الهيكل حرفياً (لا تضف أي عناوين خارجية):\n\n"
            "أهلاً بك! ... (اكتب ترحيب ومقدمة سريعة في سطر واحد فقط إذا كان سجل المحادثة فارغاً)\n\n"
            "📌 **الخلاصة القانونية:**\n"
            "[اكتب جملة واحدة أو جملتين كحد أقصى تجيب مباشرة على السؤال]\n\n"
            "📖 **السند القانوني:**\n"
            "وفقاً لـ [اسم القانون ورقم السنة مرة واحدة]:\n"
            "- **المادة [رقم]:** \"[اقتبس النص الحرفي للمادة]\"\n"
            "(اشرح المعنى بتبسيط شديد أسفل المادة إذا لزم الأمر)\n\n"
            "✅ **نصيحة عملية:**\n"
            "[خطوات تطبيقية محددة أو لمن يتجه المستخدم]\n\n"
            "⚠️ **تنبيه هام:**\n"
            "[تنبيه عن استثناءات أو تقادم، أو احذف هذا العنوان إن لم يوجد]"
        ),
        variables=["conversation_history", "context", "question"],
        description="Forced formatting via template injection to overcome recency bias.",
    ),
    "CASE_EVALUATION": PromptTemplate(
        id="case-eval-v3",
        name="Case Evaluation — مُعين",
        template=(
            "=== المواد القانونية الحاكمة (Strict Context) ===\n"
            "قيم القضية بناءً على هذه المواد فقط:\n"
            "{context}\n\n"
            "=== وصف القضية ===\n"
            "{case_description}\n\n"
            "المطلوب: قم بتقييم القضية من الناحية القانونية وحدد:\n"
            "1. نقاط القوة\n"
            "2. نقاط الضعف\n"
            "3. التوصية القانونية\n\n"
            "التقييم:"
        ),
        variables=["context", "case_description"],
        description="Case evaluation with strict context isolation.",
    ),
    "CONTRACT_ANALYSIS": PromptTemplate(
        id="contract-analysis-v3",
        name="Contract Analysis — مُعين",
        template=(
            "=== المواد القانونية الحاكمة (Strict Context) ===\n"
            "استخدم هذه المواد كمرجعية لتحليل العقد:\n"
            "{context}\n\n"
            "=== نص العقد المراد تحليله ===\n"
            "{contract_text}\n\n"
            "المطلوب: حلل العقد وحدد:\n"
            "1. المخاطر القانونية مع مستوى كل خطر (عالي/متوسط/منخفض)\n"
            "2. ملخص التحليل\n"
            "3. التوصيات\n\n"
            "التحليل:"
        ),
        variables=["context", "contract_text"],
        description="Contract risk analysis with strict boundaries.",
    ),
    "CONTRACT_REFRAMING": PromptTemplate(
        id="contract-reframe-v3",
        name="Contract Reframing — مُعين",
        template=(
            "=== المواد القانونية الحاكمة (Strict Context) ===\n"
            "التزم بهذه المواد أثناء إعادة الصياغة:\n"
            "{context}\n\n"
            "=== البند الأصلي ===\n"
            "{clause_text}\n\n"
            "=== المنظور المطلوب ===\n"
            "{target_perspective}\n\n"
            "المطلوب: أعد صياغة البند مع:\n"
            "1. الحفاظ على المعنى القانوني الأساسي\n"
            "2. تعديل الصياغة لتناسب المنظور المطلوب\n"
            "3. ملخص التغييرات\n\n"
            "البند المعاد صياغته:"
        ),
        variables=["context", "clause_text", "target_perspective"],
        description="Contract clause reframing prompt.",
    ),
    "DOCUMENT_GENERATION": PromptTemplate(
        id="doc-gen-v3",
        name="Document Generation — مُعين",
        template=(
            "=== المواد القانونية الحاكمة (Strict Context) ===\n"
            "استند على هذه المواد في صياغة المستند:\n"
            "{context}\n\n"
            "=== تفاصيل المستند ===\n"
            "نوع المستند: {document_type}\n"
            "المعلومات المدخلة:\n{parameters}\n\n"
            "المطلوب: قم بإنشاء المستند القانوني بصياغة مهنية ودقيقة.\n\n"
            "المستند:"
        ),
        variables=["context", "document_type", "parameters"],
        description="Legal document generation prompt.",
    ),
}

# ---------------------------------------------------------------------------
# LLM query-rewriting prompt (used by the query pipeline)
# ---------------------------------------------------------------------------
QUERY_REWRITE_PROMPT = (
    "You are an Egyptian legal expert. Analyze the following question "
    "and respond ONLY with valid JSON matching this exact schema. "
    "Do not include markdown formatting or json code blocks.\n\n"
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

        # Set default values if not provided
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