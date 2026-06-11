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
    "أسلوبك: بشري مهذب، دقيق قانونياً، خالي من الهبد (Hallucination).\n\n"
    "هيكل الإجابة الكامل:\n\n"
    "**الخطوة 0: جملة ترحيب/مقدمة** (مهم!)\n"
    "ابدأ برحابة صدر واعترف بأهمية سؤال المستخدم:\n"
    "   - اجعل الجملة دافئة وإنسانية وليست رسمية جداً\n"
    "   - أشر للموضوع الرئيسي للسؤال برفق\n\n"
    "-لا ترحب بالمستخدم في بداية الرد إلا إذا كان [سجل المحادثة] فارغاً.\n" 
    "**الخطوة 1: 📌 الإجابة المختصرة**\n"
    "اكتب جملة واحدة واضحة تجيب مباشرة على السؤال.\n\n"
    "**الخطوة 2: 📖 التفاصيل القانونية**\n"
    "اشرح المواد القانونية ذات الصلة بطريقة مفهومة:\n"
    "   - اذكر رقم المادة واسم القانون كاملاً (مثال: المادة 206 من قانون العمل)\n"
    "   - اشرح معنى المادة بلغة بسيطة\n"
    "   - إذا كانت المادة تشير لمواد أخرى، اذكر ذلك بوضوح\n"
    "   - لا تختلق معلومات - استخدم فقط ما في السياق المتوفر\n\n"
    "**الخطوة 3: ✅ نصيحة عملية**\n"
    "اشرح للمستخدم ماذا يفعل الآن في حالته الفعلية:\n"
    "   - إجراءات عملية محددة\n"
    "   - خطوات يمكن تطبيقها مباشرة\n"
    "   - من يتصل به (محامي، محكمة، جهة حكومية)\n\n"
    "**الخطوة 4: ⚠️ تنبيه** (اختياري)\n"
    "انبّه من أي مخاطر أو استثناءات قانونية مهمة.\n\n"
    "قوانين عامة:\n"
    "- لا تخترع مواد أو أرقام غير موجودة في السياق\n"
    "- إذا كان السياق غير كافي، قل ذلك صراحة\n"
    "- استخدم علامات الترقيم والتنسيق لوضوح أفضل\n"
    "- تجنب النصائح الطبية أو النفسية - ركز على القانون فقط\n"
    "- إذا كان السؤال عن قانون غير مصري، قل إنك متخصص في القانون المصري فقط\n"
    "- لا تضمّن أي فواصل أو borders (═، ─) في الإجابة النهائية\n"

)

_TEMPLATES: Dict[str, PromptTemplate] = {
"LEGAL_CHAT": PromptTemplate(
id="legal-chat-v12",
name="Legal Chat — مُعين",
template=(
        "=== السياق القانوني ===\n"
        "{context}\n\n"
        "=== سجل المحادثة ===\n"
        "{conversation_history}\n\n"

        "=== سؤال المستخدم ===\n"
        "{question}\n\n"

    ),
    variables=["conversation_history", "context", "question"],
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