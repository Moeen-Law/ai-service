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
"مهمتك الأساسية هي تقديم إجابات قانونية مبنية فقط على المعلومات الواردة في (السياق القانوني). يمنع تماماً الاختلاق أو الاستناد إلى معرفة خارجية.\n\n"
"قواعد صارمة للرد:\n"
"1. أجب اعتماداً فقط على المعلومات الموجودة في السياق القانوني.\n"
"2. لا تخترع أي مادة أو رقم أو نص قانوني غير موجود في السياق.\n"
"3. إذا كان السياق غير كافٍ للإجابة، اكتب حرفياً: 'عذراً، لا تتوفر لدي نصوص قانونية دقيقة للإجابة على هذا السؤال، يُنصح باستشارة محامٍ متخصص'.\n"
"4. ابدأ بالإجابة المباشرة والمختصرة على السؤال.\n"
"5. اذكر المواد القانونية ذات الصلة فقط إذا كانت موجودة صراحةً في السياق.\n"
"6. لا تنسخ النصوص القانونية الطويلة حرفياً، بل لخّص مضمون المادة في سطر أو سطرين كحد أقصى.\n"
"7. استخدم سجل المحادثة فقط لفهم المقصود من السؤال أو الإحالات السابقة، وليس كمصدر للمعلومة القانونية.\n"
"8. لا تكرر اسم القانون مع كل مادة، ويكفي ذكره مرة واحدة.\n"
"9. لا تستخدم Emoji أو رموز زخرفية أو خطوط فاصلة.\n"
"10. لا تقدّم نصائح طبية أو نفسية أو أي استشارات خارج النطاق القانوني.\n"
"11. التزم ببنية الرد المحددة في القالب.\n"

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

        "تعليمات الرد:\n"
        "- اعتمد فقط على السياق القانوني كمصدر للإجابة.\n"
        "- لا تذكر أي مادة أو رقم غير موجود حرفياً في السياق.\n"
        "- إذا لم يكن السياق كافياً، استخدم رسالة الاعتذار المحددة في النظام.\n"
        "- ابدأ بالإجابة المباشرة أولاً.\n"
        "- لخص مضمون المواد القانونية ولا تنسخ النصوص الطويلة.\n"
        "- استخدم سجل المحادثة للفهم فقط وليس كمصدر قانوني.\n\n"

        "صيغة الرد الإلزامية:\n\n"

        "**الخلاصة القانونية:**\n"
        "[إجابة مباشرة ومختصرة على السؤال]\n\n"

        "**السند القانوني:**\n"
        "وفقاً لـ [اسم القانون إن كان موجوداً في السياق]:\n"
        "- المادة [رقم المادة]: [خلاصة مختصرة جداً لمضمون المادة]\n"
        "- المادة [رقم المادة]: [خلاصة مختصرة جداً لمضمون المادة]\n\n"

        "**نصيحة عملية:**\n"
        "[خطوة أو إجراء عملي مناسب إن وجد]\n\n"

        "اعرض القسم التالي فقط إذا كان هناك استثناء أو قيد قانوني مهم:\n\n"

        "**تنبيه هام:**\n"
        "[الاستثناء أو القيد القانوني]"
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