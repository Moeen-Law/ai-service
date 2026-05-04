"""
Intent Classification

Classifies user messages into semantic intents before processing.
Uses LLM to determine if a message is chitchat, a legal query, vague, or out-of-scope.
Also extracts domain and keywords for legal queries in a single LLM call.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from app.core.domain.enums import Intent
from app.infrastructure.logging.logger import get_logger
from app.interfaces.ai.llm_service import LLMRequest, LLMServiceInterface

logger = get_logger(__name__)


@dataclass
class IntentClassificationResult:
    """
    Result of intent classification and optionally query rewriting for a user message.

    Attributes:
        intent: The classified intent (CHITCHAT, LEGAL_QUERY, VAGUE, OUT_OF_SCOPE)
        confidence: Confidence score between 0.0 and 1.0
        reasoning: Brief explanation of why this intent was chosen
        suggested_clarification: Optional clarification question if intent is VAGUE
        # Rewrite data (only when intent is LEGAL_QUERY)
        domain: Detected legal domain (e.g., "civil", "labor")
        keywords: List of extracted keywords for retrieval
        likely_articles: List of article numbers likely related to the query
    """

    intent: Intent
    confidence: float
    reasoning: str
    suggested_clarification: Optional[str] = None
    domain: Optional[str] = None
    keywords: list = field(default_factory=list)
    likely_articles: list = field(default_factory=list)


IntentResult = IntentClassificationResult


class IntentClassifier:
    """
    Classifies user messages using LLM without consuming RAG resources.

    This lightweight classifier runs before expensive retrieval operations,
    routing messages appropriately:
    - CHITCHAT → Quick response without RAG
    - LEGAL_QUERY → Full processing with retrieval
    - VAGUE → Request clarification
    - OUT_OF_SCOPE → Polite redirect
    """

    # System prompt for intent classification + query rewriting (Arabic-first)
    _SYSTEM_PROMPT = """أنت مساعد ذكي متخصص في تصنيف وتحليل الرسائل القانونية.

مهمتك المربوطة:
1. تحديد نية المستخدم من رسالته
2. إذا كانت نية قانونية → استخراج domain والكلمات المفتاحية والمواد المتعلقة

التصنيفات الممكنة:
1. **chitchat** - تحيات، شكر، أحاديث عادية (لا علاقة بالقانون)
   أمثلة: "مرحبا"، "شكراً"، "تمام"

2. **legal_query** - سؤال قانوني واضح ومحدد عن القانون المصري
   أمثلة: "ما حقوقي في الفصل التعسفي؟"، "هل يمكن فسخ العقد؟"، "حلل العقد المرفق"، "شوف العقود دي وقولي حقوقي"
   → في هذه الحالة استخرج أيضاً: domain, keywords, likely_articles

3. **vague** - سؤال قانوني لكن غير واضح أو يحتاج توضيح
   أمثلة: "حقوقي؟"، "القانون"، "مساعدة"
   → لا تستخرج domain/keywords - فقط اطلب توضيح

4. **out_of_scope** - موضوع غير قانوني تماماً
   أمثلة: "أين أشتري بيتزا؟"، "الطقس اليوم"
   → رفض لطيف و redirect للقانون

استخدم الـ LEGAL_DOMAINS المتاحة:
- penal (جنائي)
- civil (مدني)
- labor (عمل)
- constitution (دستوري)
- commercial (تجاري)
- criminal_procedure (إجراءات جنائية)

استجابتك يجب أن تكون JSON بهذا الشكل:

للـ LEGAL_QUERY:
{
    "intent": "legal_query",
    "confidence": 0.95,
    "reasoning": "السؤال يتعلق بحقوق العمل",
    "domain": "labor",
    "keywords": ["فصل", "تعسفي", "عمل"],
    "likely_articles": ["206", "207"]
}

للـ CHITCHAT/VAGUE/OUT_OF_SCOPE:
{
    "intent": "chitchat|vague|out_of_scope",
    "confidence": 0.9,
    "reasoning": "شرح مختصر",
    "suggested_clarification": "null أو سؤال توضيحي"
}

تنبيهات:
- لا تحاول الإجابة على السؤال، فقط حللّه
- استخرج domain و keywords فقط لـ legal_query
- كن حذراً من أسئلة "العمل" (labor) vs "المدني" (civil)
- لا تخترع مواد إذا لم تكن متأكداً - أرجع empty array
- أسئلة تحليل العقود والمستندات المرفقة تعتبر legal_query حتى لو كانت غير محددة تماماً
- أي طلب لتحليل أو مراجعة مستندات/عقود/اتفاقيات = legal_query"""


    def __init__(self, llm_service: LLMServiceInterface) -> None:
        """
        Initialize the intent classifier.

        Args:
            llm_service: The LLM service to use for classification
        """
        self._llm_service = llm_service

    async def classify(self, message: str) -> IntentClassificationResult:
        """
        Classify the intent of a user message.

        Args:
            message: The user message to classify

        Returns:
            IntentClassificationResult with the classified intent and metadata

        Raises:
            ValueError: If LLM response is invalid or unparseable
        """
        logger.debug("classifying_intent", message_preview=message[:100])

        prompt = self._build_classification_prompt(message)
        request = LLMRequest(
            prompt=prompt,
            system_prompt=self._SYSTEM_PROMPT,
            temperature=0.2,  # Low temperature for consistent classification
        )

        try:
            response = await self._llm_service.generate(request)
            result = self._parse_classification_response(response.content)
            logger.debug(
                "classified",
                intent=result.intent.value,
                confidence=result.confidence,
            )
            return result
        except ValueError as e:
            logger.warning("classification_parse_failed", error=str(e))
            # Graceful fallback: treat as vague if classification fails
            return IntentClassificationResult(
                intent=Intent.VAGUE,
                confidence=0.5,
                reasoning="فشل التصنيف - تم افتراض سؤال غير واضح",
            )
        except Exception as e:
            logger.error("classification_failed", error=str(e))
            raise

    def _build_classification_prompt(self, message: str) -> str:
        """
        Build the classification prompt from the system instructions and user message.

        Args:
            message: The user message to classify

        Returns:
            User message prompt for classification.
        """
        return f'الرسالة المراد تصنيفها:\n"{message}"\n\nاستجابتك (JSON فقط):'

    def _parse_classification_response(self, response_text: str) -> IntentClassificationResult:
        """
        Parse LLM response into IntentClassificationResult.

        Handles both simple intent classification and combined intent+rewrite responses.

        Args:
            response_text: The raw response from LLM

        Returns:
            Parsed IntentClassificationResult

        Raises:
            ValueError: If response format is invalid
        """
        try:
            # Extract JSON from response (handle cases with extra text)
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start == -1 or json_end <= json_start:
                raise ValueError("No JSON found in response")

            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)

            # Validate and parse intent
            intent_str = data.get("intent", "").lower()
            if intent_str not in [e.value for e in Intent]:
                raise ValueError(f"Invalid intent: {intent_str}")

            intent = Intent(intent_str)

            # Parse other fields with defaults
            raw_confidence = float(data.get("confidence", 0.5))
            if raw_confidence < 0.0 or raw_confidence > 1.0:
                logger.warning(
                    "confidence_out_of_range",
                    raw_value=raw_confidence,
                    intent=intent_str,
                )
            confidence = max(0.0, min(1.0, raw_confidence))  # Clamp to [0, 1]

            reasoning = str(data.get("reasoning", "")).strip()
            if not reasoning:
                reasoning = f"تم تصنيف الرسالة كـ {intent.value}"

            clarification = data.get("suggested_clarification")
            if clarification == "null" or clarification is None:
                clarification = None
            else:
                clarification = str(clarification).strip()

            # Parse rewrite data (only for legal_query)
            domain = None
            keywords = []
            likely_articles = []

            if intent == Intent.LEGAL_QUERY:
                domain = data.get("domain")
                if domain == "null" or not domain:
                    domain = None

                keywords = data.get("keywords", [])
                keywords = (
                    [str(k).strip() for k in keywords if k]
                    if isinstance(keywords, list)
                    else []
                )

                likely_articles = data.get("likely_articles", [])
                likely_articles = (
                    [str(a).strip() for a in likely_articles if a]
                    if isinstance(likely_articles, list)
                    else []
                )
                original_likely_articles = likely_articles
                likely_articles = [
                    a for a in likely_articles if re.fullmatch(r"\d+", a)
                ]
                dropped_articles = sorted(
                    set(original_likely_articles) - set(likely_articles)
                )
                if dropped_articles:
                    logger.warning(
                        "likely_articles_filtered",
                        dropped=dropped_articles,
                    )

            return IntentClassificationResult(
                intent=intent,
                confidence=confidence,
                reasoning=reasoning,
                suggested_clarification=clarification,
                domain=domain,
                keywords=keywords,
                likely_articles=likely_articles,
            )
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            logger.warning("classification_response_parse_failed", error=str(e))
            raise ValueError(f"Invalid classification response format: {response_text}") from e
