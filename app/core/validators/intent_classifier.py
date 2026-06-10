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
    reply: Optional[str] = None
    history: Optional[str] = None


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

    _DOCUMENT_GENERATION_TRIGGERS = (
        "إنشاء عقد",
        "كتابة عقد",
        "صياغة عقد",
        "إنشاء مذكرة",
        "إنشاء صحيفة دعوى",
        "إنشاء مستند قانوني",
    )
    _DOCUMENT_GENERATION_REPLY = (
        "هذه الخدمة مخصصة للاستشارات القانونية وتحليل الملفات فقط. "
        "لإنشاء المستندات القانونية استخدم خدمة إنشاء المستندات."
    )

    # System prompt for intent classification + query rewriting (Arabic-first)
    _SYSTEM_PROMPT = """
    You are "Mueen" (معين), an expert Egyptian legal knowledge extractor and intent classifier.
    Analyze the user's latest message and return a minimal JSON object.

    # INSTRUCTIONS:
    1. Identify the intent of the LATEST user message.
    2. If it is "legal_query", extract the domain, Arabic keywords, and PREDICT the exact Egyptian law article numbers related to the situation.
    3. If it is "chitchat", "vague", or "out_of_scope", you MUST generate a direct response acting as Mueen in the "reply" field. The "reply" MUST be in Arabic.
    4. CRITICAL: For crime-related stories (fights, murder, self-defense, theft, etc.), the domain MUST be "penal".
    5. If the user requests any of the following:
        - إنشاء عقد
        - كتابة عقد
        - صياغة عقد
        - إنشاء مذكرة
        - إنشاء صحيفة دعوى
        - إنشاء مستند قانوني

    Classify as OUT_OF_SCOPE and set reply exactly to:
    "هذه الخدمة مخصصة للاستشارات القانونية وتحليل الملفات فقط. لإنشاء المستندات القانونية استخدم خدمة إنشاء المستندات."

    # INTENT CATEGORIES:
    - "legal_query": Questions about Egyptian law, contracts, or document analysis.
    - "chitchat": Greetings, thanks, or casual talk.
    - "vague": A legal query that is too broad and needs clarification.
    - "out_of_scope": Topics completely unrelated to law.
    

    # LEGAL DOMAINS:
    You MUST choose ONLY ONE of the following or null:
    ["penal", "criminal_procedure", "civil", "labor", "constitution"]

    # ARTICLE PREDICTION RULES (CRITICAL):
    - In "likely_articles", recall from your knowledge base the exact article numbers from the matching Egyptian Code.
    - Self-Defense (دفاع شرعي) -> Predict ["245", "246", "247", "248", "249", "250", "251"]
    - Theft (سرقة) -> Predict ["311", "313", "314", "315", "316", "317", "318"]
    - Murder/Manslaughter (قتل عمد / ضرب أفضى إلى موت) -> Predict ["230", "234", "236"]
    - Extortion/Bribery (ابتزاز / رشوة) -> Predict ["325", "326", "103"]
    - Financial Damage/Breach of Trust (خيانة أمانة / تبديد) -> Predict ["341", "342"]
    - Return ONLY the numbers as strings in an array, sorted by core rule first.
    - If you are completely unsure about the specific article numbers, return an empty array []. Do NOT invent random numbers.

    # EXPECTED JSON SCHEMA:

    For "legal_query":
    {
        "intent": "legal_query",
        "domain": "penal",
        "keywords": ["دفاع شرعي", "مشاجرة", "قتل"],
        "likely_articles": ["245", "246"],
        "reply": null
    }

    For others:
    {
        "intent": "chitchat | vague | out_of_scope",
        "domain": null,
        "keywords": [],
        "likely_articles": [],
        "reply": "Your intelligent, helpful response in ARABIC here."
    }

    # CRITICAL RULES:
    - The "reply" field MUST ALWAYS BE IN ARABIC.
    - Output ONLY valid JSON, no markdown formatting like ```json.
    """
    def __init__(self, llm_service: LLMServiceInterface=None) -> None:
        """
        Initialize the intent classifier.

        Args:
            llm_service: The LLM service to use for classification
        """
        self._llm_service = llm_service

    async def classify(self, message: str, history: str = "") -> IntentClassificationResult:
        """
        Classify the intent of a user message.

        Args:
            message: The user message to classify
            history:the history of user messages

        Returns:
            IntentClassificationResult with the classified intent and metadata

        Raises:
            ValueError: If LLM response is invalid or unparseable
        """
        logger.debug("classifying_intent", message_preview=message[:100])
        document_generation_result = self._classify_document_generation_request(
            message=message,
            history=history,
        )
        if document_generation_result is not None:
            return document_generation_result

        prompt = self._build_classification_prompt(message, history)
        request = LLMRequest(
            prompt=prompt,
            system_prompt=self._SYSTEM_PROMPT,
            temperature=0.2,  # Low temperature for consistent classification
        )
        try:
            response = await self._llm_service.generate(request)
            result = self._parse_classification_response(response.content, history)
            logger.debug(
                "classified",
                intent=result.intent.value,
            )
            return result
        except ValueError as e:
            logger.warning("classification_parse_failed", error=str(e))
            return IntentClassificationResult(
                intent=Intent.VAGUE,
                confidence=0.5,
                reasoning="فشل التصنيف - تم افتراض سؤال غير واضح",
                history=history,
            )
        except Exception as e:
            logger.error("classification_failed", error=str(e))
            raise

    def _classify_document_generation_request(
        self,
        message: str,
        history: str,
    ) -> Optional[IntentClassificationResult]:
        normalized_message = re.sub(r"\s+", " ", message or "").strip()
        if not any(trigger in normalized_message for trigger in self._DOCUMENT_GENERATION_TRIGGERS):
            return None

        return IntentClassificationResult(
            intent=Intent.OUT_OF_SCOPE,
            confidence=1.0,
            reasoning="Document generation request is outside legal chat scope",
            domain=None,
            keywords=[],
            likely_articles=[],
            reply=self._DOCUMENT_GENERATION_REPLY,
            history=history,
        )

    def _build_classification_prompt(self, message: str, history: str) -> str:
        """
        Build the classification prompt from the system instructions and user message.

        Args:
            message: The user message to classify
            history

        Returns:
            User message prompt for classification.
        """
        history_block = f"=== سجل المحادثة السابق (Conversation History) ===\n{history}\n\n" if history.strip() and history != "لا يوجد سجل محادثة سابق." else ""
        return f'{history_block}=== الرسالة المراد تصنيفها (Latest Message) ===\n"{message}"\n\nاستجابتك (JSON فقط):'
    def _parse_classification_response(self, response_text: str, history: str = "") -> IntentClassificationResult:
        """
        Parse LLM response into IntentClassificationResult.

        Handles both simple intent classification and combined intent+rewrite responses.

        Args:
            response_text: The raw response from LLM
            history

        Returns:
            Parsed IntentClassificationResult

        Raises:
            ValueError: If response format is invalid
        """
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start == -1 or json_end <= json_start:
                raise ValueError("No JSON found in response")

            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)

            intent_str = data.get("intent", "").lower()
            if intent_str not in [e.value for e in Intent]:
                intent_str = "legal_query" # Fallback safe

            intent = Intent(intent_str)
            reply = data.get("reply")
            reply = None if reply in ["null", None, ""] else str(reply).strip()

            domain = data.get("domain")
            domain = None if domain in ["null", None, ""] else domain

            keywords = data.get("keywords", [])
            keywords = [str(k).strip() for k in keywords if k] if isinstance(keywords, list) else []

            likely_articles = data.get("likely_articles", [])
            likely_articles = [str(a).strip() for a in likely_articles if a] if isinstance(likely_articles, list) else []

            return IntentClassificationResult(
                intent=intent,
                confidence=1.0,
                reasoning="Fast Groq Classification",
                suggested_clarification=None,
                domain=domain,         
                keywords=keywords,      
                likely_articles=likely_articles, 
                reply=reply,
                history=history,
            )
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            logger.warning("classification_response_parse_failed", error=str(e))
            raise ValueError(f"Invalid classification response format: {response_text}") from e
