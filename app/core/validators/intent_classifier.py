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

    # System prompt for intent classification + query rewriting (Arabic-first)
    _SYSTEM_PROMPT = """
    You are "Mueen" (معين), an intelligent and friendly Egyptian AI legal assistant. 

Your task is to analyze the user's message, determine their intent, and return a strict JSON object.

# INSTRUCTIONS:
1. Read the provided "Conversation History" (if any) to understand the context.
2. Identify the intent of the LATEST user message. Use the history ONLY to resolve pronouns or missing context in the latest message (e.g., if the user asks "What if it's with poison?", the history will tell you they mean "murder"). Do NOT classify the history itself.
3. If it is a "legal_query", extract the legal domain, keywords, and likely articles (do NOT answer the legal question).
4. If it is "chitchat", "vague", or "out_of_scope", you MUST generate a direct response acting as Mueen in the "reply" field. The "reply" MUST be in Arabic.

# INTENT CATEGORIES:
1. "legal_query": Clear questions about Egyptian law, contracts, or document analysis (e.g., "ما حقوقي في الفصل التعسفي؟", "حلل هذا العقد").
2. "chitchat": Greetings, thanks, or casual talk (e.g., "شكرا", "ازيك").
3. "vague": A legal query that is too broad and needs clarification (e.g., "حقوقي؟", "القانون").
4. "out_of_scope": Topics completely unrelated to law (e.g., "الطقس", "أين أكل؟").

# LEGAL DOMAINS (Use ONLY these):
["penal", "civil", "labor", "constitution", "commercial", "criminal_procedure"]

# EXPECTED JSON SCHEMA:

For "legal_query":
{
    "intent": "legal_query",
    "confidence": 0.95,
    "reasoning": "Brief explanation in English",
    "domain": "labor",
    "keywords": ["فصل", "تعسفي", "حقوق"],
    "likely_articles": ["206", "207"],
    "reply": null
}

For "chitchat", "vague", or "out_of_scope":
{
    "intent": "chitchat | vague | out_of_scope",
    "confidence": 0.90,
    "reasoning": "Brief explanation in English",
    "domain": null,
    "keywords": [],
    "likely_articles": [],
    "reply": "Your intelligent, helpful response in ARABIC here, acting as Mueen."
}

# CRITICAL RULES:
- The "reply" field MUST ALWAYS BE IN ARABIC.
- Do NOT make up laws or articles. If unsure, return an empty array [].
- Requests to analyze attached documents ALWAYS count as "legal_query".
- Output ONLY valid JSON, without any markdown formatting like ```json.
    """

    def __init__(self, llm_service: LLMServiceInterface) -> None:
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
                confidence=result.confidence,
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
                raise ValueError(f"Invalid intent: {intent_str}")

            intent = Intent(intent_str)

            raw_confidence = float(data.get("confidence", 0.5))
            if raw_confidence < 0.0 or raw_confidence > 1.0:
                logger.warning("confidence_out_of_range", raw_value=raw_confidence, intent=intent_str)
            confidence = max(0.0, min(1.0, raw_confidence))

            reasoning = str(data.get("reasoning", "")).strip() or f"تم تصنيف الرسالة كـ {intent.value}"

            clarification = data.get("suggested_clarification")
            clarification = None if clarification in ["null", None] else str(clarification).strip()

            reply = data.get("reply")
            reply = None if reply in ["null", None, ""] else str(reply).strip()

            domain = None
            keywords = []
            likely_articles = []

            if intent == Intent.LEGAL_QUERY:
                domain = data.get("domain")
                domain = None if domain in ["null", None, ""] else domain

                keywords = data.get("keywords", [])
                keywords = [str(k).strip() for k in keywords if k] if isinstance(keywords, list) else []

                likely_articles = data.get("likely_articles", [])
                likely_articles = [str(a).strip() for a in likely_articles if a] if isinstance(likely_articles, list) else []
                original_likely_articles = likely_articles
                likely_articles = [a for a in likely_articles if re.fullmatch(r"\d+", a)]

                dropped_articles = sorted(set(original_likely_articles) - set(likely_articles))
                if dropped_articles:
                    logger.warning("likely_articles_filtered", dropped=dropped_articles)

            return IntentClassificationResult(
                intent=intent,
                confidence=confidence,
                reasoning=reasoning,
                suggested_clarification=clarification,
                domain=domain,
                keywords=keywords,
                likely_articles=likely_articles,
                reply=reply,
                history=history,
            )
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            logger.warning("classification_response_parse_failed", error=str(e))
            raise ValueError(f"Invalid classification response format: {response_text}") from e