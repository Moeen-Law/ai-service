"""
API Request Schemas

Request schemas for the AI service endpoints.
Handles shape validation only - no business rules.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from app.core.domain.enums import TaskType

from .common import ContextSchema, OptionsSchema


class ConversationMessage(BaseModel):
    """
    A single message in conversation history.
    """

    role: str = Field(
        ...,
        description="Message role (user or assistant)",
        pattern="^(user|assistant)$",
    )
    content: str = Field(
        ...,
        description="Message content",
        min_length=1,
    )


class LegalChatPayload(BaseModel):
    """
    Payload schema for LEGAL_CHAT task type.
    """

    message: str = Field(
        ...,
        description="User's legal question or message",
        min_length=1,
        max_length=10000,
    )
    conversation_history: Optional[List[ConversationMessage]] = Field(
        default_factory=list,
        description="Previous conversation messages",
        max_length=50,
    )
    files_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional uploaded file IDs to analyze with the prompt",
        max_length=10,
    )

    @field_validator("files_ids")
    @classmethod
    def validate_files_ids(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        """Validate uploaded file IDs shape and size constraints."""
        if value is None:
            return value

        if not isinstance(value, list):
            raise ValueError("files_ids must be a list of file IDs")

        for file_id in value:
            if not isinstance(file_id, str) or not file_id.strip():
                raise ValueError("files_ids must contain non-empty string IDs")
            if len(file_id.strip()) > 128:
                raise ValueError("file ID length must be <= 128 characters")

        return value

    model_config = {
        "extra": "forbid",
    }


class DocumentGenerationPayload(BaseModel):
    """
    Payload schema for DOCUMENT_GENERATION task type.
    """

    document_type: str = Field(
        ...,
        description="Type of document to generate",
        min_length=1,
        max_length=100,
        examples=["contract", "agreement", "power_of_attorney"],
    )
    parameters: Dict[str, Any] = Field(
        ...,
        description="Document parameters",
    )

    model_config = {
        "extra": "forbid",
    }


class ContractAnalysisPayload(BaseModel):
    """
    Payload schema for CONTRACT_ANALYSIS task type.
    """

    contract_text: str = Field(
        ...,
        description="Contract text to analyze",
        min_length=1,
        max_length=100000,
    )
    analysis_type: str = Field(
        ...,
        description="Type of analysis to perform",
        examples=["risk_assessment", "clause_extraction", "compliance_check"],
    )
    focus_areas: Optional[List[str]] = Field(
        default=None,
        description="Specific areas to focus analysis on",
    )

    model_config = {
        "extra": "forbid",
    }


class ContractReframingPayload(BaseModel):
    """
    Payload schema for CONTRACT_REFRAMING task type.
    """

    clause_text: str = Field(
        ...,
        description="Contract clause to reframe",
        min_length=1,
        max_length=50000,
    )
    target_perspective: str = Field(
        ...,
        description="Perspective to reframe towards",
        examples=["balanced", "party_a_favorable", "party_b_favorable"],
    )
    preserve_intent: bool = Field(
        default=True,
        description="Whether to preserve the original intent",
    )

    model_config = {
        "extra": "forbid",
    }


class CaseEvaluationPayload(BaseModel):
    """
    Payload schema for CASE_EVALUATION task type.
    """

    case_description: str = Field(
        ...,
        description="Description of the legal case",
        min_length=1,
        max_length=100000,
    )
    case_type: Optional[str] = Field(
        default=None,
        description="Type of case",
        examples=["civil", "criminal", "commercial"],
    )
    evidence_summary: Optional[str] = Field(
        default=None,
        description="Summary of available evidence",
        max_length=50000,
    )

    model_config = {
        "extra": "forbid",
    }


class TaskRequest(BaseModel):
    """
    Main AI task request schema.

    This is the primary request schema for the /ai/tasks endpoint.
    Task type must be explicitly declared - never inferred.
    """

    task_type: TaskType = Field(
        ...,
        description="Type of AI task to execute",
    )
    context: ContextSchema = Field(
        ...,
        description="Execution context (jurisdiction, language, etc.)",
    )
    payload: Dict[str, Any] = Field(
        ...,
        description="Task-specific input data",
    )
    options: Optional[OptionsSchema] = Field(
        default=None,
        description="Optional execution flags",
    )

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "task_type": "LEGAL_CHAT",
                    "context": {
                        "jurisdiction": "egypt",
                        "language": "ar",
                    },
                    "payload": {
                        "message": "ما هي شروط العقد الصحيح؟",
                        "conversation_history": [],
                        "files_ids": ["file_123", "file_456"],
                    },
                }
            ]
        },
    }

class TerminologyRequest(BaseModel):
    terminology: str = Field(..., max_length=200)
    use_rag: bool = Field(default=True)
