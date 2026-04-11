"""
API Response Schemas

Response schemas for the AI service endpoints.
Provides consistent, validated response structures.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.domain.enums import TaskStatus, TaskType

from .common import ErrorResponse, MetadataSchema


class HealthResponse(BaseModel):
    """
    Health check response schema.
    """

    status: str = Field(
        ...,
        description="Service health status",
        examples=["healthy", "degraded", "unhealthy"],
    )
    service: str = Field(
        ...,
        description="Service name",
    )
    version: str = Field(
        ...,
        description="Service version",
    )


class LegalChatResult(BaseModel):
    """
    Result schema for LEGAL_CHAT task type.
    """

    message: str = Field(
        ...,
        description="AI-generated response message",
    )
    sources: Optional[List[str]] = Field(
        default=None,
        description="Reference sources used in the response",
    )
    files_ids: Optional[List[str]] = Field(
        default=None,
        description="Generated or associated file IDs",
    )
    document_content: Optional[str] = Field(
        default=None,
        description="Generated document content when generation mode is used",
    )
    format: Optional[str] = Field(
        default=None,
        description="Generated document format",
    )


class DocumentGenerationResult(BaseModel):
    """
    Result schema for DOCUMENT_GENERATION task type.
    """

    document_content: str = Field(
        ...,
        description="Generated document content",
    )
    format: str = Field(
        default="markdown",
        description="Document format",
    )


class RiskItem(BaseModel):
    """
    Individual risk item in contract analysis.
    """

    clause: str = Field(..., description="Clause reference")
    risk_level: str = Field(..., description="Risk severity level")
    description: str = Field(..., description="Risk description")


class ContractAnalysisResult(BaseModel):
    """
    Result schema for CONTRACT_ANALYSIS task type.
    """

    risks: List[RiskItem] = Field(
        default_factory=list,
        description="Identified risks in the contract",
    )
    summary: str = Field(
        ...,
        description="Analysis summary",
    )
    recommendations: Optional[List[str]] = Field(
        default=None,
        description="Recommendations for risk mitigation",
    )


class ContractReframingResult(BaseModel):
    """
    Result schema for CONTRACT_REFRAMING task type.
    """

    reframed_clause: str = Field(
        ...,
        description="Reframed contract clause",
    )
    changes_summary: str = Field(
        ...,
        description="Summary of changes made",
    )


class CaseEvaluationResult(BaseModel):
    """
    Result schema for CASE_EVALUATION task type.
    """

    evaluation: str = Field(
        ...,
        description="Case evaluation narrative",
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Case strengths",
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="Case weaknesses",
    )
    recommendation: str = Field(
        ...,
        description="Strategic recommendation",
    )


class TaskResponse(BaseModel):
    """
    Main AI task response schema.

    This is the standard response format for all task executions.
    """

    task_id: str = Field(
        ...,
        description="Unique task identifier",
    )
    task_type: TaskType = Field(
        ...,
        description="Type of task that was executed",
    )
    status: TaskStatus = Field(
        ...,
        description="Task execution status",
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Task-specific result data",
    )
    metadata: MetadataSchema = Field(
        ...,
        description="Execution metadata",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "task_id": "550e8400-e29b-41d4-a716-446655440000",
                    "task_type": "LEGAL_CHAT",
                    "status": "success",
                    "result": {
                        "message": "شروط العقد الصحيح هي...",
                        "sources": ["article_123", "law_456"],
                    },
                    "metadata": {
                        "execution_time_ms": 1234,
                        "model_used": "gpt-4",
                        "tokens_used": 500,
                    },
                }
            ]
        },
    }


class TaskErrorResponse(BaseModel):
    """
    Error response wrapper for task endpoints.
    """

    error: ErrorResponse = Field(
        ...,
        description="Error details",
    )
