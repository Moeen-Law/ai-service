"""
Common API Schemas

Shared schema components used across multiple endpoints.
These schemas handle shape validation only - no business logic.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ContextSchema(BaseModel):
    """
    Execution context for AI tasks.

    Provides the necessary context for task execution including
    jurisdiction, language, and optional domain specification.
    """

    jurisdiction: str = Field(
        ...,
        description="Legal jurisdiction (e.g., 'egypt', 'uae')",
        min_length=1,
        max_length=50,
        examples=["egypt", "uae", "saudi_arabia"],
    )
    language: str = Field(
        ...,
        description="Response language code",
        min_length=2,
        max_length=10,
        examples=["ar", "en"],
    )
    domain: Optional[str] = Field(
        default=None,
        description="Legal domain (e.g., 'civil', 'criminal', 'commercial')",
        max_length=50,
        examples=["civil", "criminal", "commercial"],
    )

    model_config = {
        "extra": "forbid",
    }


class OptionsSchema(BaseModel):
    """
    Optional execution flags for task processing.

    Allows customization of task execution behavior.
    """

    include_sources: bool = Field(
        default=True,
        description="Include source references in response",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens in response",
        ge=1,
        le=4096,
    )
    temperature: Optional[float] = Field(
        default=None,
        description="LLM temperature setting",
        ge=0.0,
        le=2.0,
    )

    model_config = {
        "extra": "forbid",
    }


class ErrorDetail(BaseModel):
    """
    Individual error detail for validation errors.
    """

    field: str = Field(..., description="Field that caused the error")
    message: str = Field(..., description="Error message")


class ErrorResponse(BaseModel):
    """
    Standard error response format.

    Used for all error responses to ensure consistency.
    """

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[List[ErrorDetail]] = Field(
        default=None,
        description="Detailed error information",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracing",
    )


class MetadataSchema(BaseModel):
    """
    Response metadata for task execution.
    """

    execution_time_ms: int = Field(
        ...,
        description="Execution time in milliseconds",
        ge=0,
    )
    model_used: Optional[str] = Field(
        default=None,
        description="AI model used for generation",
    )
    tokens_used: Optional[int] = Field(
        default=None,
        description="Total tokens consumed",
        ge=0,
    )
