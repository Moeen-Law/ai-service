"""
Domain Entities

Core business entities representing the fundamental concepts in the AI service.
These are pure domain objects with no infrastructure dependencies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from app.core.domain.enums import (
    Jurisdiction,
    Language,
    LegalDomain,
    TaskStatus,
    TaskType,
)


@dataclass(frozen=True)
class Context:
    """
    Execution context for AI tasks.

    Immutable value object containing all context required for task execution.
    Includes jurisdiction, language, and optional domain specification.
    """

    jurisdiction: Jurisdiction
    language: Language
    domain: Optional[LegalDomain] = None

    def __post_init__(self) -> None:
        """Validate context invariants."""
        if not isinstance(self.jurisdiction, Jurisdiction):
            raise ValueError(f"Invalid jurisdiction: {self.jurisdiction}")
        if not isinstance(self.language, Language):
            raise ValueError(f"Invalid language: {self.language}")


@dataclass(frozen=True)
class ExecutionOptions:
    """
    Optional execution flags for task processing.

    Immutable value object for customizing task execution behavior.
    """

    include_sources: bool = True
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate options invariants."""
        if self.max_tokens is not None and (
            self.max_tokens < 1 or self.max_tokens > 4096
        ):
            raise ValueError(
                f"max_tokens must be between 1 and 4096, got {self.max_tokens}"
            )
        if self.temperature is not None and (
            self.temperature < 0.0 or self.temperature > 2.0
        ):
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )


@dataclass
class Task:
    """
    Represents an AI task to be executed.

    This is the core domain entity that flows through the orchestration pipeline.
    Tasks are created from incoming requests and carry all necessary information
    for workflow execution.
    """

    task_type: TaskType
    context: Context
    payload: Dict[str, Any]
    task_id: UUID = field(default_factory=uuid4)
    options: Optional[ExecutionOptions] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: TaskStatus = TaskStatus.PENDING

    def __post_init__(self) -> None:
        """Validate task invariants."""
        if not isinstance(self.task_type, TaskType):
            raise ValueError(f"Invalid task type: {self.task_type}")
        if not isinstance(self.context, Context):
            raise ValueError("context must be a Context instance")
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a dictionary")

    def mark_success(self) -> None:
        """Mark the task as successfully completed."""
        object.__setattr__(self, "status", TaskStatus.SUCCESS)

    def mark_failed(self) -> None:
        """Mark the task as failed."""
        object.__setattr__(self, "status", TaskStatus.FAILED)


@dataclass(frozen=True)
class ResultMetadata:
    """
    Metadata about task execution.

    Immutable value object containing execution metrics and information.
    """

    execution_time_ms: int
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate metadata invariants."""
        if self.execution_time_ms < 0:
            raise ValueError("execution_time_ms cannot be negative")
        if self.tokens_used is not None and self.tokens_used < 0:
            raise ValueError("tokens_used cannot be negative")


@dataclass(frozen=True)
class Result:
    """
    Result of a task execution.

    Immutable value object containing the outcome of workflow execution.
    This is returned by workflows and ultimately sent back to the client.
    """

    task_id: UUID
    task_type: TaskType
    status: TaskStatus
    data: Dict[str, Any]
    metadata: ResultMetadata
    errors: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Validate result invariants."""
        if not isinstance(self.status, TaskStatus):
            raise ValueError(f"Invalid status: {self.status}")
        if not isinstance(self.data, dict):
            raise ValueError("data must be a dictionary")

    @property
    def is_success(self) -> bool:
        """Check if the result represents a successful execution."""
        return self.status == TaskStatus.SUCCESS

    @property
    def is_failed(self) -> bool:
        """Check if the result represents a failed execution."""
        return self.status == TaskStatus.FAILED

    @classmethod
    def success(
        cls,
        task_id: UUID,
        task_type: TaskType,
        data: Dict[str, Any],
        metadata: ResultMetadata,
    ) -> "Result":
        """Factory method for creating a successful result."""
        return cls(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.SUCCESS,
            data=data,
            metadata=metadata,
        )

    @classmethod
    def failure(
        cls,
        task_id: UUID,
        task_type: TaskType,
        errors: List[str],
        metadata: ResultMetadata,
    ) -> "Result":
        """Factory method for creating a failed result."""
        return cls(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.FAILED,
            data={},
            metadata=metadata,
            errors=errors,
        )
