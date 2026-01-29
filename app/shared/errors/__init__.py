"""
Custom Exceptions
"""

from app.shared.errors.exceptions import (
    AIServiceError,
    ContextValidationError,
    ExternalServiceError,
    LLMServiceError,
    PayloadValidationError,
    PromptServiceError,
    RAGServiceError,
    TaskError,
    TaskExecutionError,
    TaskNotSupportedError,
    TaskValidationError,
    ValidationError,
    WorkflowError,
    WorkflowExecutionError,
    WorkflowNotFoundError,
)

__all__ = [
    # Base
    "AIServiceError",
    # Validation
    "ValidationError",
    "TaskValidationError",
    "ContextValidationError",
    "PayloadValidationError",
    # Task
    "TaskError",
    "TaskNotSupportedError",
    "TaskExecutionError",
    # Workflow
    "WorkflowError",
    "WorkflowNotFoundError",
    "WorkflowExecutionError",
    # External Services
    "ExternalServiceError",
    "LLMServiceError",
    "RAGServiceError",
    "PromptServiceError",
]
