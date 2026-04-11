"""
Custom Domain Exceptions

Centralized exception hierarchy for the AI service.
All exceptions are domain-specific and carry meaningful error information.
"""

from typing import Any, Dict, List, Optional


class AIServiceError(Exception):
    """
    Base exception for all AI service errors.

    All custom exceptions should inherit from this class.
    """

    def __init__(
        self,
        message: str,
        code: str = "AI_SERVICE_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(AIServiceError):
    """Base exception for validation errors."""

    def __init__(
        self,
        message: str,
        code: str = "VALIDATION_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class TaskValidationError(ValidationError):
    """Exception raised when task validation fails."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        error_details = details or {}
        if field:
            error_details["field"] = field
        super().__init__(message, "TASK_VALIDATION_ERROR", error_details)


class ContextValidationError(ValidationError):
    """Exception raised when context validation fails."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        error_details = details or {}
        if field:
            error_details["field"] = field
        super().__init__(message, "CONTEXT_VALIDATION_ERROR", error_details)


class PayloadValidationError(ValidationError):
    """Exception raised when payload validation fails."""

    def __init__(
        self,
        message: str,
        task_type: Optional[str] = None,
        missing_fields: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        error_details = details or {}
        if task_type:
            error_details["task_type"] = task_type
        if missing_fields:
            error_details["missing_fields"] = missing_fields
        super().__init__(message, "PAYLOAD_VALIDATION_ERROR", error_details)


# =============================================================================
# Task Errors
# =============================================================================


class TaskError(AIServiceError):
    """Base exception for task-related errors."""

    def __init__(
        self,
        message: str,
        code: str = "TASK_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class TaskNotSupportedError(TaskError):
    """Exception raised when a task type is not supported."""

    def __init__(
        self,
        task_type: str,
        supported_types: Optional[List[str]] = None,
    ) -> None:
        message = f"Task type '{task_type}' is not supported"
        details: Dict[str, Any] = {"task_type": task_type}
        if supported_types:
            details["supported_types"] = supported_types
        super().__init__(message, "TASK_NOT_SUPPORTED", details)


class TaskExecutionError(TaskError):
    """Exception raised when task execution fails."""

    def __init__(
        self,
        message: str,
        task_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        error_details = details or {}
        if task_id:
            error_details["task_id"] = task_id
        super().__init__(message, "TASK_EXECUTION_ERROR", error_details)


# =============================================================================
# Workflow Errors
# =============================================================================


class WorkflowError(AIServiceError):
    """Base exception for workflow-related errors."""

    def __init__(
        self,
        message: str,
        code: str = "WORKFLOW_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)


class WorkflowNotFoundError(WorkflowError):
    """Exception raised when a workflow is not found for a task type."""

    def __init__(self, task_type: str) -> None:
        message = f"No workflow registered for task type '{task_type}'"
        super().__init__(message, "WORKFLOW_NOT_FOUND", {"task_type": task_type})


class WorkflowExecutionError(WorkflowError):
    """Exception raised when workflow execution fails."""

    def __init__(
        self,
        message: str,
        workflow_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        error_details = details or {}
        if workflow_name:
            error_details["workflow"] = workflow_name
        super().__init__(message, "WORKFLOW_EXECUTION_ERROR", error_details)


# =============================================================================
# External Service Errors
# =============================================================================


class ExternalServiceError(AIServiceError):
    """Base exception for external service errors."""

    def __init__(
        self,
        message: str,
        service_name: str,
        code: str = "EXTERNAL_SERVICE_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        error_details = details or {}
        error_details["service"] = service_name
        super().__init__(message, code, error_details)


class LLMServiceError(ExternalServiceError):
    """Exception raised when LLM service fails."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, "llm_service", "LLM_SERVICE_ERROR", details)


class RAGServiceError(ExternalServiceError):
    """Exception raised when RAG service fails."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, "rag_service", "RAG_SERVICE_ERROR", details)


class PromptServiceError(ExternalServiceError):
    """Exception raised when Prompt service fails."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, "prompt_service", "PROMPT_SERVICE_ERROR", details)


class FilesServiceError(ExternalServiceError):
    """Exception raised when Files service operations fail."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, "files_service", "FILES_SERVICE_ERROR", details)


class FileExtractionError(AIServiceError):
    """Exception raised when uploaded file content extraction fails."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, "FILE_EXTRACTION_ERROR", details)
