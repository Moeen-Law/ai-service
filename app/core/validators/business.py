"""
Business Validators

Validates semantic correctness of tasks and payloads.
Enforces business rules that go beyond shape validation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from app.core.domain.entities import Context, Task
from app.core.domain.enums import LegalDomain, TaskType
from app.shared.errors.exceptions import (
    ContextValidationError,
    PayloadValidationError,
    TaskValidationError,
)


class PayloadValidator(ABC):
    """
    Abstract base class for task-specific payload validators.

    Each task type should have a corresponding validator that
    enforces its specific business rules.
    """

    @abstractmethod
    def validate(self, payload: Dict[str, Any], context: Context) -> None:
        """
        Validate the payload for a specific task type.

        Args:
            payload: The task payload to validate
            context: The execution context

        Raises:
            PayloadValidationError: If validation fails
        """
        pass


class LegalChatPayloadValidator(PayloadValidator):
    """Validator for LEGAL_CHAT task payloads."""

    REQUIRED_FIELDS = ["message"]
    MAX_MESSAGE_LENGTH = 10000
    MAX_HISTORY_LENGTH = 50
    MAX_FILES_COUNT = 10
    MAX_FILE_ID_LENGTH = 128

    def validate(self, payload: Dict[str, Any], context: Context) -> None:
        """Validate LEGAL_CHAT payload."""
        # Check required fields
        missing_fields = [f for f in self.REQUIRED_FIELDS if f not in payload]
        if missing_fields:
            raise PayloadValidationError(
                message="Missing required fields in payload",
                task_type="LEGAL_CHAT",
                missing_fields=missing_fields,
            )

        # Validate message
        message = payload.get("message")
        if not message or not isinstance(message, str):
            raise PayloadValidationError(
                message="message must be a non-empty string",
                task_type="LEGAL_CHAT",
            )

        if len(message) > self.MAX_MESSAGE_LENGTH:
            raise PayloadValidationError(
                message=f"message exceeds maximum length of {self.MAX_MESSAGE_LENGTH}",
                task_type="LEGAL_CHAT",
                details={
                    "max_length": self.MAX_MESSAGE_LENGTH,
                    "actual_length": len(message),
                },
            )

        # Validate conversation history if present
        history = payload.get("conversation_history", [])
        if not isinstance(history, list):
            raise PayloadValidationError(
                message="conversation_history must be a list",
                task_type="LEGAL_CHAT",
            )

        if len(history) > self.MAX_HISTORY_LENGTH:
            raise PayloadValidationError(
                message=f"conversation_history exceeds maximum length of {self.MAX_HISTORY_LENGTH}",
                task_type="LEGAL_CHAT",
            )

        # Validate uploaded file IDs if present
        files_ids = payload.get("files_ids")
        if files_ids is None:
            return

        if not isinstance(files_ids, list):
            raise PayloadValidationError(
                message="files_ids must be a list",
                task_type="LEGAL_CHAT",
            )

        if len(files_ids) > self.MAX_FILES_COUNT:
            raise PayloadValidationError(
                message=f"files_ids exceeds maximum length of {self.MAX_FILES_COUNT}",
                task_type="LEGAL_CHAT",
            )

        for file_id in files_ids:
            if not isinstance(file_id, str) or not file_id.strip():
                raise PayloadValidationError(
                    message="files_ids must contain non-empty strings",
                    task_type="LEGAL_CHAT",
                )
            if len(file_id.strip()) > self.MAX_FILE_ID_LENGTH:
                raise PayloadValidationError(
                    message=(
                        "files_ids contains a value exceeding maximum length "
                        f"of {self.MAX_FILE_ID_LENGTH}"
                    ),
                    task_type="LEGAL_CHAT",
                )


class DocumentGenerationPayloadValidator(PayloadValidator):
    """Validator for DOCUMENT_GENERATION task payloads."""

    REQUIRED_FIELDS = ["document_type", "parameters"]
    SUPPORTED_DOCUMENT_TYPES = [
        "contract",
        "agreement",
        "power_of_attorney",
        "memo",
        "legal_notice",
        "complaint",
    ]

    def validate(self, payload: Dict[str, Any], context: Context) -> None:
        """Validate DOCUMENT_GENERATION payload."""
        # Check required fields
        missing_fields = [f for f in self.REQUIRED_FIELDS if f not in payload]
        if missing_fields:
            raise PayloadValidationError(
                message="Missing required fields in payload",
                task_type="DOCUMENT_GENERATION",
                missing_fields=missing_fields,
            )

        # Validate document type
        doc_type = payload.get("document_type")
        if not doc_type or not isinstance(doc_type, str):
            raise PayloadValidationError(
                message="document_type must be a non-empty string",
                task_type="DOCUMENT_GENERATION",
            )

        # Validate parameters is a dict
        parameters = payload.get("parameters")
        if not isinstance(parameters, dict):
            raise PayloadValidationError(
                message="parameters must be an object",
                task_type="DOCUMENT_GENERATION",
            )


class ContractAnalysisPayloadValidator(PayloadValidator):
    """Validator for CONTRACT_ANALYSIS task payloads."""

    MAX_CONTRACT_LENGTH = 100000
    MAX_FILES_COUNT = 10
    MAX_FILE_ID_LENGTH = 128

    def validate(self, payload: Dict[str, Any], context: Context) -> None:
        """Validate CONTRACT_ANALYSIS payload.

        Either `contract_text` (string) must be present, or `files_ids` (list of file ids).
        """
        contract_text = payload.get("contract_text")
        files_ids = payload.get("files_ids")

        if not contract_text and not files_ids:
            raise PayloadValidationError(
                message="Either contract_text or files_ids must be provided",
                task_type="CONTRACT_ANALYSIS",
            )

        if contract_text:
            if not isinstance(contract_text, str) or not contract_text.strip():
                raise PayloadValidationError(
                    message="contract_text must be a non-empty string",
                    task_type="CONTRACT_ANALYSIS",
                )
            if len(contract_text) > self.MAX_CONTRACT_LENGTH:
                raise PayloadValidationError(
                    message=(
                        f"contract_text exceeds maximum length of {self.MAX_CONTRACT_LENGTH}"
                    ),
                    task_type="CONTRACT_ANALYSIS",
                )

        if files_ids is not None:
            if not isinstance(files_ids, list):
                raise PayloadValidationError(
                    message="files_ids must be a list",
                    task_type="CONTRACT_ANALYSIS",
                )
            if len(files_ids) > self.MAX_FILES_COUNT:
                raise PayloadValidationError(
                    message=f"files_ids exceeds maximum length of {self.MAX_FILES_COUNT}",
                    task_type="CONTRACT_ANALYSIS",
                )
            for file_id in files_ids:
                if not isinstance(file_id, str) or not file_id.strip():
                    raise PayloadValidationError(
                        message="files_ids must contain non-empty strings",
                        task_type="CONTRACT_ANALYSIS",
                    )
                if len(file_id.strip()) > self.MAX_FILE_ID_LENGTH:
                    raise PayloadValidationError(
                        message=(
                            "files_ids contains a value exceeding maximum length "
                            f"of {self.MAX_FILE_ID_LENGTH}"
                        ),
                        task_type="CONTRACT_ANALYSIS",
                    )


class ContractReframingPayloadValidator(PayloadValidator):
    """Validator for CONTRACT_REFRAMING task payloads."""

    REQUIRED_FIELDS = ["clause_text", "target_perspective"]
    MAX_CLAUSE_LENGTH = 50000
    SUPPORTED_PERSPECTIVES = ["balanced", "party_a_favorable", "party_b_favorable"]

    def validate(self, payload: Dict[str, Any], context: Context) -> None:
        """Validate CONTRACT_REFRAMING payload."""
        # Check required fields
        missing_fields = [f for f in self.REQUIRED_FIELDS if f not in payload]
        if missing_fields:
            raise PayloadValidationError(
                message="Missing required fields in payload",
                task_type="CONTRACT_REFRAMING",
                missing_fields=missing_fields,
            )

        # Validate clause text
        clause_text = payload.get("clause_text")
        if not clause_text or not isinstance(clause_text, str):
            raise PayloadValidationError(
                message="clause_text must be a non-empty string",
                task_type="CONTRACT_REFRAMING",
            )

        if len(clause_text) > self.MAX_CLAUSE_LENGTH:
            raise PayloadValidationError(
                message=f"clause_text exceeds maximum length of {self.MAX_CLAUSE_LENGTH}",
                task_type="CONTRACT_REFRAMING",
            )

        # Validate target perspective
        target_perspective = payload.get("target_perspective")
        if not target_perspective or not isinstance(target_perspective, str):
            raise PayloadValidationError(
                message="target_perspective must be a non-empty string",
                task_type="CONTRACT_REFRAMING",
            )


class CaseEvaluationPayloadValidator(PayloadValidator):
    """Validator for CASE_EVALUATION task payloads."""

    REQUIRED_FIELDS = ["case_description"]
    MAX_DESCRIPTION_LENGTH = 100000

    def validate(self, payload: Dict[str, Any], context: Context) -> None:
        """Validate CASE_EVALUATION payload."""
        # Check required fields
        missing_fields = [f for f in self.REQUIRED_FIELDS if f not in payload]
        if missing_fields:
            raise PayloadValidationError(
                message="Missing required fields in payload",
                task_type="CASE_EVALUATION",
                missing_fields=missing_fields,
            )

        # Validate case description
        case_description = payload.get("case_description")
        if not case_description or not isinstance(case_description, str):
            raise PayloadValidationError(
                message="case_description must be a non-empty string",
                task_type="CASE_EVALUATION",
            )

        if len(case_description) > self.MAX_DESCRIPTION_LENGTH:
            raise PayloadValidationError(
                message=f"case_description exceeds maximum length of {self.MAX_DESCRIPTION_LENGTH}",
                task_type="CASE_EVALUATION",
            )


class TerminologyPayloadValidator(PayloadValidator):
    def validate(self, payload: Dict[str, Any], context: Context) -> None:
        term = payload.get("terminology")
        if not term or not isinstance(term, str) or not term.strip():
            raise PayloadValidationError(
                message="terminology must be a non-empty string",
                task_type="LEGAL_TERMINOLOGY",
            )
        if len(term) > 200:
            raise PayloadValidationError(
                message="terminology exceeds maximum length of 200",
                task_type="LEGAL_TERMINOLOGY",
            )


class BusinessValidator:
    """
    Central business validation coordinator.

    Validates tasks against business rules including:
    - Context requirements
    - Jurisdiction constraints
    - Task-specific payload rules
    """

    def __init__(self) -> None:
        """Initialize the validator with task-specific validators."""
        self._payload_validators: Dict[TaskType, PayloadValidator] = {
            TaskType.LEGAL_CHAT: LegalChatPayloadValidator(),
            TaskType.DOCUMENT_GENERATION: DocumentGenerationPayloadValidator(),
            TaskType.CONTRACT_ANALYSIS: ContractAnalysisPayloadValidator(),
            TaskType.CONTRACT_REFRAMING: ContractReframingPayloadValidator(),
            TaskType.CASE_EVALUATION: CaseEvaluationPayloadValidator(),
            TaskType.LEGAL_TERMINOLOGY: TerminologyPayloadValidator(),
        }

    def validate(self, task: Task) -> None:
        """
        Validate a task against all business rules.

        Args:
            task: The task to validate

        Raises:
            ContextValidationError: If context validation fails
            PayloadValidationError: If payload validation fails
            TaskValidationError: If task validation fails
        """
        # Validate context rules
        self._validate_context_rules(task)

        # Validate payload for specific task type
        self._validate_payload(task)

    def _validate_context_rules(self, task: Task) -> None:
        """
        Validate context-specific business rules.

        Args:
            task: The task containing the context

        Raises:
            ContextValidationError: If context validation fails
        """
        # Domain is required for certain task types
        tasks_requiring_domain = {
            TaskType.CASE_EVALUATION,
        }

        if task.task_type in tasks_requiring_domain and task.context.domain is None:
            raise ContextValidationError(
                message=f"domain is required for {task.task_type.value} tasks",
                field="domain",
            )

    def _validate_payload(self, task: Task) -> None:
        """
        Validate task payload using the appropriate validator.

        Args:
            task: The task to validate

        Raises:
            PayloadValidationError: If payload validation fails
        """
        validator = self._payload_validators.get(task.task_type)
        if validator is None:
            raise TaskValidationError(
                message=f"No validator registered for task type: {task.task_type}",
            )

        validator.validate(task.payload, task.context)


# Singleton instance
business_validator = BusinessValidator()
