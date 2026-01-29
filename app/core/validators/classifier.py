"""
Task Classifier

Handles explicit task classification from incoming requests.
Maps request data to domain TaskType - no inference, no guessing.
"""

from typing import Any, Dict, Optional

from app.core.domain.entities import Context, ExecutionOptions, Task
from app.core.domain.enums import Jurisdiction, Language, LegalDomain, TaskType
from app.shared.errors.exceptions import (
    ContextValidationError,
    TaskNotSupportedError,
    TaskValidationError,
)


class TaskClassifier:
    """
    Classifies incoming requests into domain Task entities.

    This class is responsible for:
    1. Validating that the task type is supported
    2. Mapping string values to domain enums
    3. Creating properly typed Task entities

    No inference is performed - task type must be explicitly declared.
    """

    # Supported task types - central registry of capabilities
    SUPPORTED_TASK_TYPES = frozenset(TaskType)

    # Mapping of string values to Jurisdiction enum
    JURISDICTION_MAP = {j.value: j for j in Jurisdiction}

    # Mapping of string values to Language enum
    LANGUAGE_MAP = {lang.value: lang for lang in Language}

    # Mapping of string values to LegalDomain enum
    DOMAIN_MAP = {d.value: d for d in LegalDomain}

    def classify(
        self,
        task_type: TaskType,
        context_data: Dict[str, Any],
        payload: Dict[str, Any],
        options_data: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """
        Classify and create a Task from request data.

        Args:
            task_type: The explicitly declared task type
            context_data: Context dictionary from the request
            payload: Task-specific payload data
            options_data: Optional execution options

        Returns:
            A fully constructed Task domain entity

        Raises:
            TaskNotSupportedError: If task type is not supported
            ContextValidationError: If context is invalid
            TaskValidationError: If task data is invalid
        """
        # Step 1: Validate task type is supported
        self._validate_task_type(task_type)

        # Step 2: Build domain Context
        context = self._build_context(context_data)

        # Step 3: Build execution options if provided
        options = self._build_options(options_data) if options_data else None

        # Step 4: Create and return the Task entity
        return Task(
            task_type=task_type,
            context=context,
            payload=payload,
            options=options,
        )

    def _validate_task_type(self, task_type: TaskType) -> None:
        """
        Validate that the task type is supported.

        Args:
            task_type: The task type to validate

        Raises:
            TaskNotSupportedError: If task type is not supported
        """
        if task_type not in self.SUPPORTED_TASK_TYPES:
            raise TaskNotSupportedError(
                task_type=str(task_type),
                supported_types=[t.value for t in self.SUPPORTED_TASK_TYPES],
            )

    def _build_context(self, context_data: Dict[str, Any]) -> Context:
        """
        Build a Context domain entity from request data.

        Args:
            context_data: Dictionary containing context fields

        Returns:
            A Context domain entity

        Raises:
            ContextValidationError: If context data is invalid
        """
        # Validate and map jurisdiction
        jurisdiction_str = context_data.get("jurisdiction")
        if not jurisdiction_str:
            raise ContextValidationError(
                message="jurisdiction is required",
                field="jurisdiction",
            )

        jurisdiction = self.JURISDICTION_MAP.get(jurisdiction_str.lower())
        if jurisdiction is None:
            raise ContextValidationError(
                message=f"Unsupported jurisdiction: '{jurisdiction_str}'",
                field="jurisdiction",
                details={"supported": list(self.JURISDICTION_MAP.keys())},
            )

        # Validate and map language
        language_str = context_data.get("language")
        if not language_str:
            raise ContextValidationError(
                message="language is required",
                field="language",
            )

        language = self.LANGUAGE_MAP.get(language_str.lower())
        if language is None:
            raise ContextValidationError(
                message=f"Unsupported language: '{language_str}'",
                field="language",
                details={"supported": list(self.LANGUAGE_MAP.keys())},
            )

        # Map domain (optional)
        domain = None
        domain_str = context_data.get("domain")
        if domain_str:
            domain = self.DOMAIN_MAP.get(domain_str.lower())
            if domain is None:
                raise ContextValidationError(
                    message=f"Unsupported domain: '{domain_str}'",
                    field="domain",
                    details={"supported": list(self.DOMAIN_MAP.keys())},
                )

        return Context(
            jurisdiction=jurisdiction,
            language=language,
            domain=domain,
        )

    def _build_options(self, options_data: Dict[str, Any]) -> ExecutionOptions:
        """
        Build ExecutionOptions from request data.

        Args:
            options_data: Dictionary containing options fields

        Returns:
            An ExecutionOptions domain entity

        Raises:
            TaskValidationError: If options data is invalid
        """
        try:
            return ExecutionOptions(
                include_sources=options_data.get("include_sources", True),
                max_tokens=options_data.get("max_tokens"),
                temperature=options_data.get("temperature"),
            )
        except ValueError as e:
            raise TaskValidationError(
                message=str(e),
                field="options",
            )


# Singleton instance for use across the application
task_classifier = TaskClassifier()
