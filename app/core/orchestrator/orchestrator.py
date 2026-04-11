"""
Task Orchestrator

Central execution controller that owns the task lifecycle.
Coordinates classification, validation, workflow selection, and execution.
"""

import time
from typing import Any, Dict, Optional

from app.core.domain.entities import Result, ResultMetadata, Task
from app.core.domain.enums import TaskStatus, TaskType
from app.core.validators.business import BusinessValidator, business_validator
from app.core.validators.classifier import TaskClassifier, task_classifier
from app.core.workflows.base import BaseWorkflow
from app.core.workflows.registry import WorkflowRegistry, workflow_registry
from app.infrastructure.logging.logger import get_logger
from app.shared.errors.exceptions import (
    AIServiceError,
    TaskExecutionError,
    WorkflowExecutionError,
)

logger = get_logger(__name__)


def _stringify_exception(exc: BaseException) -> str:
    """Return a useful human-readable message for exceptions with empty str()."""
    message = str(exc).strip()
    if message:
        return message

    repr_value = repr(exc).strip()
    if repr_value:
        return repr_value

    return f"{type(exc).__name__} with empty message"


class TaskOrchestrator:
    """
    Central orchestrator for AI task execution.

    Owns the complete task lifecycle:
    1. Task classification (explicit, no inference)
    2. Business validation
    3. Workflow selection
    4. Execution with timing
    5. Result packaging

    This class is the single entry point for task processing.
    """

    def __init__(
        self,
        classifier: Optional[TaskClassifier] = None,
        validator: Optional[BusinessValidator] = None,
        registry: Optional[WorkflowRegistry] = None,
    ) -> None:
        """
        Initialize the orchestrator with its dependencies.

        Args:
            classifier: Task classifier (defaults to global instance)
            validator: Business validator (defaults to global instance)
            registry: Workflow registry (defaults to global instance)
        """
        self._classifier = classifier or task_classifier
        self._validator = validator or business_validator
        self._registry = registry or workflow_registry

    async def execute(
        self,
        task_type: TaskType,
        context_data: Dict[str, Any],
        payload: Dict[str, Any],
        options_data: Optional[Dict[str, Any]] = None,
    ) -> Result:
        """
        Execute an AI task through the complete pipeline.

        This is the main entry point for task execution. It:
        1. Classifies the request into a Task domain entity
        2. Validates business rules
        3. Selects the appropriate workflow
        4. Executes the workflow
        5. Returns a structured Result

        Args:
            task_type: The explicitly declared task type
            context_data: Context dictionary from the request
            payload: Task-specific payload data
            options_data: Optional execution options

        Returns:
            Result object containing the execution outcome

        Raises:
            AIServiceError: If any step in the pipeline fails
        """
        start_time = time.time()
        task: Optional[Task] = None

        logger.info(
            "Task execution started",
            task_type=task_type.value,
            jurisdiction=context_data.get("jurisdiction"),
        )

        try:
            # Step 1: Classify request into Task domain entity
            task = self._classifier.classify(
                task_type=task_type,
                context_data=context_data,
                payload=payload,
                options_data=options_data,
            )

            logger.debug(
                "Task classified",
                task_id=str(task.task_id),
                task_type=task.task_type.value,
            )

            # Step 2: Validate business rules
            self._validator.validate(task)

            logger.debug("Business validation passed", task_id=str(task.task_id))

            # Step 3: Select workflow
            workflow = self._registry.get(task.task_type)

            logger.debug(
                "Workflow selected",
                task_id=str(task.task_id),
                workflow=type(workflow).__name__,
            )

            # Step 4: Execute workflow
            result = await self._execute_workflow(
                workflow=workflow,
                task=task,
            )

            # Step 5: Mark task as successful
            task.mark_success()

            execution_time = time.time() - start_time
            logger.info(
                "Task execution completed",
                task_id=str(task.task_id),
                task_type=task.task_type.value,
                execution_time_ms=round(execution_time * 1000, 2),
            )

            return result

        except AIServiceError as e:
            # Re-raise domain exceptions as-is
            if task:
                task.mark_failed()

            logger.warning(
                "Task execution failed with domain error",
                task_id=str(task.task_id) if task else None,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

        except Exception as e:
            # Wrap unexpected exceptions
            if task:
                task.mark_failed()

            error_message = _stringify_exception(e)

            logger.exception(
                "Task execution failed with unexpected error",
                task_id=str(task.task_id) if task else None,
                error_type=type(e).__name__,
                error_message=error_message,
            )
            raise TaskExecutionError(
                message=f"Unexpected error during task execution: {error_message}",
                task_id=str(task.task_id) if task else None,
            ) from e

    async def _execute_workflow(
        self,
        workflow: BaseWorkflow,
        task: Task,
    ) -> Result:
        """
        Execute a workflow with timing and error handling.

        Args:
            workflow: The workflow to execute
            task: The task to process

        Returns:
            Result from the workflow

        Raises:
            WorkflowExecutionError: If workflow execution fails
        """
        start_time = time.time()

        try:
            # Execute pre-hook
            await workflow.pre_execute(task.context, task.payload)

            # Execute main workflow logic
            result = await workflow.execute(
                task_id=str(task.task_id),
                context=task.context,
                payload=task.payload,
                options=task.options,
            )

            # Execute post-hook
            result = await workflow.post_execute(result)

            return result

        except AIServiceError:
            # Re-raise domain exceptions
            raise

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            cause = e.__cause__ or e.__context__
            error_message = _stringify_exception(e)
            details = {
                "execution_time_ms": execution_time_ms,
                "root_error_type": type(e).__name__,
                "root_error_message": error_message,
                "root_error_repr": repr(e),
            }
            if cause is not None:
                details["cause_error_type"] = type(cause).__name__
                details["cause_error_message"] = _stringify_exception(cause)

            raise WorkflowExecutionError(
                message=f"Workflow execution failed: {error_message}",
                workflow_name=workflow.name,
                details=details,
            ) from e


# Global orchestrator instance
task_orchestrator = TaskOrchestrator()
