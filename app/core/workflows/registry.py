"""
Workflow Registry

Central registry that maps TaskType to Workflow implementations.
Provides deterministic workflow selection - no inference.
"""

from typing import Dict, Optional, Type

from app.core.domain.enums import TaskType
from app.core.workflows.base import BaseWorkflow
from app.shared.errors.exceptions import WorkflowNotFoundError


class WorkflowRegistry:
    """
    Registry for workflow implementations.

    Maps each TaskType to its corresponding workflow implementation.
    Provides the single source of truth for supported capabilities.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._workflows: Dict[TaskType, BaseWorkflow] = {}

    def register(self, task_type: TaskType, workflow: BaseWorkflow) -> None:
        """
        Register a workflow for a task type.

        Args:
            task_type: The task type this workflow handles
            workflow: The workflow implementation

        Raises:
            ValueError: If a workflow is already registered for this task type
        """
        if task_type in self._workflows:
            raise ValueError(
                f"Workflow already registered for task type: {task_type.value}. "
                f"Existing: {self._workflows[task_type].name}"
            )
        self._workflows[task_type] = workflow

    def get(self, task_type: TaskType) -> BaseWorkflow:
        """
        Get the workflow for a task type.

        Args:
            task_type: The task type to get the workflow for

        Returns:
            The registered workflow

        Raises:
            WorkflowNotFoundError: If no workflow is registered for this task type
        """
        workflow = self._workflows.get(task_type)
        if workflow is None:
            raise WorkflowNotFoundError(task_type=task_type.value)
        return workflow

    def has(self, task_type: TaskType) -> bool:
        """
        Check if a workflow is registered for a task type.

        Args:
            task_type: The task type to check

        Returns:
            True if a workflow is registered, False otherwise
        """
        return task_type in self._workflows

    def list_supported_types(self) -> list[TaskType]:
        """
        List all supported task types.

        Returns:
            List of task types that have registered workflows
        """
        return list(self._workflows.keys())

    def unregister(self, task_type: TaskType) -> None:
        """
        Unregister a workflow for a task type.

        Primarily used for testing purposes.

        Args:
            task_type: The task type to unregister
        """
        self._workflows.pop(task_type, None)

    def clear(self) -> None:
        """
        Clear all registered workflows.

        Primarily used for testing purposes.
        """
        self._workflows.clear()


# Global registry instance
workflow_registry = WorkflowRegistry()
