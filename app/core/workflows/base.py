"""
Workflow Interface

Defines the shared execution contract that all workflows must implement.
This is the core abstraction that enables workflow replaceability.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from app.core.domain.entities import Context, ExecutionOptions, Result


class BaseWorkflow(ABC):
    """
    Abstract base class for all workflows.

    All workflows must implement this interface to be registered
    with the workflow registry and executed by the orchestrator.

    A workflow encapsulates the logic for a specific task type,
    coordinating calls to external services (RAG, LLM, Prompt)
    and transforming results.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the workflow name for logging and debugging.

        Returns:
            Human-readable workflow name
        """
        pass

    @abstractmethod
    async def execute(
        self,
        task_id: str,
        context: Context,
        payload: Dict[str, Any],
        options: Optional[ExecutionOptions] = None,
    ) -> Result:
        """
        Execute the workflow.

        This method contains the core logic for processing a task.
        It should coordinate calls to external services and return
        a structured result.

        Args:
            task_id: Unique identifier for the task
            context: Execution context (jurisdiction, language, etc.)
            payload: Task-specific input data
            options: Optional execution flags

        Returns:
            Result object containing the execution outcome

        Raises:
            WorkflowExecutionError: If workflow execution fails
        """
        pass

    async def pre_execute(
        self,
        context: Context,
        payload: Dict[str, Any],
    ) -> None:
        """
        Optional hook executed before the main workflow logic.

        Override this method to perform any setup or pre-processing.

        Args:
            context: Execution context
            payload: Task-specific input data
        """
        pass

    async def post_execute(
        self,
        result: Result,
    ) -> Result:
        """
        Optional hook executed after the main workflow logic.

        Override this method to perform any cleanup or post-processing.

        Args:
            result: The result from the main execution

        Returns:
            Potentially modified result
        """
        return result
