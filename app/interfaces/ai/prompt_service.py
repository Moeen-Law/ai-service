"""
Prompt Service Interface

Abstract interface defining the contract for prompt management.
This is a port in the hexagonal architecture pattern.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PromptTemplate:
    """
    Represents a prompt template.
    """

    id: str
    name: str
    template: str
    variables: List[str]
    description: Optional[str] = None
    version: str = "1.0"


@dataclass(frozen=True)
class AssembledPrompt:
    """
    Result of assembling a prompt from template and variables.
    """

    prompt: str
    system_prompt: Optional[str]
    template_id: str
    template_version: str
    metadata: Optional[Dict[str, Any]] = None


class PromptServiceInterface(ABC):
    """
    Abstract interface for Prompt service.

    Defines the contract for managing and assembling prompts
    for different task types and contexts.
    """

    @abstractmethod
    async def get_template(
        self,
        task_type: str,
        jurisdiction: str,
        language: str,
    ) -> PromptTemplate:
        """
        Get the appropriate prompt template for a task.

        Args:
            task_type: The type of task
            jurisdiction: Legal jurisdiction
            language: Response language

        Returns:
            PromptTemplate for the task

        Raises:
            PromptServiceError: If template retrieval fails
        """
        pass

    @abstractmethod
    async def assemble_prompt(
        self,
        template: PromptTemplate,
        variables: Dict[str, Any],
        context_documents: Optional[List[str]] = None,
    ) -> AssembledPrompt:
        """
        Assemble a complete prompt from template and variables.

        Args:
            template: The prompt template to use
            variables: Variables to substitute in the template
            context_documents: Optional list of context documents

        Returns:
            AssembledPrompt ready for LLM

        Raises:
            PromptServiceError: If assembly fails
        """
        pass

    @abstractmethod
    async def get_system_prompt(
        self,
        task_type: str,
        jurisdiction: str,
        language: str,
    ) -> str:
        """
        Get the system prompt for a task type.

        Args:
            task_type: The type of task
            jurisdiction: Legal jurisdiction
            language: Response language

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the Prompt service is available.

        Returns:
            True if service is healthy, False otherwise
        """
        pass
