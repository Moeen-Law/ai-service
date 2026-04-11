"""
LLM Service Interface

Abstract interface defining the contract for LLM service interactions.
This is a port in the hexagonal architecture pattern.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class LLMRequest:
    """
    Request object for LLM service calls.
    """

    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class LLMResponse:
    """
    Response object from LLM service calls.
    """

    content: str
    model: str
    tokens_used: int
    finish_reason: str
    metadata: Optional[Dict[str, Any]] = None


class LLMServiceInterface(ABC):
    """
    Abstract interface for LLM service.

    Defines the contract for generating text responses using
    a language model. Implementations may connect to different
    LLM providers (OpenAI, Azure OpenAI, Anthropic, etc.)
    """

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            request: The LLM request containing prompt and parameters

        Returns:
            LLMResponse with the generated content

        Raises:
            LLMServiceError: If the LLM service call fails
        """
        pass

    @abstractmethod
    async def generate_with_context(
        self,
        prompt: str,
        context: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a response with additional context.

        Convenience method for common pattern of prompt + context.

        Args:
            prompt: The user prompt/question
            context: Additional context to include
            system_prompt: Optional system prompt
            max_tokens: Optional token limit

        Returns:
            LLMResponse with the generated content
        """
        pass

    @abstractmethod
    async def astream(self, prompt: str):
        """
        Stream LLM response tokens asynchronously.

        Args:
            prompt: The formatted prompt to send to the LLM

        Yields:
            String tokens as they are generated
        """
        pass

    @abstractmethod
    async def generate_with_tools(
        self,
        request: LLMRequest,
        tools: Sequence[Any],
        max_iterations: int = 6,
    ) -> LLMResponse:
        """
        Generate a response using model-driven tool calling.

        Args:
            request: The LLM request with prompt/system prompt
            tools: A sequence of tool objects usable by the model
            max_iterations: Maximum tool-calling loop iterations

        Returns:
            LLMResponse with the final assistant content
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the LLM service is available.

        Returns:
            True if service is healthy, False otherwise
        """
        pass
