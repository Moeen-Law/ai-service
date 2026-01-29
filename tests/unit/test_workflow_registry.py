"""
Unit Tests for Workflow Registry

Tests the workflow registration and retrieval mechanism.
"""

import pytest
from typing import Any, Dict, Optional

from app.core.domain.entities import Context, ExecutionOptions, Result
from app.core.domain.enums import TaskType
from app.core.workflows.base import BaseWorkflow
from app.core.workflows.registry import WorkflowRegistry
from app.shared.errors.exceptions import WorkflowNotFoundError


class MockWorkflow(BaseWorkflow):
    """Mock workflow for testing."""

    @property
    def name(self) -> str:
        return "MockWorkflow"

    async def execute(
        self,
        task_id: str,
        context: Context,
        payload: Dict[str, Any],
        options: Optional[ExecutionOptions] = None,
    ) -> Result:
        return Result.success(
            content="Mock result",
            sources=[],
            metadata=None,
        )


class TestWorkflowRegistry:
    """Test suite for WorkflowRegistry."""

    @pytest.fixture
    def registry(self) -> WorkflowRegistry:
        """Create a fresh registry for each test."""
        return WorkflowRegistry()

    @pytest.fixture
    def mock_workflow(self) -> MockWorkflow:
        """Create a mock workflow."""
        return MockWorkflow()

    def test_register_workflow(
        self, registry: WorkflowRegistry, mock_workflow: MockWorkflow
    ) -> None:
        """Test registering a workflow."""
        registry.register(TaskType.LEGAL_CHAT, mock_workflow)

        assert registry.has(TaskType.LEGAL_CHAT)

    def test_get_registered_workflow(
        self, registry: WorkflowRegistry, mock_workflow: MockWorkflow
    ) -> None:
        """Test retrieving a registered workflow."""
        registry.register(TaskType.LEGAL_CHAT, mock_workflow)

        retrieved = registry.get(TaskType.LEGAL_CHAT)

        assert retrieved is mock_workflow

    def test_get_unregistered_workflow_raises(self, registry: WorkflowRegistry) -> None:
        """Test that getting unregistered workflow raises error."""
        with pytest.raises(WorkflowNotFoundError) as exc_info:
            registry.get(TaskType.LEGAL_CHAT)

        assert "LEGAL_CHAT" in str(exc_info.value)

    def test_has_returns_false_for_unregistered(
        self, registry: WorkflowRegistry
    ) -> None:
        """Test has() returns False for unregistered workflow."""
        assert registry.has(TaskType.LEGAL_CHAT) is False

    def test_list_supported_types(
        self, registry: WorkflowRegistry, mock_workflow: MockWorkflow
    ) -> None:
        """Test listing supported task types."""
        registry.register(TaskType.LEGAL_CHAT, mock_workflow)
        registry.register(TaskType.CONTRACT_ANALYSIS, mock_workflow)

        supported = registry.list_supported_types()

        assert TaskType.LEGAL_CHAT in supported
        assert TaskType.CONTRACT_ANALYSIS in supported
        assert len(supported) == 2

    def test_register_same_type_raises_error(self, registry: WorkflowRegistry) -> None:
        """Test that re-registering same type raises ValueError."""
        workflow1 = MockWorkflow()
        workflow2 = MockWorkflow()

        registry.register(TaskType.LEGAL_CHAT, workflow1)

        with pytest.raises(ValueError) as exc_info:
            registry.register(TaskType.LEGAL_CHAT, workflow2)

        assert "already registered" in str(exc_info.value).lower()

    def test_multiple_registrations(
        self, registry: WorkflowRegistry, mock_workflow: MockWorkflow
    ) -> None:
        """Test registering workflows for all task types."""
        for task_type in TaskType:
            registry.register(task_type, mock_workflow)

        assert len(registry.list_supported_types()) == len(TaskType)
        for task_type in TaskType:
            assert registry.has(task_type)
