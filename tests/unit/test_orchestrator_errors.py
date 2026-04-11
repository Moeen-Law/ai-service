"""Unit tests for orchestrator error wrapping and detail propagation."""

from typing import Any, Dict, Optional

import pytest

from app.core.domain.entities import Context, ExecutionOptions, Task
from app.core.domain.enums import Jurisdiction, Language, TaskType
from app.core.orchestrator.orchestrator import TaskOrchestrator
from app.core.workflows.base import BaseWorkflow
from app.shared.errors.exceptions import WorkflowExecutionError


class ExplodingWorkflow(BaseWorkflow):
    @property
    def name(self) -> str:
        return "ExplodingWorkflow"

    async def execute(
        self,
        task_id: str,
        context: Context,
        payload: Dict[str, Any],
        options: Optional[ExecutionOptions] = None,
    ):
        raise RuntimeError()


@pytest.mark.asyncio
async def test_execute_workflow_includes_root_error_details_for_empty_exception() -> (
    None
):
    orchestrator = TaskOrchestrator()
    workflow = ExplodingWorkflow()
    task = Task(
        task_type=TaskType.LEGAL_CHAT,
        context=Context(jurisdiction=Jurisdiction.EGYPT, language=Language.ARABIC),
        payload={"message": "test"},
    )

    with pytest.raises(WorkflowExecutionError) as exc_info:
        await orchestrator._execute_workflow(workflow=workflow, task=task)

    error = exc_info.value
    assert error.message.startswith("Workflow execution failed:")
    assert "RuntimeError" in error.message
    assert error.details.get("workflow") == "ExplodingWorkflow"
    assert error.details.get("root_error_type") == "RuntimeError"
    assert "RuntimeError" in str(error.details.get("root_error_message"))
    assert "RuntimeError" in str(error.details.get("root_error_repr"))
