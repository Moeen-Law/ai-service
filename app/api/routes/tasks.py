"""
AI Task Routes

Main endpoint for executing AI tasks.
This layer handles HTTP concerns only - no business logic.
"""

import time
import uuid
from typing import Any, Dict

from fastapi import APIRouter, status

from app.api.schemas.common import MetadataSchema
from app.api.schemas.requests import TaskRequest
from app.api.schemas.responses import TaskErrorResponse, TaskResponse
from app.core.domain.enums import TaskStatus, TaskType

router = APIRouter(prefix="/ai", tags=["AI Tasks"])


@router.post(
    "/tasks",
    response_model=TaskResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute AI Task",
    description="Main endpoint for executing AI tasks. Task type must be explicitly declared.",
    responses={
        400: {
            "model": TaskErrorResponse,
            "description": "Bad Request - Invalid task type or context",
        },
        422: {
            "model": TaskErrorResponse,
            "description": "Validation Error - Invalid request schema",
        },
        500: {
            "model": TaskErrorResponse,
            "description": "Internal Server Error",
        },
    },
)
async def execute_task(request: TaskRequest) -> TaskResponse:
    """
    Execute an AI task.

    This endpoint receives structured AI task requests, validates them,
    and routes them to the appropriate workflow for execution.

    The task_type must be explicitly declared - it is never inferred.

    Args:
        request: The task request containing task_type, context, payload, and options.

    Returns:
        TaskResponse with the execution result and metadata.

    Raises:
        HTTPException: If task validation fails or execution errors occur.
    """
    start_time = time.time()
    task_id = str(uuid.uuid4())

    # TODO: This is a placeholder implementation
    # The actual implementation will:
    # 1. Classify task type (already explicit in request)
    # 2. Validate business rules (via validators)
    # 3. Select and execute workflow (via orchestrator)
    # 4. Return structured response

    # For now, return a stub response to prove the API works
    execution_time_ms = int((time.time() - start_time) * 1000)

    # Placeholder result based on task type
    stub_results: Dict[TaskType, Dict[str, Any]] = {
        TaskType.LEGAL_CHAT: {
            "message": "هذه استجابة تجريبية. سيتم تنفيذ المنطق الفعلي لاحقاً.",
            "sources": ["stub_source_1", "stub_source_2"],
        },
        TaskType.DOCUMENT_GENERATION: {
            "document_content": "# مستند تجريبي\n\nمحتوى المستند سيتم توليده لاحقاً.",
            "format": "markdown",
        },
        TaskType.CONTRACT_ANALYSIS: {
            "risks": [],
            "summary": "تحليل العقد سيتم تنفيذه لاحقاً.",
            "recommendations": [],
        },
        TaskType.CONTRACT_REFRAMING: {
            "reframed_clause": "البند المعاد صياغته سيظهر هنا.",
            "changes_summary": "ملخص التغييرات سيتم توليده لاحقاً.",
        },
        TaskType.CASE_EVALUATION: {
            "evaluation": "تقييم القضية سيتم تنفيذه لاحقاً.",
            "strengths": [],
            "weaknesses": [],
            "recommendation": "التوصية ستظهر هنا.",
        },
    }

    return TaskResponse(
        task_id=task_id,
        task_type=request.task_type,
        status=TaskStatus.SUCCESS,
        result=stub_results.get(request.task_type, {}),
        metadata=MetadataSchema(
            execution_time_ms=execution_time_ms,
            model_used="stub",
            tokens_used=0,
        ),
    )
