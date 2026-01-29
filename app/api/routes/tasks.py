"""
AI Task Routes

Main endpoint for executing AI tasks.
This layer handles HTTP concerns only - no business logic.
"""

from fastapi import APIRouter, HTTPException, status

from app.api.schemas.common import MetadataSchema
from app.api.schemas.requests import TaskRequest
from app.api.schemas.responses import TaskErrorResponse, TaskResponse
from app.core.domain.enums import TaskStatus
from app.core.orchestrator import task_orchestrator
from app.shared.errors.exceptions import (
    AIServiceError,
    ContextValidationError,
    PayloadValidationError,
    TaskNotSupportedError,
    ValidationError,
    WorkflowNotFoundError,
)

# Import workflows to trigger registration
import app.core.workflows  # noqa: F401

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
    try:
        # Execute task through the orchestrator pipeline
        result = await task_orchestrator.execute(
            task_type=request.task_type,
            context_data=request.context.model_dump(),
            payload=request.payload,
            options_data=request.options.model_dump() if request.options else None,
        )

        # Map domain Result to API TaskResponse
        return TaskResponse(
            task_id=str(result.task_id),
            task_type=result.task_type,
            status=result.status,
            result=result.data,
            metadata=MetadataSchema(
                execution_time_ms=result.metadata.execution_time_ms,
                model_used=result.metadata.model_used,
                tokens_used=result.metadata.tokens_used,
            ),
        )

    except (TaskNotSupportedError, WorkflowNotFoundError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": e.code,
                "message": e.message,
                "details": e.details,
            },
        )

    except (ContextValidationError, PayloadValidationError, ValidationError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": e.code,
                "message": e.message,
                "details": e.details,
            },
        )

    except AIServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": e.code,
                "message": e.message,
                "details": e.details,
            },
        )
