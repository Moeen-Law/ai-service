"""
AI Task Routes

Main endpoint for executing AI tasks + SSE streaming for LEGAL_CHAT.
This layer handles HTTP concerns only - no business logic.
"""

import json

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.schemas.common import MetadataSchema
from app.api.schemas.requests import TaskRequest
from app.api.schemas.responses import TaskErrorResponse, TaskResponse
from app.core.domain.enums import TaskStatus, TaskType
from app.core.orchestrator import task_orchestrator
from app.core.workflows.legal_chat import LegalChatWorkflow
from app.core.workflows.registry import workflow_registry
from app.infrastructure.logging.logger import get_logger
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

logger = get_logger(__name__)

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
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
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


@router.post(
    "/tasks/stream",
    summary="Stream AI Task (SSE)",
    description=(
        "Server-Sent Events endpoint for LEGAL_CHAT streaming. "
        "Sends token-by-token responses followed by sources and [DONE]."
    ),
)
async def stream_task(request: TaskRequest) -> StreamingResponse:
    """
    Stream an AI task response using Server-Sent Events.

    Currently only LEGAL_CHAT supports streaming.  Other task types
    fall back to a single SSE event containing the full result.

    SSE event types:
    - ``token``: individual LLM token
    - ``sources``: JSON array of cited sources + timing
    - ``generation``: metadata for generated file outputs (files_ids, format)
    - ``done``: signals the end of the stream (data = ``[DONE]``)
    - ``error``: error message
    """
    if request.task_type != TaskType.LEGAL_CHAT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "STREAMING_NOT_SUPPORTED",
                "message": f"Streaming is only supported for LEGAL_CHAT, got {request.task_type.value}",
            },
        )

    workflow: LegalChatWorkflow = workflow_registry.get(TaskType.LEGAL_CHAT)  # type: ignore[assignment]

    def _stringify_exception(exc: BaseException) -> str:
        message = str(exc).strip()
        if message:
            return message
        repr_value = repr(exc).strip()
        if repr_value:
            return repr_value
        return f"{type(exc).__name__} with empty message"

    async def _event_generator():
        try:
            payload = request.payload or {}
            files_ids = payload.get("files_ids")
            if files_ids is None:
                files_ids = payload.get("file_ids")

            logger.info(
                "stream_request_received",
                user_message=payload.get("message", ""),
                files_ids=files_ids,
                task_type=request.task_type.value,
            )

            async for event in workflow.stream(
                question=payload.get("message", ""),
                retrieval_k=payload.get("retrieval_k", 4),
                conversation_history=payload.get("conversation_history", []),
                files_ids=files_ids,
                jurisdiction=request.context.jurisdiction,
                language=request.context.language,
                include_sources=(
                    request.options.include_sources if request.options else True
                ),
            ):
                yield event
        except Exception as exc:
            error_message = _stringify_exception(exc)
            yield (
                f"data: {json.dumps({'type': 'error', 'content': error_message}, ensure_ascii=False)}\n\n"
            )
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
