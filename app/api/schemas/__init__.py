"""
API Request/Response Schemas
"""

from app.api.schemas.common import (
    ContextSchema,
    ErrorDetail,
    ErrorResponse,
    MetadataSchema,
    OptionsSchema,
)
from app.api.schemas.requests import (
    CaseEvaluationPayload,
    ContractAnalysisPayload,
    ContractReframingPayload,
    ConversationMessage,
    DocumentGenerationPayload,
    LegalChatPayload,
    TaskRequest,
)
from app.api.schemas.responses import (
    CaseEvaluationResult,
    ContractAnalysisResult,
    ContractReframingResult,
    DocumentGenerationResult,
    HealthResponse,
    LegalChatResult,
    RiskItem,
    TaskErrorResponse,
    TaskResponse,
)

__all__ = [
    # Common
    "ContextSchema",
    "OptionsSchema",
    "ErrorDetail",
    "ErrorResponse",
    "MetadataSchema",
    # Requests
    "TaskRequest",
    "ConversationMessage",
    "LegalChatPayload",
    "DocumentGenerationPayload",
    "ContractAnalysisPayload",
    "ContractReframingPayload",
    "CaseEvaluationPayload",
    # Responses
    "HealthResponse",
    "TaskResponse",
    "TaskErrorResponse",
    "LegalChatResult",
    "DocumentGenerationResult",
    "ContractAnalysisResult",
    "ContractReframingResult",
    "CaseEvaluationResult",
    "RiskItem",
]
