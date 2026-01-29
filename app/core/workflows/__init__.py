"""
Workflows - Use case implementations for each task type.
"""

from app.core.domain.enums import TaskType
from app.core.workflows.base import BaseWorkflow
from app.core.workflows.case_evaluation import CaseEvaluationWorkflow
from app.core.workflows.contract_analysis import ContractAnalysisWorkflow
from app.core.workflows.contract_reframing import ContractReframingWorkflow
from app.core.workflows.document_generation import DocumentGenerationWorkflow
from app.core.workflows.legal_chat import LegalChatWorkflow
from app.core.workflows.registry import WorkflowRegistry, workflow_registry


def register_all_workflows(registry: WorkflowRegistry | None = None) -> None:
    """
    Register all workflow implementations with the registry.

    Args:
        registry: The registry to use (defaults to global instance)
    """
    reg = registry or workflow_registry

    # Register each workflow for its corresponding task type
    reg.register(TaskType.LEGAL_CHAT, LegalChatWorkflow())
    reg.register(TaskType.DOCUMENT_GENERATION, DocumentGenerationWorkflow())
    reg.register(TaskType.CONTRACT_ANALYSIS, ContractAnalysisWorkflow())
    reg.register(TaskType.CONTRACT_REFRAMING, ContractReframingWorkflow())
    reg.register(TaskType.CASE_EVALUATION, CaseEvaluationWorkflow())


# Auto-register workflows when module is imported
register_all_workflows()


__all__ = [
    "BaseWorkflow",
    "WorkflowRegistry",
    "workflow_registry",
    "register_all_workflows",
    "LegalChatWorkflow",
    "DocumentGenerationWorkflow",
    "ContractAnalysisWorkflow",
    "ContractReframingWorkflow",
    "CaseEvaluationWorkflow",
]
