# Workflow Implementation Guide

## Overview

Workflows represent **use cases** in the Clean Architecture pattern. Each task type has a corresponding workflow that orchestrates the execution.

## Workflow Responsibilities

A workflow:

1. Validates task-specific requirements
2. Coordinates calls to external services (RAG, LLM, Prompt)
3. Transforms and structures the response
4. Handles task-specific error scenarios

A workflow does NOT:

- Contain AI/LLM implementation
- Make direct HTTP calls (uses service interfaces)
- Handle API concerns (validation, serialization)
- Manage application state

## Base Workflow Interface

All workflows must implement:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseWorkflow(ABC):
    @abstractmethod
    async def execute(
        self,
        context: Context,
        payload: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the workflow.
        
        Args:
            context: Execution context (jurisdiction, language, etc.)
            payload: Task-specific input data
            options: Optional execution flags
            
        Returns:
            Task-specific result dictionary
        """
        pass
```

## Implementation Pattern

### 1. Legal Chat Workflow

**Purpose**: Handle conversational legal queries

**Steps**:

1. Validate conversation history format
2. Call RAG service to retrieve relevant legal documents
3. Call Prompt service to assemble context + query + history
4. Call LLM service to generate response
5. Structure response with sources

**External Dependencies**:

- RAGService (for retrieval)
- PromptService (for prompt assembly)
- LLMService (for generation)

### 2. Document Generation Workflow

**Purpose**: Generate legal documents from templates

**Steps**:

1. Validate required parameters for document type
2. Call RAG service to retrieve document template
3. Call Prompt service to assemble generation instructions
4. Call LLM service to populate template
5. Format and return generated document

**External Dependencies**:

- RAGService (for templates)
- PromptService (for instructions)
- LLMService (for generation)

### 3. Contract Analysis Workflow

**Purpose**: Analyze contract content for risks and issues

**Steps**:

1. Validate contract text
2. Call RAG service to retrieve analysis guidelines
3. Call Prompt service to structure analysis instructions
4. Call LLM service to perform analysis
5. Structure findings by risk level

**External Dependencies**:

- RAGService (for guidelines)
- PromptService (for instructions)
- LLMService (for analysis)

## Workflow Registration

Workflows are registered in the `WorkflowRegistry`:

```python
from core.workflows.legal_chat import LegalChatWorkflow
from core.workflows.document_generation import DocumentGenerationWorkflow

registry = WorkflowRegistry()
registry.register(TaskType.LEGAL_CHAT, LegalChatWorkflow())
registry.register(TaskType.DOCUMENT_GENERATION, DocumentGenerationWorkflow())
```

## Error Handling

Workflows should raise specific exceptions:

- `TaskValidationError` - Invalid payload for task type
- `ContextValidationError` - Missing required context
- `ExternalServiceError` - RAG/LLM/Prompt service failure
- `WorkflowExecutionError` - Unexpected workflow error

The orchestrator and API layer handle these exceptions.

## Testing Workflows

Each workflow should have:

1. **Unit tests** with mocked dependencies
2. **Integration tests** with stub implementations

Example:

```python
async def test_legal_chat_workflow():
    # Arrange
    rag_service = MockRAGService()
    llm_service = MockLLMService()
    workflow = LegalChatWorkflow(rag_service, llm_service)
    
    # Act
    result = await workflow.execute(context, payload)
    
    # Assert
    assert result["message"] is not None
    assert "sources" in result
```

## Adding a New Workflow

1. Create workflow file in `app/core/workflows/`
2. Extend `BaseWorkflow`
3. Implement `execute()` method
4. Register in `WorkflowRegistry`
5. Add to `TaskType` enum
6. Document payload schema
7. Write tests

## Interface-Driven Development

When implementing a workflow:

1. **Define what you need** from external services (interface)
2. **Implement workflow logic** using those interfaces
3. **Stub implementations** for testing
4. **Teammate provides real implementations** later

This allows parallel development without blocking.
