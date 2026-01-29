# AI Service - Complete Project Documentation

**Version:** 0.1.0  
**Date:** January 24, 2026  
**Project:** Legal Tech AI Microservice  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Architecture & Design Principles](#architecture--design-principles)
4. [Folder Structure](#folder-structure)
5. [Layer Details](#layer-details)
6. [Domain Models](#domain-models)
7. [API Specification](#api-specification)
8. [Workflow System](#workflow-system)
9. [Error Handling](#error-handling)
10. [Configuration](#configuration)
11. [Setup & Installation](#setup--installation)
12. [Development Guide](#development-guide)
13. [Testing Strategy](#testing-strategy)
14. [Deployment](#deployment)
15. [Code Examples](#code-examples)

---

## Executive Summary

This AI Service is a **backend microservice** built with Python and FastAPI, designed to orchestrate AI tasks for a legal-tech application. It acts as a **platform service**, receiving structured requests from other microservices, routing them to appropriate workflows, and returning structured responses.

### Key Characteristics

- **Architecture:** Clean Architecture with explicit layer separation
- **Framework:** FastAPI (Python 3.11+)
- **Design Pattern:** Task-based routing with workflow orchestration
- **Responsibility:** API layer, validation, routing, orchestration
- **NOT Responsible For:** RAG implementation, LLM execution, prompt engineering

### Core Principle

> "The AI Service coordinates AI workflows but delegates the actual AI operations to specialized external services."

---

## Project Overview

### Purpose

This service serves as the **central orchestration layer** for AI-powered legal operations, including:

- Legal chat/consultation
- Document generation
- Contract analysis
- Contract reframing
- Case evaluation

### System Context

```txt
┌─────────────────┐
│ Other Services  │ (User Service, Document Service, etc.)
│ (Microservices) │
└────────┬────────┘
         │ HTTP POST /v1/ai/tasks
         ▼
┌─────────────────────────────────────────┐
│         AI Service (THIS)                │
│  ┌────────────────────────────────┐     │
│  │  API Layer (FastAPI)           │     │
│  └────────────┬───────────────────┘     │
│               ▼                          │
│  ┌────────────────────────────────┐     │
│  │  Task Orchestrator             │     │
│  └────────────┬───────────────────┘     │
│               ▼                          │
│  ┌────────────────────────────────┐     │
│  │  Workflow Selection            │     │
│  └────────────┬───────────────────┘     │
└───────────────┼──────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│   External AI Services (Teammate's Code)  │
│   - RAG Service (Vector DB, Retrieval)   │
│   - LLM Service (GPT-4, Claude, etc.)    │
│   - Prompt Service (Prompt Engineering)  │
└──────────────────────────────────────────┘
```

### Team Responsibilities

#### **AI Service (This Project)**

✅ API endpoints and routing  
✅ Request/response validation  
✅ Task classification  
✅ Workflow orchestration  
✅ Error handling  
✅ Logging and tracing  

#### **External Services (Other Teammates)**

❌ RAG implementation  
❌ Vector database operations  
❌ LLM provider integration  
❌ Prompt engineering  
❌ Model execution  

---

## Architecture & Design Principles

### Clean Architecture

The project follows **Clean Architecture** (Uncle Bob), with dependencies flowing inward:

```txt
┌─────────────────────────────────────────────────┐
│  API Layer (Presentation)                       │
│  - FastAPI routes                               │
│  - Pydantic schemas                             │
│  - HTTP concerns                                │
└──────────────────┬──────────────────────────────┘
                   │ depends on ▼
┌─────────────────────────────────────────────────┐
│  Core Layer (Application/Business Logic)        │
│  - Task Orchestrator                            │
│  - Workflows (Use Cases)                        │
│  - Domain Models                                │
│  - Business Validators                          │
└──────────────────┬──────────────────────────────┘
                   │ depends on ▼
┌─────────────────────────────────────────────────┐
│  Interfaces Layer (Abstract Ports)              │
│  - RAGService (abstract)                        │
│  - LLMService (abstract)                        │
│  - PromptService (abstract)                     │
└──────────────────▲──────────────────────────────┘
                   │ implemented by
┌─────────────────────────────────────────────────┐
│  Infrastructure Layer (Frameworks & Drivers)    │
│  - Configuration                                │
│  - Logging                                      │
│  - Concrete Adapters                            │
└─────────────────────────────────────────────────┘
```

### Design Principles

1. **Dependency Inversion**
   - Core logic depends on abstractions, not implementations
   - Easy to swap external services without core changes

2. **Single Responsibility**
   - Each layer has one reason to change
   - Each workflow handles one task type

3. **Explicit Over Implicit**
   - Task types declared in request (no guessing)
   - Clear interfaces for external dependencies

4. **Stateless Design**
   - No session management
   - All context passed in requests
   - Horizontally scalable

5. **Interface-Driven Development**
   - Define what you need (interface)
   - Implement business logic using interfaces
   - Teammate provides concrete implementation later

---

## Folder Structure

### Complete Tree

```txt
graduation-project-ai-service/
├── app/                                # Application Code
│   ├── __init__.py
│   │
│   ├── api/                           # API Layer
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI app
│   │   ├── dependencies.py            # DI container
│   │   ├── middleware.py              # CORS, logging, tracing
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── health.py              # Health check
│   │   │   └── tasks.py               # POST /v1/ai/tasks
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── requests.py            # Input models
│   │       ├── responses.py           # Output models
│   │       └── common.py              # Shared models
│   │
│   ├── core/                          # Core Business Logic
│   │   ├── __init__.py
│   │   ├── domain/
│   │   │   ├── __init__.py
│   │   │   ├── task_types.py          # TaskType enum
│   │   │   ├── context.py             # Context model
│   │   │   ├── task.py                # Task entity
│   │   │   └── result.py              # Result entity
│   │   ├── orchestrator/
│   │   │   ├── __init__.py
│   │   │   ├── task_orchestrator.py   # Main orchestrator
│   │   │   └── workflow_registry.py   # TaskType → Workflow
│   │   ├── workflows/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # Abstract workflow
│   │   │   ├── legal_chat.py
│   │   │   ├── document_generation.py
│   │   │   ├── contract_analysis.py
│   │   │   ├── contract_reframing.py
│   │   │   └── case_evaluation.py
│   │   └── validators/
│   │       ├── __init__.py
│   │       ├── task_validator.py
│   │       └── context_validator.py
│   │
│   ├── interfaces/                    # Abstract Ports
│   │   ├── __init__.py
│   │   ├── ai/
│   │   │   ├── __init__.py
│   │   │   ├── rag_service.py         # RAG interface
│   │   │   ├── llm_service.py         # LLM interface
│   │   │   └── prompt_service.py      # Prompt interface
│   │   └── external/
│   │       ├── __init__.py
│   │       └── document_service.py
│   │
│   ├── infrastructure/                # Infrastructure
│   │   ├── __init__.py
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   ├── settings.py            # Environment config
│   │   │   └── constants.py
│   │   ├── logging/
│   │   │   ├── __init__.py
│   │   │   ├── logger.py
│   │   │   └── request_context.py
│   │   └── adapters/
│   │       ├── __init__.py
│   │       └── stub_ai_adapter.py     # Test stub
│   │
│   └── shared/                        # Shared Utilities
│       ├── __init__.py
│       ├── errors/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── task_errors.py
│       │   └── validation_errors.py
│       └── utils/
│           ├── __init__.py
│           ├── serializers.py
│           └── validators.py
│
├── tests/                             # Test Suite
│   ├── __init__.py
│   ├── unit/                          # Unit tests
│   │   ├── test_orchestrator.py
│   │   ├── test_workflows.py
│   │   └── test_validators.py
│   ├── integration/                   # Integration tests
│   │   └── test_api.py
│   └── fixtures/
│       └── sample_requests.json
│
├── docs/                              # Documentation
│   ├── architecture.md
│   ├── api_specification.md
│   └── workflow_guide.md
│
├── main.py                            # Entry point
├── requirements.txt                   # Dependencies
├── requirements-dev.txt               # Dev dependencies
├── pyproject.toml                     # Project config
├── .env.example                       # Env template
├── .gitignore
└── README.md
```

### Layer Responsibilities

| Layer              | Purpose                                   | Depends On |
| ------------------ | ----------------------------------------- | ---------- |
| **API**            | HTTP interface, routing, serialization    | Core       |
| **Core**           | Business logic, orchestration, workflows  | Interfaces |
| **Interfaces**     | Abstract contracts (ports)                | Nothing    |
| **Infrastructure** | Config, logging, concrete implementations | Interfaces |
| **Shared**         | Cross-cutting utilities                   | Nothing    |

---

## Layer Details

### 1. API Layer (`app/api/`)

**Purpose:** Handle HTTP concerns, request/response serialization

#### Key Files

**`main.py`** - FastAPI application initialization

```python
from fastapi import FastAPI
from app.api.routes import health, tasks
from app.api.middleware import setup_middleware

app = FastAPI(
    title="AI Service",
    description="Legal Tech AI Orchestration Service",
    version="0.1.0"
)

setup_middleware(app)
app.include_router(health.router)
app.include_router(tasks.router, prefix="/v1")
```

**`routes/tasks.py`** - Main task endpoint

```python
from fastapi import APIRouter, Depends
from app.api.schemas.requests import TaskRequest
from app.api.schemas.responses import TaskResponse
from app.core.orchestrator.task_orchestrator import TaskOrchestrator

router = APIRouter()

@router.post("/ai/tasks", response_model=TaskResponse)
async def execute_task(
    request: TaskRequest,
    orchestrator: TaskOrchestrator = Depends(get_orchestrator)
):
    """Execute an AI task"""
    result = await orchestrator.execute(request)
    return result
```

**`schemas/requests.py`** - Input validation

```python
from pydantic import BaseModel, Field
from app.core.domain.task_types import TaskType

class Context(BaseModel):
    jurisdiction: str = Field(..., description="Legal jurisdiction")
    language: str = Field(..., description="Response language")
    domain: str | None = Field(None, description="Legal domain")

class TaskRequest(BaseModel):
    task_type: TaskType
    context: Context
    payload: dict[str, Any]
    options: dict[str, Any] | None = None
```

#### Responsibilities

- ✅ HTTP routing
- ✅ Request validation (Pydantic)
- ✅ Response serialization
- ✅ Error handling (HTTP status codes)
- ❌ NO business logic
- ❌ NO direct AI operations

---

### 2. Core Layer (`app/core/`)

**Purpose:** Business logic, orchestration, domain models

#### Domain Models (`core/domain/`)

**`task_types.py`** - Task type enumeration

```python
from enum import Enum

class TaskType(str, Enum):
    LEGAL_CHAT = "LEGAL_CHAT"
    DOCUMENT_GENERATION = "DOCUMENT_GENERATION"
    CONTRACT_ANALYSIS = "CONTRACT_ANALYSIS"
    CONTRACT_REFRAMING = "CONTRACT_REFRAMING"
    CASE_EVALUATION = "CASE_EVALUATION"
```

**`context.py`** - Execution context

```python
from dataclasses import dataclass

@dataclass
class Context:
    jurisdiction: str
    language: str
    domain: str | None = None
    
    def validate(self) -> None:
        """Validate context requirements"""
        if self.jurisdiction not in SUPPORTED_JURISDICTIONS:
            raise ContextValidationError(
                f"Unsupported jurisdiction: {self.jurisdiction}"
            )
```

#### Orchestrator (`core/orchestrator/`)

**`task_orchestrator.py`** - Routes tasks to workflows

```python
class TaskOrchestrator:
    def __init__(self, registry: WorkflowRegistry):
        self.registry = registry
    
    async def execute(
        self,
        task_type: TaskType,
        context: Context,
        payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute task by delegating to appropriate workflow"""
        
        # Get workflow for task type
        workflow = self.registry.get_workflow(task_type)
        
        # Execute workflow
        result = await workflow.execute(context, payload)
        
        return result
```

**`workflow_registry.py`** - Maps TaskType → Workflow

```python
class WorkflowRegistry:
    def __init__(self):
        self._workflows: dict[TaskType, BaseWorkflow] = {}
    
    def register(self, task_type: TaskType, workflow: BaseWorkflow):
        self._workflows[task_type] = workflow
    
    def get_workflow(self, task_type: TaskType) -> BaseWorkflow:
        if task_type not in self._workflows:
            raise TaskNotSupportedError(f"No workflow for {task_type}")
        return self._workflows[task_type]
```

#### Workflows (`core/workflows/`)

**`base.py`** - Abstract workflow interface

```python
from abc import ABC, abstractmethod

class BaseWorkflow(ABC):
    @abstractmethod
    async def execute(
        self,
        context: Context,
        payload: dict[str, Any],
        options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute the workflow"""
        pass
```

**`legal_chat.py`** - Example workflow implementation

```python
class LegalChatWorkflow(BaseWorkflow):
    def __init__(
        self,
        rag_service: RAGService,
        llm_service: LLMService,
        prompt_service: PromptService
    ):
        self.rag = rag_service
        self.llm = llm_service
        self.prompt = prompt_service
    
    async def execute(
        self,
        context: Context,
        payload: dict[str, Any],
        options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # Validate payload
        message = payload.get("message")
        if not message:
            raise TaskValidationError("message required")
        
        # Retrieve relevant documents (RAG)
        docs = await self.rag.retrieve(
            query=message,
            jurisdiction=context.jurisdiction
        )
        
        # Assemble prompt
        prompt = await self.prompt.assemble(
            template="legal_chat",
            variables={
                "query": message,
                "docs": docs,
                "language": context.language
            }
        )
        
        # Generate response (LLM)
        response = await self.llm.generate(prompt)
        
        return {
            "message": response.content,
            "sources": [doc.id for doc in docs]
        }
```

#### Validators (`core/validators/`)

**`task_validator.py`** - Validate task-specific requirements

```python
class TaskValidator:
    def validate(self, task_type: TaskType, payload: dict) -> None:
        if task_type == TaskType.LEGAL_CHAT:
            self._validate_chat(payload)
        elif task_type == TaskType.DOCUMENT_GENERATION:
            self._validate_document(payload)
        # ... etc
    
    def _validate_chat(self, payload: dict) -> None:
        if "message" not in payload:
            raise TaskValidationError("message required for LEGAL_CHAT")
```

---

### 3. Interfaces Layer (`app/interfaces/`)

**Purpose:** Abstract contracts (ports) for external dependencies

#### AI Interfaces (`interfaces/ai/`)

**`rag_service.py`** - RAG service interface

```python
from abc import ABC, abstractmethod

class RAGService(ABC):
    """Abstract interface for RAG (Retrieval-Augmented Generation)"""
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        jurisdiction: str,
        top_k: int = 5
    ) -> list[Document]:
        """
        Retrieve relevant documents.
        
        Args:
            query: Search query
            jurisdiction: Legal jurisdiction for filtering
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        pass
```

**`llm_service.py`** - LLM service interface

```python
class LLMService(ABC):
    """Abstract interface for LLM operations"""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> LLMResponse:
        """
        Generate text using LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum response length
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        pass
```

**`prompt_service.py`** - Prompt service interface

```python
class PromptService(ABC):
    """Abstract interface for prompt assembly"""
    
    @abstractmethod
    async def assemble(
        self,
        template: str,
        variables: dict[str, Any]
    ) -> str:
        """
        Assemble prompt from template.
        
        Args:
            template: Template name
            variables: Template variables
            
        Returns:
            Assembled prompt
        """
        pass
```

#### Why Interfaces?

1. **Parallel Development**: You implement workflows while teammate builds AI services
2. **Testing**: Easy to mock for unit tests
3. **Flexibility**: Swap implementations (OpenAI → Claude → Local model)
4. **Clear Contracts**: Explicit expectations

---

### 4. Infrastructure Layer (`app/infrastructure/`)

**Purpose:** Configuration, logging, concrete implementations

#### Configuration (`infrastructure/config/`)

**`settings.py`** - Environment-based configuration

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Service
    SERVICE_NAME: str = "ai-service"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # API
    API_V1_PREFIX: str = "/v1"
    ENABLE_DOCS: bool = True
    
    # CORS
    CORS_ORIGINS: list[str] = []
    
    class Config:
        env_file = ".env"
        case_sensitive = True

def get_settings() -> Settings:
    return Settings()
```

**`constants.py`** - Application constants

```python
SUPPORTED_JURISDICTIONS = ["egypt", "uae", "saudi"]
SUPPORTED_LANGUAGES = ["ar", "en"]
DEFAULT_MAX_TOKENS = 1000
REQUEST_TIMEOUT = 30
```

#### Logging (`infrastructure/logging/`)

**`logger.py`** - Structured logging

```python
import structlog

def setup_logging(log_level: str, log_format: str) -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
            if log_format == "json"
            else structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

logger = structlog.get_logger()
```

**`request_context.py`** - Request tracing

```python
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id")

def get_request_id() -> str:
    return request_id_var.get(str(uuid.uuid4()))

def set_request_id(request_id: str) -> None:
    request_id_var.set(request_id)
```

#### Adapters (`infrastructure/adapters/`)

**`stub_ai_adapter.py`** - Testing stub

```python
class StubRAGService(RAGService):
    """Stub implementation for testing"""
    
    async def retrieve(
        self,
        query: str,
        jurisdiction: str,
        top_k: int = 5
    ) -> list[Document]:
        return [
            Document(id="doc1", content="Sample legal text"),
            Document(id="doc2", content="Another legal text")
        ]
```

---

### 5. Shared Layer (`app/shared/`)

**Purpose:** Cross-cutting concerns

#### Errors (`shared/errors/`)

**`base.py`** - Base exception classes

```python
class AIServiceError(Exception):
    """Base exception for all AI service errors"""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)
```

**`task_errors.py`** - Task-specific errors

```python
class TaskNotSupportedError(AIServiceError):
    def __init__(self, task_type: str):
        super().__init__(
            message=f"Task type '{task_type}' is not supported",
            code="TASK_NOT_SUPPORTED"
        )

class TaskValidationError(AIServiceError):
    def __init__(self, message: str):
        super().__init__(message, code="TASK_VALIDATION_ERROR")
```

**`validation_errors.py`** - Validation errors

```python
class ContextValidationError(AIServiceError):
    def __init__(self, message: str):
        super().__init__(message, code="CONTEXT_VALIDATION_ERROR")
```

---

## Domain Models

### TaskType Enum

Defines all supported task types:

```python
class TaskType(str, Enum):
    LEGAL_CHAT = "LEGAL_CHAT"
    DOCUMENT_GENERATION = "DOCUMENT_GENERATION"
    CONTRACT_ANALYSIS = "CONTRACT_ANALYSIS"
    CONTRACT_REFRAMING = "CONTRACT_REFRAMING"
    CASE_EVALUATION = "CASE_EVALUATION"
```

### Context Model

Execution context for all tasks:

```python
@dataclass
class Context:
    jurisdiction: str  # e.g., "egypt", "uae"
    language: str      # e.g., "ar", "en"
    domain: str | None = None  # e.g., "civil", "commercial"
```

### Task Entity

Represents a task request:

```python
@dataclass
class Task:
    task_id: str
    task_type: TaskType
    context: Context
    payload: dict[str, Any]
    options: dict[str, Any] | None = None
    created_at: datetime
```

### Result Entity

Represents task execution result:

```python
@dataclass
class Result:
    task_id: str
    task_type: TaskType
    status: str  # "success", "error"
    result: dict[str, Any]
    metadata: dict[str, Any]
    execution_time_ms: int
```

---

## API Specification

### Base URL

```txt
http://localhost:8000/v1
```

### Endpoints

#### 1. Health Check

**GET** `/health`

**Response:**

```json
{
  "status": "healthy",
  "service": "ai-service",
  "version": "0.1.0",
  "timestamp": "2026-01-24T12:00:00Z"
}
```

#### 2. Execute AI Task

**POST** `/ai/tasks`

**Request:**

```json
{
  "task_type": "LEGAL_CHAT",
  "context": {
    "jurisdiction": "egypt",
    "language": "ar",
    "domain": "civil"
  },
  "payload": {
    "message": "ما هي شروط العقد الصحيح؟",
    "conversation_history": []
  },
  "options": {
    "max_tokens": 1000,
    "temperature": 0.7
  }
}
```

**Response (Success - 200):**

```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "task_type": "LEGAL_CHAT",
  "status": "success",
  "result": {
    "message": "شروط العقد الصحيح هي...",
    "sources": ["law_123", "article_456"]
  },
  "metadata": {
    "execution_time_ms": 1234,
    "model_used": "gpt-4",
    "tokens_used": 450
  }
}
```

**Response (Error - 400):**

```json
{
  "error": {
    "code": "TASK_NOT_SUPPORTED",
    "message": "Task type 'UNKNOWN' is not supported",
    "request_id": "req_123"
  }
}
```

### Task-Specific Payloads

#### LEGAL_CHAT

```json
{
  "message": "User question",
  "conversation_history": [
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous answer"}
  ]
}
```

#### DOCUMENT_GENERATION

```json
{
  "document_type": "contract",
  "parameters": {
    "party_a": "أحمد محمد",
    "party_b": "شركة XYZ",
    "subject": "توريد معدات",
    "clauses": ["payment", "delivery", "warranty"]
  }
}
```

#### CONTRACT_ANALYSIS

```json
{
  "contract_text": "نص العقد الكامل...",
  "analysis_type": "risk_assessment",
  "focus_areas": ["obligations", "penalties", "termination"]
}
```

### HTTP Status Codes

| Code | Meaning             | When                                          |
| ---- | ------------------- | --------------------------------------------- |
| 200  | Success             | Task executed successfully                    |
| 400  | Bad Request         | Invalid task type or missing required context |
| 422  | Validation Error    | Request doesn't match schema                  |
| 500  | Internal Error      | Unexpected server error                       |
| 503  | Service Unavailable | External AI services down                     |

---

## Workflow System

### Workflow Lifecycle

```txt
1. Request arrives at API endpoint
   ↓
2. Pydantic validates request schema
   ↓
3. Orchestrator receives validated request
   ↓
4. Orchestrator looks up workflow from registry
   ↓
5. Workflow validates task-specific requirements
   ↓
6. Workflow calls RAG service (retrieve documents)
   ↓
7. Workflow calls Prompt service (assemble prompt)
   ↓
8. Workflow calls LLM service (generate response)
   ↓
9. Workflow structures result
   ↓
10. Orchestrator returns result to API
   ↓
11. API serializes response and returns to client
```

### Workflow Pattern

Each workflow follows this pattern:

```python
class SomeWorkflow(BaseWorkflow):
    def __init__(
        self,
        rag: RAGService,
        llm: LLMService,
        prompt: PromptService
    ):
        self.rag = rag
        self.llm = llm
        self.prompt = prompt
    
    async def execute(
        self,
        context: Context,
        payload: dict[str, Any],
        options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # 1. Validate payload
        self._validate_payload(payload)
        
        # 2. Retrieve context (RAG)
        docs = await self.rag.retrieve(...)
        
        # 3. Assemble prompt
        prompt = await self.prompt.assemble(...)
        
        # 4. Generate response (LLM)
        response = await self.llm.generate(prompt)
        
        # 5. Structure result
        return {
            "message": response.content,
            "sources": [doc.id for doc in docs]
        }
```

### Adding a New Workflow

1. Create file in `app/core/workflows/`
2. Extend `BaseWorkflow`
3. Implement `execute()` method
4. Register in workflow registry
5. Add task type to `TaskType` enum
6. Document payload schema
7. Write tests

---

## Error Handling

### Exception Hierarchy

```txt
AIServiceError (base)
├── TaskNotSupportedError
├── TaskValidationError
├── ContextValidationError
├── ExternalServiceError
│   ├── RAGServiceError
│   ├── LLMServiceError
│   └── PromptServiceError
└── WorkflowExecutionError
```

### FastAPI Exception Handlers

```python
@app.exception_handler(TaskNotSupportedError)
async def task_not_supported_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "request_id": get_request_id()
            }
        }
    )

@app.exception_handler(TaskValidationError)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "request_id": get_request_id()
            }
        }
    )
```

### Error Response Format

All errors follow this structure:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "request_id": "uuid",
    "details": {}  // Optional
  }
}
```

---

## Configuration

### Environment Variables

Create `.env` file from `.env.example`:

```bash
# Service
SERVICE_NAME=ai-service
VERSION=0.1.0
ENVIRONMENT=development

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# API
API_V1_PREFIX=/v1
ENABLE_DOCS=true

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Future: External Services
# RAG_SERVICE_URL=http://localhost:8001
# LLM_SERVICE_URL=http://localhost:8002
```

### Accessing Configuration

```python
from app.infrastructure.config.settings import get_settings

settings = get_settings()
print(settings.PORT)  # 8000
```

---

## Setup & Installation

### Prerequisites

- Python 3.11 or higher
- pip
- Virtual environment tool (venv, virtualenv, or conda)

### Installation Steps

1. **Clone/Navigate to project:**

   ```bash
   cd graduation-project-ai-service
   ```

2. **Create virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install development dependencies (optional):**

   ```bash
   pip install -r requirements-dev.txt
   ```

5. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

6. **Run the service:**

   ```bash
   python main.py
   ```

7. **Access API documentation:**
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

### Verification

Test health endpoint:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "service": "ai-service",
  "version": "0.1.0"
}
```

---

## Development Guide

### Code Style

- **Formatter:** Black (line length: 100)
- **Import sorting:** isort
- **Type checking:** mypy
- **Linting:** flake8

Format code:

```bash
black app/ tests/
isort app/ tests/
```

Type check:

```bash
mypy app/
```

### Project Structure Guidelines

1. **Keep layers separate**
   - API layer should not import from Infrastructure
   - Core should not import concrete implementations

2. **Follow naming conventions**
   - Files: `snake_case.py`
   - Classes: `PascalCase`
   - Functions/methods: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`

3. **Type hints everywhere**

   ```python
   def execute(self, task: Task) -> Result:
       pass
   ```

4. **Document public APIs**

   ```python
   def execute(self, task: Task) -> Result:
       """
       Execute a task.
       
       Args:
           task: Task to execute
           
       Returns:
           Execution result
           
       Raises:
           TaskNotSupportedError: If task type not supported
       """
   ```

### Adding a New Feature

#### **Example: Add new task type "LEGAL_RESEARCH"**

1. **Add to TaskType enum:**

   ```python
   # app/core/domain/task_types.py
   class TaskType(str, Enum):
       # ... existing
       LEGAL_RESEARCH = "LEGAL_RESEARCH"
   ```

2. **Create workflow:**

   ```python
   # app/core/workflows/legal_research.py
   class LegalResearchWorkflow(BaseWorkflow):
       async def execute(self, context, payload, options):
           # Implementation
           pass
   ```

3. **Register workflow:**

   ```python
   # app/api/dependencies.py
   registry.register(
       TaskType.LEGAL_RESEARCH,
       LegalResearchWorkflow(rag, llm, prompt)
   )
   ```

4. **Document payload schema:**

   ```python
   # Add to docs/api_specification.md
   ```

5. **Write tests:**

   ```python
   # tests/unit/test_legal_research_workflow.py
   ```

---

## Testing Strategy

### Test Structure

```txt
tests/
├── unit/                  # Fast, isolated tests
│   ├── test_orchestrator.py
│   ├── test_workflows.py
│   └── test_validators.py
├── integration/           # API-level tests
│   └── test_api.py
└── fixtures/              # Test data
    └── sample_requests.json
```

### Unit Testing

Test workflows with mocked dependencies:

```python
import pytest
from app.core.workflows.legal_chat import LegalChatWorkflow

@pytest.mark.asyncio
async def test_legal_chat_workflow():
    # Arrange
    rag = MockRAGService()
    llm = MockLLMService()
    prompt = MockPromptService()
    workflow = LegalChatWorkflow(rag, llm, prompt)
    
    context = Context(jurisdiction="egypt", language="ar")
    payload = {"message": "Test question"}
    
    # Act
    result = await workflow.execute(context, payload)
    
    # Assert
    assert "message" in result
    assert "sources" in result
    assert len(result["sources"]) > 0
```

### Integration Testing

Test API endpoints:

```python
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_execute_task():
    response = client.post("/v1/ai/tasks", json={
        "task_type": "LEGAL_CHAT",
        "context": {
            "jurisdiction": "egypt",
            "language": "ar"
        },
        "payload": {
            "message": "Test question"
        }
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "success"
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=app tests/

# Specific file
pytest tests/unit/test_orchestrator.py

# Specific test
pytest tests/unit/test_orchestrator.py::test_execute_task
```

### Test Coverage Goals

- **Core logic:** 90%+
- **Workflows:** 85%+
- **API routes:** 80%+
- **Overall:** 85%+

---

## Deployment

### Docker Deployment (Future)

**Dockerfile:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY main.py .

EXPOSE 8000

CMD ["python", "main.py"]
```

**Build and run:**

```bash
docker build -t ai-service .
docker run -p 8000:8000 --env-file .env ai-service
```

### Environment-Specific Configuration

**Development:**

```env
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
ENABLE_DOCS=true
```

**Production:**

```env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
ENABLE_DOCS=false
```

### Health Checks

Kubernetes/Docker health check:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
```

---

## Code Examples

### Complete Request Flow

**1. Client sends request:**

```python
import httpx

response = httpx.post("http://localhost:8000/v1/ai/tasks", json={
    "task_type": "LEGAL_CHAT",
    "context": {
        "jurisdiction": "egypt",
        "language": "ar"
    },
    "payload": {
        "message": "ما هي شروط العقد الصحيح؟"
    }
})

print(response.json())
```

**2. FastAPI route receives and validates:**

```python
# app/api/routes/tasks.py
@router.post("/ai/tasks")
async def execute_task(request: TaskRequest):
    # Pydantic validates automatically
    result = await orchestrator.execute(request)
    return result
```

**3. Orchestrator routes to workflow:**

```python
# app/core/orchestrator/task_orchestrator.py
workflow = self.registry.get_workflow(request.task_type)
result = await workflow.execute(
    request.context,
    request.payload,
    request.options
)
```

**4. Workflow executes:**

```python
# app/core/workflows/legal_chat.py
# Retrieve documents
docs = await self.rag.retrieve(query=message)

# Assemble prompt
prompt = await self.prompt.assemble("legal_chat", {
    "query": message,
    "docs": docs
})

# Generate response
response = await self.llm.generate(prompt)

return {"message": response.content, "sources": [d.id for d in docs]}
```

### Dependency Injection Example

```python
# app/api/dependencies.py
from app.core.orchestrator.task_orchestrator import TaskOrchestrator
from app.infrastructure.adapters.stub_ai_adapter import StubRAGService

def get_rag_service() -> RAGService:
    # In production, return real implementation
    # For now, return stub
    return StubRAGService()

def get_orchestrator() -> TaskOrchestrator:
    rag = get_rag_service()
    llm = get_llm_service()
    prompt = get_prompt_service()
    
    registry = WorkflowRegistry()
    registry.register(
        TaskType.LEGAL_CHAT,
        LegalChatWorkflow(rag, llm, prompt)
    )
    
    return TaskOrchestrator(registry)
```

### Logging Example

```python
from app.infrastructure.logging.logger import logger

logger.info(
    "task_executed",
    task_type=task_type,
    execution_time_ms=execution_time,
    request_id=request_id
)
```

Output (JSON format):

```json
{
  "event": "task_executed",
  "task_type": "LEGAL_CHAT",
  "execution_time_ms": 1234,
  "request_id": "req_123",
  "timestamp": "2026-01-24T12:00:00Z",
  "level": "info"
}
```

---

## Appendix

### Key Technologies

- **FastAPI:** Web framework
- **Pydantic:** Data validation
- **Uvicorn:** ASGI server
- **Structlog:** Structured logging
- **Pytest:** Testing framework
- **Black:** Code formatter
- **MyPy:** Type checker

### Glossary

- **Clean Architecture:** Architectural pattern with dependency inversion
- **Workflow:** Use case implementation for a task type
- **Orchestrator:** Component that routes tasks to workflows
- **Port:** Abstract interface for external dependencies
- **Adapter:** Concrete implementation of a port
- **Context:** Execution context (jurisdiction, language, etc.)
- **Payload:** Task-specific input data

### Future Enhancements

1. **Authentication:** JWT-based auth for inter-service communication
2. **Rate Limiting:** Protect against abuse
3. **Caching:** Response caching for repeated requests
4. **Async Processing:** Background jobs for long-running tasks
5. **Metrics:** Prometheus metrics integration
6. **Tracing:** OpenTelemetry distributed tracing
7. **Database:** Store conversation history
8. **Message Queue:** RabbitMQ/Kafka for async processing

---

## Conclusion

This AI Service provides a solid, scalable foundation for orchestrating AI tasks in a legal-tech application. By following Clean Architecture principles and maintaining clear boundaries between layers, the service is:

- **Testable:** Easy to write unit and integration tests
- **Maintainable:** Clear structure and separation of concerns
- **Scalable:** Stateless design enables horizontal scaling
- **Flexible:** Easy to swap implementations without core changes
- **Collaborative:** Clear interfaces enable parallel team development

The service is ready to integrate with external RAG, LLM, and Prompt services once they are implemented by your teammates.

---

**Document Version:** 1.0  
**Last Updated:** January 24, 2026  
**Author:** AI Service Development Team  
**Status:** Initial Implementation
