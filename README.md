# AI Service - Legal Tech Platform

A clean, scalable FastAPI backend for AI task orchestration in a legal-tech microservices system.

## 🎯 Purpose

This service acts as a **platform service** that:

- Receives structured AI task requests from other microservices
- Validates and classifies tasks explicitly
- Routes tasks to appropriate workflows
- Coordinates RAG, LLM, and prompt services (delegated to other components)
- Returns structured responses

## 🏗️ Architecture

This service follows **Clean Architecture** principles with clear layer separation:

```txt
API Layer (FastAPI routes, schemas)
    ↓
Core Layer (Orchestrator, Workflows, Domain Models)
    ↓
Interfaces Layer (Abstract ports for RAG/LLM)
    ↓
Infrastructure Layer (Config, Logging, Adapters)
```

### Key Principles

- **Task-based routing**: Single endpoint `/v1/ai/tasks` with explicit task types
- **Explicit classification**: Task type specified in request, never inferred
- **Dependency inversion**: Core depends on interfaces, not implementations
- **Stateless**: All context passed in requests
- **No AI logic**: RAG/LLM handled by external services

## 📁 Project Structure

```txt
graduation-project-ai-service/
├── app/
│   ├── api/                    # HTTP interface
│   ├── core/                   # Business logic
│   ├── interfaces/             # Abstract ports
│   ├── infrastructure/         # Config, logging
│   └── shared/                 # Cross-cutting concerns
├── tests/                      # Test suite
├── docs/                       # Documentation
└── main.py                     # Entry point
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip

### Installation

1. Clone the repository
2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy environment configuration:

   ```bash
   cp .env.example .env
   ```

5. Run the service:

   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

## 📋 Task Types

Supported task types:

- `LEGAL_CHAT` - Interactive legal assistance
- `DOCUMENT_GENERATION` - Generate legal documents
- `CONTRACT_ANALYSIS` - Analyze contract contents
- `CONTRACT_REFRAMING` - Reframe contract clauses
- `CASE_EVALUATION` - Evaluate legal cases

## 🧪 Testing

Run tests:

```bash
pytest
```

With coverage:

```bash
pytest --cov=app tests/
```

## 📚 Documentation

See `docs/` folder for:

- Architecture decisions
- API specifications
- Workflow implementation guide

## 🔧 Development

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Code formatting:

```bash
black app/ tests/
isort app/ tests/
```

Type checking:

```bash
mypy app/
```

## 📝 API Example

```bash
curl -X POST http://localhost:8000/v1/ai/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "LEGAL_CHAT",
    "context": {
      "jurisdiction": "egypt",
      "language": "ar"
    },
    "payload": {
      "message": "ما هي شروط العقد الصحيح؟"
    }
  }'
```
