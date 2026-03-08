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

## 🐳 Docker

The project includes a production-grade, multi-stage `Dockerfile` that produces a lean, secure image.

### Building the Image

```bash
docker build -t ai-service .
```

To tag for the private registry:

```bash
docker build -t registry.moeenlaw.com/ai-service:latest .
```

You can also apply a version tag:

```bash
docker build -t registry.moeenlaw.com/ai-service:1.0.0 \
             -t registry.moeenlaw.com/ai-service:latest .
```

### Running the Container

Pass your `.env` file and expose port **8000**:

```bash
docker run -d \
  --name ai-service \
  --env-file .env \
  -p 8000:8000 \
  registry.moeenlaw.com/ai-service:latest
```

Verify the service is healthy:

```bash
docker ps                         # STATUS should show (healthy)
curl http://localhost:8000/docs   # Should return the Swagger UI
```

### Pushing to the Registry

1. **Log in** to the private registry (first time only):

   ```bash
   docker login registry.moeenlaw.com
   ```

2. **Build & tag** (if not already tagged):

   ```bash
   docker build -t registry.moeenlaw.com/ai-service:latest .
   ```

3. **Push** the image:

   ```bash
   docker push registry.moeenlaw.com/ai-service:latest
   ```

   Push a specific version tag:

   ```bash
   docker push registry.moeenlaw.com/ai-service:1.0.0
   ```

### Environment Variables

All configuration is passed via environment variables. See [`.env.example`](.env.example) for the full list. Key variables for production:

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | `development` / `staging` / `production` | `development` |
| `HOST` | Bind address | `0.0.0.0` |
| `PORT` | Listen port | `8000` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `GEMINI_API_KEY` | Google Gemini API key | — |
| `QDRANT_URL` | Qdrant vector DB URL | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant API key | — |
