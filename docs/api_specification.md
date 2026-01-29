# API Specification

## Base URL

```txt
http://localhost:8000/v1
```

## Endpoints

### 1. Health Check

**GET** `/health`

Returns the service health status.

**Response**:

```json
{
  "status": "healthy",
  "service": "ai-service",
  "version": "0.1.0"
}
```

---

### 2. Execute AI Task

**POST** `/ai/tasks`

Main endpoint for executing AI tasks.

#### Request Body

```json
{
  "task_type": "LEGAL_CHAT",
  "context": {
    "jurisdiction": "egypt",
    "language": "ar",
    "domain": "civil"
  },
  "payload": {
    // Task-specific data
  },
  "options": {
    // Optional behavior flags
  }
}
```

#### Request Schema

| Field       | Type   | Required | Description                                                                                     |
| ----------- | ------ | -------- | ----------------------------------------------------------------------------------------------- |
| `task_type` | enum   | Yes      | One of: LEGAL_CHAT, DOCUMENT_GENERATION, CONTRACT_ANALYSIS, CONTRACT_REFRAMING, CASE_EVALUATION |
| `context`   | object | Yes      | Execution context (jurisdiction, language, etc.)                                                |
| `payload`   | object | Yes      | Task-specific input data                                                                        |
| `options`   | object | No       | Optional execution flags                                                                        |

#### Context Schema

| Field          | Type   | Required | Description                                            |
| -------------- | ------ | -------- | ------------------------------------------------------ |
| `jurisdiction` | string | Yes      | Legal jurisdiction (e.g., "egypt", "uae")              |
| `language`     | string | Yes      | Response language (e.g., "ar", "en")                   |
| `domain`       | string | No       | Legal domain (e.g., "civil", "criminal", "commercial") |

#### Response Schema

```json
{
  "task_id": "uuid",
  "task_type": "LEGAL_CHAT",
  "status": "success",
  "result": {
    // Task-specific output
  },
  "metadata": {
    "execution_time_ms": 1234,
    "model_used": "gpt-4"
  }
}
```

---

## Task-Specific Payloads

### LEGAL_CHAT

**Request Payload**:

```json
{
  "message": "ما هي شروط العقد الصحيح؟",
  "conversation_history": [
    {
      "role": "user",
      "content": "مرحبا"
    },
    {
      "role": "assistant",
      "content": "مرحبا بك، كيف يمكنني مساعدتك؟"
    }
  ]
}
```

**Response Result**:

```json
{
  "message": "شروط العقد الصحيح هي...",
  "sources": ["article_123", "law_456"]
}
```

---

### DOCUMENT_GENERATION

**Request Payload**:

```json
{
  "document_type": "contract",
  "parameters": {
    "party_a": "أحمد محمد",
    "party_b": "شركة XYZ",
    "subject": "توريد معدات"
  }
}
```

**Response Result**:

```json
{
  "document_content": "...",
  "format": "markdown"
}
```

---

### CONTRACT_ANALYSIS

**Request Payload**:

```json
{
  "contract_text": "...",
  "analysis_type": "risk_assessment"
}
```

**Response Result**:

```json
{
  "risks": [
    {
      "clause": "البند 3.2",
      "risk_level": "high",
      "description": "..."
    }
  ],
  "summary": "..."
}
```

---

## Error Responses

### Validation Error (422)

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request data",
    "details": [
      {
        "field": "task_type",
        "message": "field required"
      }
    ]
  }
}
```

### Task Not Supported (400)

```json
{
  "error": {
    "code": "TASK_NOT_SUPPORTED",
    "message": "Task type 'UNKNOWN' is not supported"
  }
}
```

### Internal Error (500)

```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An unexpected error occurred",
    "request_id": "uuid"
  }
}
```

---

## Status Codes

| Code | Description                                               |
| ---- | --------------------------------------------------------- |
| 200  | Success                                                   |
| 400  | Bad Request (invalid task type, missing required context) |
| 422  | Validation Error (invalid request schema)                 |
| 500  | Internal Server Error                                     |
| 503  | Service Unavailable (downstream services unavailable)     |
