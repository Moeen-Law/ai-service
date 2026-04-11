"""
Integration Tests for Task API

Tests the /v1/ai/tasks endpoint with full orchestration pipeline.
"""

import pytest
from fastapi.testclient import TestClient
from uuid import uuid4

from app.api.routes import tasks as tasks_route
from app.api.main import create_app
from app.core.domain.entities import Result, ResultMetadata
from app.core.domain.enums import TaskType


class TestTasksAPI:
    """Integration tests for the tasks API."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        app = create_app()
        return TestClient(app)

    # --- Health Check Tests ---

    def test_health_endpoint(self, client: TestClient) -> None:
        """Test health endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "ai-service"

    def test_ready_endpoint(self, client: TestClient) -> None:
        """Test readiness endpoint."""
        response = client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        # Ready endpoint returns healthy status
        assert data["status"] == "healthy"

    # --- LEGAL_CHAT Task Tests ---

    def test_legal_chat_success(self, client: TestClient) -> None:
        """Test successful LEGAL_CHAT task execution."""
        response = client.post(
            "/v1/ai/tasks",
            json={
                "task_type": "LEGAL_CHAT",
                "context": {
                    "jurisdiction": "EGYPT",
                    "language": "ar",
                },
                "payload": {
                    "message": "ما هي شروط الزواج المدني؟",
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_type"] == "LEGAL_CHAT"
        assert data["status"] == "success"
        assert "result" in data
        # Result contains either 'message' or 'content' depending on task type
        assert "message" in data["result"] or "content" in data["result"]
        assert "sources" in data["result"]

    def test_legal_chat_english(self, client: TestClient) -> None:
        """Test LEGAL_CHAT with English language."""
        response = client.post(
            "/v1/ai/tasks",
            json={
                "task_type": "LEGAL_CHAT",
                "context": {
                    "jurisdiction": "UAE",
                    "language": "en",
                },
                "payload": {
                    "message": "What are the requirements for company registration?",
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_legal_chat_missing_message(self, client: TestClient) -> None:
        """Test LEGAL_CHAT without required message field."""
        response = client.post(
            "/v1/ai/tasks",
            json={
                "task_type": "LEGAL_CHAT",
                "context": {
                    "jurisdiction": "EGYPT",
                    "language": "ar",
                },
                "payload": {},
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert data["detail"]["code"] == "PAYLOAD_VALIDATION_ERROR"

    # --- CONTRACT_ANALYSIS Task Tests ---

    def test_contract_analysis_success(self, client: TestClient) -> None:
        """Test successful CONTRACT_ANALYSIS task execution."""
        response = client.post(
            "/v1/ai/tasks",
            json={
                "task_type": "CONTRACT_ANALYSIS",
                "context": {
                    "jurisdiction": "EGYPT",
                    "language": "ar",
                    "domain": "COMMERCIAL",
                },
                "payload": {
                    "contract_text": "نص العقد التجاري...",
                    "analysis_type": "risk_assessment",
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_type"] == "CONTRACT_ANALYSIS"
        assert data["status"] == "success"

    def test_contract_analysis_missing_domain(self, client: TestClient) -> None:
        """Test CONTRACT_ANALYSIS without required domain context."""
        response = client.post(
            "/v1/ai/tasks",
            json={
                "task_type": "CONTRACT_ANALYSIS",
                "context": {
                    "jurisdiction": "EGYPT",
                    "language": "ar",
                    # Missing domain
                },
                "payload": {
                    "contract_text": "نص العقد...",
                    "analysis_type": "risk_assessment",
                },
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert "domain" in data["detail"]["message"].lower()

    # --- Validation Error Tests ---

    def test_unsupported_jurisdiction(self, client: TestClient) -> None:
        """Test that unsupported jurisdiction returns proper error."""
        response = client.post(
            "/v1/ai/tasks",
            json={
                "task_type": "LEGAL_CHAT",
                "context": {
                    "jurisdiction": "USA",  # Not supported
                    "language": "en",
                },
                "payload": {
                    "message": "Test query",
                },
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert data["detail"]["code"] == "CONTEXT_VALIDATION_ERROR"

    def test_invalid_task_type(self, client: TestClient) -> None:
        """Test that invalid task type returns validation error."""
        response = client.post(
            "/v1/ai/tasks",
            json={
                "task_type": "INVALID_TASK",
                "context": {
                    "jurisdiction": "EGYPT",
                    "language": "ar",
                },
                "payload": {},
            },
        )

        assert response.status_code == 422
        # Pydantic validation error for invalid enum

    # --- Request ID Propagation Tests ---

    def test_request_id_header_propagation(self, client: TestClient) -> None:
        """Test that X-Request-ID is propagated to response."""
        custom_request_id = "test-request-123"

        response = client.post(
            "/v1/ai/tasks",
            headers={"X-Request-ID": custom_request_id},
            json={
                "task_type": "LEGAL_CHAT",
                "context": {
                    "jurisdiction": "EGYPT",
                    "language": "ar",
                },
                "payload": {
                    "message": "Test query",
                },
            },
        )

        assert response.headers.get("X-Request-ID") == custom_request_id

    def test_request_id_generated_if_not_provided(self, client: TestClient) -> None:
        """Test that X-Request-ID is generated if not in request."""
        response = client.post(
            "/v1/ai/tasks",
            json={
                "task_type": "LEGAL_CHAT",
                "context": {
                    "jurisdiction": "EGYPT",
                    "language": "ar",
                },
                "payload": {
                    "message": "Test query",
                },
            },
        )

        assert "X-Request-ID" in response.headers
        # Should be a valid UUID format
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) == 36  # UUID format

    # --- All Task Types Tests ---

    def test_document_generation(self, client: TestClient) -> None:
        """Test DOCUMENT_GENERATION task."""
        response = client.post(
            "/v1/ai/tasks",
            json={
                "task_type": "DOCUMENT_GENERATION",
                "context": {
                    "jurisdiction": "EGYPT",
                    "language": "ar",
                },
                "payload": {
                    "document_type": "contract",
                    "parameters": {
                        "landlord_name": "المالك",
                        "tenant_name": "المستأجر",
                    },
                },
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_contract_reframing(self, client: TestClient) -> None:
        """Test CONTRACT_REFRAMING task."""
        response = client.post(
            "/v1/ai/tasks",
            json={
                "task_type": "CONTRACT_REFRAMING",
                "context": {
                    "jurisdiction": "UAE",
                    "language": "en",
                },
                "payload": {
                    "clause_text": "The seller shall not be liable...",
                    "target_perspective": "balanced",
                },
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_case_evaluation(self, client: TestClient) -> None:
        """Test CASE_EVALUATION task."""
        response = client.post(
            "/v1/ai/tasks",
            json={
                "task_type": "CASE_EVALUATION",
                "context": {
                    "jurisdiction": "SAUDI_ARABIA",
                    "language": "ar",
                    "domain": "COMMERCIAL",  # Required for case evaluation
                },
                "payload": {
                    "case_description": "وصف القضية التجارية...",
                },
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_stream_with_files_ids_supported(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test stream endpoint accepts files_ids and forwards them to workflow."""

        captured_files_ids = []

        class FakeStreamWorkflow:
            async def stream(self, **kwargs):
                captured_files_ids.extend(kwargs.get("files_ids") or [])
                yield 'data: {"type":"token","content":"ok"}\n\n'
                yield "data: [DONE]\n\n"

        monkeypatch.setattr(
            tasks_route.workflow_registry,
            "get",
            lambda _task_type: FakeStreamWorkflow(),
        )

        response = client.post(
            "/v1/ai/tasks/stream",
            json={
                "task_type": "LEGAL_CHAT",
                "context": {
                    "jurisdiction": "EGYPT",
                    "language": "ar",
                },
                "payload": {
                    "message": "حلل الملف",
                    "files_ids": ["file_123"],
                },
            },
        )

        assert response.status_code == 200
        assert "ok" in response.text
        assert captured_files_ids == ["file_123"]

    def test_legal_chat_prompt_generation_returns_files_ids(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test LEGAL_CHAT generation prompt returns generated file IDs."""

        async def fake_execute(*args, **kwargs):
            return Result.success(
                task_id=uuid4(),
                task_type=TaskType.LEGAL_CHAT,
                data={
                    "message": "Generated contract text",
                    "document_content": "Generated contract text",
                    "format": "docx",
                    "files_ids": ["mock_file_123"],
                },
                metadata=ResultMetadata(
                    execution_time_ms=12,
                    model_used="fake-model",
                    tokens_used=20,
                ),
            )

        monkeypatch.setattr(tasks_route.task_orchestrator, "execute", fake_execute)

        response = client.post(
            "/v1/ai/tasks",
            json={
                "task_type": "LEGAL_CHAT",
                "context": {
                    "jurisdiction": "EGYPT",
                    "language": "ar",
                },
                "payload": {
                    "message": "Generate a contract for apartment rental between two parties",
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_type"] == "LEGAL_CHAT"
        assert data["status"] == "success"
        assert "result" in data
        assert "files_ids" in data["result"]
        assert isinstance(data["result"]["files_ids"], list)
        assert len(data["result"]["files_ids"]) == 1
