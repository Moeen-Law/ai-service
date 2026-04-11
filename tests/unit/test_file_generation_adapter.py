"""Unit tests for Files service generation adapter."""

import httpx
import pytest

from app.infrastructure.adapters.file_generation_adapter import (
    HTTPFileGenerationService,
)
from app.infrastructure.config.settings import get_settings
from app.interfaces.external.file_generation_service import FileGenerationRequest


@pytest.mark.asyncio
async def test_generate_docx_requests_upload_url_then_uploads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("FILES_SERVICE_BASE_URL", "https://gateway.moeenlaw.com")
    monkeypatch.setenv(
        "FILES_SERVICE_UPLOAD_URL_PATH", "/files/api/v1/files/upload-url"
    )
    monkeypatch.setenv("FILES_SERVICE_AUTH_TOKEN", "token-123")
    monkeypatch.setenv("FILES_SERVICE_UPLOAD_BUCKET", "AI_DOCUMENTS")

    service = HTTPFileGenerationService()

    upload_url_endpoint = "https://gateway.moeenlaw.com/files/api/v1/files/upload-url"
    upload_url = "https://blob.moeenlaw.com/user-documents/file-123/doc.docx?sig=abc"

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == upload_url_endpoint and request.method == "POST":
            assert request.headers.get("Authorization") == "Bearer token-123"
            payload = request.read().decode("utf-8")
            assert '"fileName":"rental.docx"' in payload
            assert (
                '"mimeType":"application/vnd.openxmlformats-officedocument.wordprocessingml.document"'
                in payload
            )
            assert '"bucket":"AI_DOCUMENTS"' in payload
            assert "entityId" not in payload
            assert "uploaderId" not in payload
            return httpx.Response(
                200,
                json={
                    "fileId": "8820811d-289e-45d1-9df6-1255c0cdd607",
                    "uploadUrl": upload_url,
                    "expiresInSeconds": 900,
                    "httpMethod": "PUT",
                },
            )

        if str(request.url) == upload_url and request.method == "PUT":
            assert request.headers.get("Content-Type") == (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            assert request.content
            return httpx.Response(200)

        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    service._build_client = lambda: httpx.AsyncClient(transport=transport, timeout=1.0)  # type: ignore[method-assign]

    result = await service.generate_docx(
        FileGenerationRequest(
            source_prompt="Generate rental contract",
            content="Document content",
            filename="rental.docx",
        )
    )

    assert result.file_id == "8820811d-289e-45d1-9df6-1255c0cdd607"
    assert result.filename == "rental.docx"
    assert result.size_bytes > 0


@pytest.mark.asyncio
async def test_generate_docx_uses_bucket_from_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("FILES_SERVICE_BASE_URL", "https://gateway.moeenlaw.com")
    monkeypatch.setenv(
        "FILES_SERVICE_UPLOAD_URL_PATH", "/files/api/v1/files/upload-url"
    )
    monkeypatch.setenv("FILES_SERVICE_UPLOAD_BUCKET", "AI_DOCUMENTS")

    service = HTTPFileGenerationService()
    upload_url_endpoint = "https://gateway.moeenlaw.com/files/api/v1/files/upload-url"
    upload_url = "https://blob.moeenlaw.com/user-documents/file-456/doc.docx?sig=abc"

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == upload_url_endpoint and request.method == "POST":
            payload = request.read().decode("utf-8")
            assert '"bucket":"CUSTOM_BUCKET"' in payload
            return httpx.Response(
                200,
                json={
                    "fileId": "8820811d-289e-45d1-9df6-1255c0cdd608",
                    "uploadUrl": upload_url,
                    "httpMethod": "PUT",
                },
            )

        if str(request.url) == upload_url and request.method == "PUT":
            return httpx.Response(200)

        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    service._build_client = lambda: httpx.AsyncClient(transport=transport, timeout=1.0)  # type: ignore[method-assign]

    result = await service.generate_docx(
        FileGenerationRequest(
            source_prompt="Generate",
            content="abc",
            filename="doc.docx",
            metadata={"bucket": "CUSTOM_BUCKET"},
        )
    )

    assert result.file_id == "8820811d-289e-45d1-9df6-1255c0cdd608"


@pytest.mark.asyncio
async def test_generate_docx_sanitizes_arabic_filename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("FILES_SERVICE_BASE_URL", "https://gateway.moeenlaw.com")
    monkeypatch.setenv(
        "FILES_SERVICE_UPLOAD_URL_PATH", "/files/api/v1/files/upload-url"
    )
    monkeypatch.setenv("FILES_SERVICE_UPLOAD_BUCKET", "AI_DOCUMENTS")

    service = HTTPFileGenerationService()
    upload_url_endpoint = "https://gateway.moeenlaw.com/files/api/v1/files/upload-url"
    upload_url = "https://blob.moeenlaw.com/ai-documents/file-789/generated_document.docx?sig=abc"

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == upload_url_endpoint and request.method == "POST":
            payload = request.read().decode("utf-8")
            assert '"fileName":"generated_document.docx"' in payload
            return httpx.Response(
                200,
                json={
                    "fileId": "8820811d-289e-45d1-9df6-1255c0cdd609",
                    "uploadUrl": upload_url,
                    "httpMethod": "PUT",
                },
            )

        if str(request.url) == upload_url and request.method == "PUT":
            return httpx.Response(200)

        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    service._build_client = lambda: httpx.AsyncClient(transport=transport, timeout=1.0)  # type: ignore[method-assign]

    result = await service.generate_docx(
        FileGenerationRequest(
            source_prompt="Generate",
            content="abc",
            filename="عقد عمل.docx",
        )
    )

    assert result.file_id == "8820811d-289e-45d1-9df6-1255c0cdd609"
