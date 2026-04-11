"""Unit tests for Files service retrieval adapter."""

import httpx
import pytest

from app.infrastructure.adapters.file_service_adapter import HTTPFileService
from app.infrastructure.config.settings import get_settings
from app.shared.errors.exceptions import FilesServiceError


@pytest.mark.asyncio
async def test_fetch_file_uses_metadata_then_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("FILES_SERVICE_BASE_URL", "https://gateway.moeenlaw.com")
    monkeypatch.setenv(
        "FILES_SERVICE_DOWNLOAD_PATH_TEMPLATE", "/files/api/v1/files/{file_id}"
    )
    monkeypatch.setenv("FILES_SERVICE_AUTH_TOKEN", "token-123")

    service = HTTPFileService()

    file_id = "e0717f22-cf4c-4e76-83d1-3d63128342d0"
    metadata_url = f"https://gateway.moeenlaw.com/files/api/v1/files/{file_id}"
    download_url = "https://blob.moeenlaw.com/user-documents/sample.pdf?sig=abc"

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == metadata_url:
            assert request.headers.get("Authorization") == "Bearer token-123"
            return httpx.Response(
                200,
                json={
                    "fileId": file_id,
                    "originalName": "sample.pdf",
                    "contentType": "application/pdf",
                    "status": "AVAILABLE",
                    "downloadUrl": download_url,
                },
            )

        if str(request.url) == download_url:
            return httpx.Response(
                200,
                content=b"%PDF-1.4 test",
                headers={"content-type": "application/pdf"},
            )

        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    service._build_client = lambda: httpx.AsyncClient(transport=transport, timeout=1.0)  # type: ignore[method-assign]

    result = await service.fetch_file(file_id)

    assert result.file_id == file_id
    assert result.filename == "sample.pdf"
    assert result.content_type == "application/pdf"
    assert result.content.startswith(b"%PDF")


@pytest.mark.asyncio
async def test_fetch_file_raises_when_metadata_missing_download_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("FILES_SERVICE_BASE_URL", "https://gateway.moeenlaw.com")
    monkeypatch.setenv(
        "FILES_SERVICE_DOWNLOAD_PATH_TEMPLATE", "/files/api/v1/files/{file_id}"
    )

    service = HTTPFileService()

    file_id = "e0717f22-cf4c-4e76-83d1-3d63128342d0"
    metadata_url = f"https://gateway.moeenlaw.com/files/api/v1/files/{file_id}"

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == metadata_url:
            return httpx.Response(
                200,
                json={
                    "fileId": file_id,
                    "status": "AVAILABLE",
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    service._build_client = lambda: httpx.AsyncClient(transport=transport, timeout=1.0)  # type: ignore[method-assign]

    with pytest.raises(FilesServiceError):
        await service.fetch_file(file_id)
