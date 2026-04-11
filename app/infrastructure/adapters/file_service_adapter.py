"""
Files Service HTTP Adapter

Production implementation for retrieving uploaded files as raw binary
from the external Files service using file IDs.
"""

from typing import Dict, List, Optional

import httpx

from app.infrastructure.config.settings import get_settings
from app.interfaces.external.file_service import (
    ExternalFile,
    ExternalFileServiceInterface,
)
from app.shared.errors.exceptions import FilesServiceError


class HTTPFileService(ExternalFileServiceInterface):
    """HTTP-based implementation of external file retrieval."""

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.FILES_SERVICE_BASE_URL.rstrip("/")
        self._download_path_template = settings.FILES_SERVICE_DOWNLOAD_PATH_TEMPLATE
        self._auth_token = settings.FILES_SERVICE_AUTH_TOKEN
        self._timeout = settings.FILES_SERVICE_TIMEOUT_SECONDS
        self._verify_tls = settings.FILES_SERVICE_VERIFY_TLS

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        return headers

    def _build_file_url(self, file_id: str) -> str:
        if not self._base_url:
            raise FilesServiceError(
                message="FILES_SERVICE_BASE_URL is not configured",
                details={"field": "FILES_SERVICE_BASE_URL"},
            )

        path = self._download_path_template.format(file_id=file_id)
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self._base_url}{path}"

    @staticmethod
    def _extract_filename(content_disposition: Optional[str], fallback_id: str) -> str:
        """Best-effort filename extraction from Content-Disposition header."""
        if not content_disposition:
            return fallback_id

        parts = [p.strip() for p in content_disposition.split(";")]
        for part in parts:
            if part.lower().startswith("filename="):
                value = part.split("=", 1)[1].strip().strip('"')
                return value or fallback_id
        return fallback_id

    async def fetch_file(self, file_id: str) -> ExternalFile:
        """Fetch one file by ID and return raw binary content + metadata."""
        url = self._build_file_url(file_id)

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                verify=self._verify_tls,
            ) as client:
                response = await client.get(url, headers=self._build_headers())

            if response.status_code >= 400:
                raise FilesServiceError(
                    message="Files service returned an error response",
                    details={
                        "file_id": file_id,
                        "status_code": response.status_code,
                    },
                )

            content = response.content
            if not content:
                raise FilesServiceError(
                    message="Retrieved file is empty",
                    details={"file_id": file_id},
                )

            filename = self._extract_filename(
                response.headers.get("content-disposition"),
                fallback_id=file_id,
            )

            return ExternalFile(
                file_id=file_id,
                content=content,
                filename=filename,
                content_type=response.headers.get("content-type"),
            )

        except FilesServiceError:
            raise
        except httpx.TimeoutException as exc:
            raise FilesServiceError(
                message="Files service request timed out",
                details={"file_id": file_id, "url": url},
            ) from exc
        except httpx.HTTPError as exc:
            raise FilesServiceError(
                message="Files service HTTP request failed",
                details={"file_id": file_id, "url": url},
            ) from exc

    async def fetch_files(self, file_ids: List[str]) -> List[ExternalFile]:
        """Fetch multiple files while preserving the provided order."""
        files: List[ExternalFile] = []
        for file_id in file_ids:
            files.append(await self.fetch_file(file_id))
        return files

    async def health_check(self) -> bool:
        """Best-effort health check for Files service connectivity."""
        if not self._base_url:
            return False

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                verify=self._verify_tls,
            ) as client:
                response = await client.get(
                    self._base_url, headers=self._build_headers()
                )
            return response.status_code < 500
        except Exception:
            return False


file_service = HTTPFileService()
