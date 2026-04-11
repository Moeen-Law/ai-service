"""
Files Service HTTP Adapter

Production implementation for retrieving uploaded files using metadata
endpoint + presigned download URL flow.
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

    def _build_json_headers(self) -> Dict[str, str]:
        headers = self._build_headers()
        headers["Accept"] = "application/json"
        return headers

    def _build_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self._timeout,
            verify=self._verify_tls,
        )

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
        """Fetch one file metadata then download raw binary content."""
        metadata_url = self._build_file_url(file_id)

        try:
            async with self._build_client() as client:
                metadata_response = await client.get(
                    metadata_url,
                    headers=self._build_json_headers(),
                )

                if metadata_response.status_code >= 400:
                    raise FilesServiceError(
                        message="Files service returned an error response",
                        details={
                            "file_id": file_id,
                            "status_code": metadata_response.status_code,
                            "url": metadata_url,
                        },
                    )

                metadata = metadata_response.json()
                if not isinstance(metadata, dict):
                    raise FilesServiceError(
                        message="Files service metadata response is invalid",
                        details={"file_id": file_id, "url": metadata_url},
                    )

                file_status = str(metadata.get("status") or "").strip().upper()
                if file_status and file_status != "AVAILABLE":
                    raise FilesServiceError(
                        message="Requested file is not available",
                        details={
                            "file_id": file_id,
                            "status": file_status,
                        },
                    )

                download_url = str(metadata.get("downloadUrl") or "").strip()
                if not download_url:
                    raise FilesServiceError(
                        message="Files service did not return downloadUrl",
                        details={"file_id": file_id, "url": metadata_url},
                    )

                download_response = await client.get(download_url)

            if download_response.status_code >= 400:
                raise FilesServiceError(
                    message="File download URL returned an error response",
                    details={
                        "file_id": file_id,
                        "status_code": download_response.status_code,
                    },
                )

            content = download_response.content
            if not content:
                raise FilesServiceError(
                    message="Retrieved file is empty",
                    details={"file_id": file_id},
                )

            filename = str(metadata.get("originalName") or "").strip()
            if not filename:
                filename = self._extract_filename(
                    download_response.headers.get("content-disposition"),
                    fallback_id=file_id,
                )

            content_type = str(metadata.get("contentType") or "").strip() or None
            if content_type is None:
                content_type = download_response.headers.get("content-type")

            return ExternalFile(
                file_id=file_id,
                content=content,
                filename=filename,
                content_type=content_type,
            )

        except FilesServiceError:
            raise
        except httpx.TimeoutException as exc:
            raise FilesServiceError(
                message="Files service request timed out",
                details={"file_id": file_id, "url": metadata_url},
            ) from exc
        except httpx.HTTPError as exc:
            raise FilesServiceError(
                message="Files service HTTP request failed",
                details={"file_id": file_id, "url": metadata_url},
            ) from exc
        except ValueError as exc:
            raise FilesServiceError(
                message="Failed to parse files service response",
                details={"file_id": file_id, "url": metadata_url},
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
            async with self._build_client() as client:
                response = await client.get(
                    self._base_url, headers=self._build_headers()
                )
            return response.status_code < 500
        except Exception:
            return False


file_service = HTTPFileService()
