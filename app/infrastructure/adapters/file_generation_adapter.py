"""
Mock File Generation Adapter

Generates DOCX bytes locally and returns a dummy file ID until
Files service upload contract is finalized.
"""

from datetime import datetime, timezone
from io import BytesIO
from uuid import uuid4

from docx import Document as DocxDocument

from app.interfaces.external.file_generation_service import (
    FileGenerationRequest,
    FileGenerationServiceInterface,
    GeneratedFile,
)
from app.shared.errors.exceptions import FileGenerationError


class MockFileGenerationService(FileGenerationServiceInterface):
    """Mock DOCX generation service with deterministic dummy IDs."""

    DUMMY_PREFIX = "mock_file_"

    async def generate_docx(self, request: FileGenerationRequest) -> GeneratedFile:
        try:
            document = DocxDocument()
            document.add_heading(request.filename, level=1)
            document.add_paragraph(request.content)
            document.add_paragraph(
                f"Generated at: {datetime.now(timezone.utc).isoformat()}"
            )

            buffer = BytesIO()
            document.save(buffer)
            docx_bytes = buffer.getvalue()

            return GeneratedFile(
                file_id=f"{self.DUMMY_PREFIX}{uuid4().hex}",
                filename=request.filename,
                content_type=request.content_type,
                size_bytes=len(docx_bytes),
            )
        except Exception as exc:
            raise FileGenerationError(
                message="Failed to generate DOCX artifact",
                details={"filename": request.filename, "error": str(exc)},
            ) from exc

    async def health_check(self) -> bool:
        return True


file_generation_service = MockFileGenerationService()
