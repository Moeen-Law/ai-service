"""
Uploaded File Text Extraction Service

Converts binary files from the Files service into normalized plain text
for LLM consumption with deterministic safety limits.
"""

from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional

from docx import Document as DocxDocument
from pypdf import PdfReader

from app.infrastructure.config.settings import get_settings
from app.interfaces.external.file_service import ExternalFile
from app.shared.errors.exceptions import FileExtractionError


@dataclass(frozen=True)
class ExtractedFileText:
    """Normalized extracted text from one uploaded file."""

    file_id: str
    filename: str
    content_type: Optional[str]
    text: str
    truncated: bool = False


class FileTextExtractor:
    """Extract plain text from supported binary file formats."""

    def __init__(self) -> None:
        settings = get_settings()
        self._max_size_bytes = settings.FILES_MAX_SIZE_BYTES
        self._max_chars_per_file = settings.FILES_MAX_EXTRACTED_CHARS_PER_FILE
        self._max_chars_total = settings.FILES_MAX_EXTRACTED_TOTAL_CHARS

    def extract_many(self, files: List[ExternalFile]) -> List[ExtractedFileText]:
        """Extract text from multiple files while enforcing a global text budget."""
        remaining_total = self._max_chars_total
        extracted_files: List[ExtractedFileText] = []

        for file in files:
            if file.size_bytes > self._max_size_bytes:
                raise FileExtractionError(
                    message="Uploaded file exceeds maximum allowed size",
                    details={
                        "file_id": file.file_id,
                        "max_size_bytes": self._max_size_bytes,
                        "actual_size_bytes": file.size_bytes,
                    },
                )

            text = self._extract_file_text(file).strip()
            if not text:
                raise FileExtractionError(
                    message="Uploaded file does not contain extractable text",
                    details={"file_id": file.file_id, "filename": file.filename},
                )

            truncated = False
            if len(text) > self._max_chars_per_file:
                text = text[: self._max_chars_per_file]
                truncated = True

            if remaining_total <= 0:
                raise FileExtractionError(
                    message="Combined extracted text exceeds total prompt budget",
                    details={"max_total_chars": self._max_chars_total},
                )

            if len(text) > remaining_total:
                text = text[:remaining_total]
                truncated = True

            remaining_total -= len(text)
            extracted_files.append(
                ExtractedFileText(
                    file_id=file.file_id,
                    filename=file.filename or file.file_id,
                    content_type=file.content_type,
                    text=text,
                    truncated=truncated,
                )
            )

        return extracted_files

    def _extract_file_text(self, file: ExternalFile) -> str:
        """Extract text from one file based on file type hints."""
        file_type = self._detect_file_type(file)

        if file_type == "pdf":
            return self._extract_pdf_text(file.content)
        if file_type == "docx":
            return self._extract_docx_text(file.content)
        if file_type == "text":
            return self._decode_text_bytes(file.content)

        raise FileExtractionError(
            message="Unsupported file type for analysis",
            details={
                "file_id": file.file_id,
                "filename": file.filename,
                "content_type": file.content_type,
            },
        )

    @staticmethod
    def _detect_file_type(file: ExternalFile) -> str:
        """Infer file type from content-type and filename extension."""
        content_type = (file.content_type or "").lower()
        filename = (file.filename or "").lower()

        if "pdf" in content_type or filename.endswith(".pdf"):
            return "pdf"
        if "wordprocessingml.document" in content_type or filename.endswith(".docx"):
            return "docx"
        if (
            content_type.startswith("text/")
            or "application/json" in content_type
            or filename.endswith(".txt")
            or filename.endswith(".md")
            or filename.endswith(".json")
        ):
            return "text"

        return "unknown"

    @staticmethod
    def _extract_pdf_text(content: bytes) -> str:
        """Extract text from PDF bytes using pypdf."""
        try:
            reader = PdfReader(BytesIO(content))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            return "\n".join(pages)
        except Exception as exc:
            raise FileExtractionError(
                message="Failed to extract text from PDF file",
                details={"error": str(exc)},
            ) from exc

    @staticmethod
    def _extract_docx_text(content: bytes) -> str:
        """Extract text from DOCX bytes using python-docx."""
        try:
            document = DocxDocument(BytesIO(content))
            return "\n".join(paragraph.text for paragraph in document.paragraphs)
        except Exception as exc:
            raise FileExtractionError(
                message="Failed to extract text from DOCX file",
                details={"error": str(exc)},
            ) from exc

    @staticmethod
    def _decode_text_bytes(content: bytes) -> str:
        """Decode plain-text-like bytes with practical encoding fallbacks."""
        for encoding in ("utf-8", "utf-16", "cp1256", "latin-1"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue

        raise FileExtractionError(message="Failed to decode text file bytes")
