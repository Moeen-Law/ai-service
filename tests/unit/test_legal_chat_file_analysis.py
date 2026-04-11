"""Unit tests for LEGAL_CHAT workflow with uploaded files analysis."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pytest

from app.core.domain.entities import Context
from app.core.domain.enums import Intent, Jurisdiction, Language
from app.core.services.file_text_extractor import ExtractedFileText, FileTextExtractor
from app.core.services.query_pipeline import PipelineResult
from app.core.workflows.legal_chat import LegalChatWorkflow
from app.interfaces.ai.llm_service import LLMRequest, LLMResponse, LLMServiceInterface
from app.interfaces.ai.prompt_service import (
    AssembledPrompt,
    PromptServiceInterface,
    PromptTemplate,
)
from app.interfaces.ai.rag_service import (
    Document,
    RAGQuery,
    RAGResponse,
    RAGServiceInterface,
)
from app.interfaces.external.file_service import (
    ExternalFile,
    ExternalFileServiceInterface,
)
from app.interfaces.external.file_generation_service import (
    FileGenerationRequest,
    FileGenerationServiceInterface,
    GeneratedFile,
)
from app.shared.errors.exceptions import FileExtractionError, PayloadValidationError


class FakeLLMService(LLMServiceInterface):
    def __init__(
        self,
        response_text: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._response_text = response_text
        self._tool_calls = tool_calls or []
        self.last_prompt: Optional[str] = None
        self.last_tool_prompt: Optional[str] = None
        self.last_tool_invocations: List[Dict[str, Any]] = []
        self.last_stream_prompt: Optional[str] = None

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.last_prompt = request.prompt
        return LLMResponse(
            content=self._response_text,
            model="fake-llm",
            tokens_used=12,
            finish_reason="stop",
        )

    async def generate_with_context(
        self,
        prompt: str,
        context: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        return await self.generate(LLMRequest(prompt=prompt))

    async def astream(self, prompt: str):
        self.last_stream_prompt = prompt
        for token in ["tok1", "tok2"]:
            yield token

    async def generate_with_tools(
        self,
        request: LLMRequest,
        tools: List[Any],
        max_iterations: int = 6,
    ) -> LLMResponse:
        self.last_tool_prompt = request.prompt
        invocations: List[Dict[str, Any]] = []

        for call in self._tool_calls:
            tool_name = str(call.get("name", "")).strip()
            if not tool_name:
                continue

            tool_obj = next((tool for tool in tools if tool.name == tool_name), None)
            if tool_obj is None:
                continue

            args = call.get("args", {})
            if not isinstance(args, dict):
                args = {}
            result = await tool_obj.ainvoke(args)
            invocations.append({"name": tool_name, "args": args, "result": result})

        self.last_tool_invocations = invocations

        return LLMResponse(
            content=self._response_text,
            model="fake-llm",
            tokens_used=12,
            finish_reason="stop",
            metadata={
                "tool_calls_count": len(invocations),
                "tool_invocations": invocations,
            },
        )

    async def health_check(self) -> bool:
        return True


class FakePromptService(PromptServiceInterface):
    async def get_template(
        self,
        task_type: str,
        jurisdiction: str,
        language: str,
    ) -> PromptTemplate:
        return PromptTemplate(
            id="test",
            name="test",
            template="Context:\n{context}\nQuestion:\n{question}",
            variables=["context", "question"],
        )

    async def assemble_prompt(
        self,
        template: PromptTemplate,
        variables: Dict[str, Any],
        context_documents: Optional[List[str]] = None,
    ) -> AssembledPrompt:
        prompt = template.template.replace(
            "{context}", str(variables.get("context", ""))
        )
        prompt = prompt.replace("{question}", str(variables.get("question", "")))
        return AssembledPrompt(
            prompt=prompt,
            system_prompt=None,
            template_id=template.id,
            template_version=template.version,
        )

    async def get_system_prompt(
        self, task_type: str, jurisdiction: str, language: str
    ) -> str:
        return "system"

    async def health_check(self) -> bool:
        return True


class FakeRAGService(RAGServiceInterface):
    async def initialize(self) -> None:
        return None

    async def retrieve(self, query: RAGQuery) -> RAGResponse:
        return RAGResponse(documents=[], query=query.query, total_found=0)

    async def hybrid_retrieve(
        self,
        clean_query: str,
        expanded_query: str,
        domain: Optional[str] = None,
        k: int = 15,
    ) -> List[Document]:
        return []

    def lookup_article(
        self, article_number: str, domain: Optional[str] = None
    ) -> List[Document]:
        return []

    def detect_domain_semantic(
        self, question: str, threshold: float = 0.6
    ) -> Optional[str]:
        return None

    async def retrieve_by_ids(self, document_ids: List[str]) -> List[Document]:
        return []

    async def search_similar(
        self, text: str, jurisdiction: str, top_k: int = 5
    ) -> List[Document]:
        return []

    async def health_check(self) -> bool:
        return True


class FakeFileService(ExternalFileServiceInterface):
    def __init__(self, files: List[ExternalFile]) -> None:
        self._files = files

    async def fetch_file(self, file_id: str) -> ExternalFile:
        return self._files[0]

    async def fetch_files(self, file_ids: List[str]) -> List[ExternalFile]:
        return self._files

    async def health_check(self) -> bool:
        return True


class FakeFileGenerationService(FileGenerationServiceInterface):
    async def generate_docx(self, request: FileGenerationRequest) -> GeneratedFile:
        return GeneratedFile(
            file_id="mock_file_123",
            filename=request.filename,
            content_type=request.content_type,
            size_bytes=max(1, len(request.content.encode("utf-8"))),
        )

    async def health_check(self) -> bool:
        return True


def _decode_sse_json_events(raw_events: List[str]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for event in raw_events:
        if not event.startswith("data: "):
            continue
        data = event[len("data: ") :].strip()
        if data == "[DONE]":
            continue
        payloads.append(json.loads(data))
    return payloads


@dataclass
class FakeExtractor(FileTextExtractor):
    extracted_files: List[ExtractedFileText]

    def __init__(self, extracted_files: List[ExtractedFileText]) -> None:
        self.extracted_files = extracted_files

    def extract_many(self, files: List[ExternalFile]) -> List[ExtractedFileText]:
        return self.extracted_files


@pytest.fixture
def egypt_context() -> Context:
    return Context(
        jurisdiction=Jurisdiction.EGYPT,
        language=Language.ARABIC,
    )


@pytest.mark.asyncio
async def test_legal_chat_files_ids_bypass_chitchat_shortcut(
    monkeypatch: pytest.MonkeyPatch,
    egypt_context: Context,
) -> None:
    llm = FakeLLMService(
        response_text="file analysis response",
        tool_calls=[
            {"name": "get_uploaded_files_content", "args": {"file_ids": ["f1"]}}
        ],
    )
    workflow = LegalChatWorkflow(
        rag_service=FakeRAGService(),
        llm_service=llm,
        prompt_service=FakePromptService(),
        file_service=FakeFileService(
            files=[
                ExternalFile(
                    file_id="f1",
                    filename="notes.txt",
                    content_type="text/plain",
                    content=b"sample",
                )
            ]
        ),
        file_text_extractor=FakeExtractor(
            extracted_files=[
                ExtractedFileText(
                    file_id="f1",
                    filename="notes.txt",
                    content_type="text/plain",
                    text="important file content",
                )
            ]
        ),
    )

    async def fake_pipeline_run(**_: Any) -> PipelineResult:
        return PipelineResult(
            context="",
            sources=[],
            preferred_domain=None,
            question_numbers=[],
        )

    monkeypatch.setattr(workflow._pipeline, "run", fake_pipeline_run)

    result = await workflow.execute(
        task_id=str(uuid4()),
        context=egypt_context,
        payload={
            "message": "hello",
            "files_ids": ["f1"],
        },
    )

    assert result.data["message"] == "file analysis response"
    assert llm.last_tool_prompt is not None
    assert llm.last_tool_invocations
    tool_result = llm.last_tool_invocations[0]["result"]
    assert "files_context" in tool_result


@pytest.mark.asyncio
async def test_legal_chat_without_files_returns_llm_response(
    monkeypatch: pytest.MonkeyPatch,
    egypt_context: Context,
) -> None:
    llm = FakeLLMService(response_text="assistant response")
    workflow = LegalChatWorkflow(
        rag_service=FakeRAGService(),
        llm_service=llm,
        prompt_service=FakePromptService(),
    )

    async def fake_pipeline_run(**_: Any) -> PipelineResult:
        return PipelineResult(
            context="legal context",
            sources=[],
            preferred_domain="civil",
            question_numbers=[],
        )

    monkeypatch.setattr(workflow._pipeline, "run", fake_pipeline_run)

    result = await workflow.execute(
        task_id=str(uuid4()),
        context=egypt_context,
        payload={"message": "شكرا"},
    )

    assert result.data["intent"] == Intent.LEGAL_QUERY.value
    assert result.data["message"] == "assistant response"
    assert result.metadata.model_used == "fake-llm"


@pytest.mark.asyncio
async def test_legal_chat_files_extraction_error_is_payload_validation(
    monkeypatch: pytest.MonkeyPatch,
    egypt_context: Context,
) -> None:
    class FailingExtractor(FileTextExtractor):
        def __init__(self) -> None:
            pass

        def extract_many(self, files: List[ExternalFile]) -> List[ExtractedFileText]:
            raise FileExtractionError(message="bad file", details={"file_id": "f1"})

    workflow = LegalChatWorkflow(
        rag_service=FakeRAGService(),
        llm_service=FakeLLMService(
            response_text="unused",
            tool_calls=[
                {
                    "name": "get_uploaded_files_content",
                    "args": {"file_ids": ["f1"]},
                }
            ],
        ),
        prompt_service=FakePromptService(),
        file_service=FakeFileService(
            files=[
                ExternalFile(
                    file_id="f1",
                    filename="bad.bin",
                    content_type="application/octet-stream",
                    content=b"x",
                )
            ]
        ),
        file_text_extractor=FailingExtractor(),
    )

    async def fake_pipeline_run(**_: Any) -> PipelineResult:
        return PipelineResult(
            context="some legal context",
            sources=[],
            preferred_domain="civil",
            question_numbers=[],
        )

    monkeypatch.setattr(workflow._pipeline, "run", fake_pipeline_run)

    with pytest.raises(PayloadValidationError) as exc_info:
        await workflow.execute(
            task_id=str(uuid4()),
            context=egypt_context,
            payload={
                "message": "analyze",
                "files_ids": ["f1"],
            },
        )

    assert exc_info.value.code == "PAYLOAD_VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_legal_chat_generation_prompt_returns_content_and_files_ids(
    monkeypatch: pytest.MonkeyPatch,
    egypt_context: Context,
) -> None:
    llm = FakeLLMService(
        response_text="Generated contract text",
        tool_calls=[
            {
                "name": "generate_docx",
                "args": {
                    "document_content": "Generated contract text",
                    "filename": "generated_contract.docx",
                },
            }
        ],
    )
    workflow = LegalChatWorkflow(
        rag_service=FakeRAGService(),
        llm_service=llm,
        prompt_service=FakePromptService(),
        file_generation_service=FakeFileGenerationService(),
    )

    async def fake_pipeline_run(**_: Any) -> PipelineResult:
        return PipelineResult(
            context="relevant legal context",
            sources=[{"id": "src1"}],
            preferred_domain="civil",
            question_numbers=[],
        )

    monkeypatch.setattr(workflow._pipeline, "run", fake_pipeline_run)

    result = await workflow.execute(
        task_id=str(uuid4()),
        context=egypt_context,
        payload={
            "message": "Generate a contract for apartment rental between two parties",
        },
    )

    assert result.data["intent"] == "document_generation"
    assert result.data["format"] == "docx"
    assert result.data["document_content"] == "Generated contract text"
    assert result.data["message"] == "Generated contract text"
    assert result.data["files_ids"] == ["mock_file_123"]


@pytest.mark.asyncio
async def test_stream_legal_chat_with_files_ids_includes_uploaded_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeLLMService(
        response_text="analysis answer",
        tool_calls=[
            {"name": "get_uploaded_files_content", "args": {"file_ids": ["f1"]}}
        ],
    )
    workflow = LegalChatWorkflow(
        rag_service=FakeRAGService(),
        llm_service=llm,
        prompt_service=FakePromptService(),
        file_service=FakeFileService(
            files=[
                ExternalFile(
                    file_id="f1",
                    filename="notes.txt",
                    content_type="text/plain",
                    content=b"sample",
                )
            ]
        ),
        file_text_extractor=FakeExtractor(
            extracted_files=[
                ExtractedFileText(
                    file_id="f1",
                    filename="notes.txt",
                    content_type="text/plain",
                    text="uploaded content",
                )
            ]
        ),
    )

    async def fake_pipeline_run(**_: Any) -> PipelineResult:
        return PipelineResult(
            context="legal context",
            sources=[],
            preferred_domain="civil",
            question_numbers=[],
        )

    monkeypatch.setattr(workflow._pipeline, "run", fake_pipeline_run)

    raw_events = []
    async for event in workflow.stream(
        question="what does this mean",
        files_ids=["f1"],
        jurisdiction="egypt",
        language="ar",
    ):
        raw_events.append(event)

    assert llm.last_tool_prompt is not None
    assert llm.last_tool_invocations
    assert llm.last_tool_invocations[0]["name"] == "get_uploaded_files_content"
    assert any(event.strip() == "data: [DONE]" for event in raw_events)


@pytest.mark.asyncio
async def test_stream_generation_prompt_emits_generation_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = LegalChatWorkflow(
        rag_service=FakeRAGService(),
        llm_service=FakeLLMService(
            response_text="Generated contract text",
            tool_calls=[
                {
                    "name": "generate_docx",
                    "args": {
                        "document_content": "Generated contract text",
                        "filename": "generated_contract.docx",
                    },
                }
            ],
        ),
        prompt_service=FakePromptService(),
        file_generation_service=FakeFileGenerationService(),
    )

    async def fake_pipeline_run(**_: Any) -> PipelineResult:
        return PipelineResult(
            context="legal context",
            sources=[],
            preferred_domain="civil",
            question_numbers=[],
        )

    monkeypatch.setattr(workflow._pipeline, "run", fake_pipeline_run)

    raw_events = []
    async for event in workflow.stream(
        question="Generate a contract for apartment rental",
        jurisdiction="egypt",
        language="ar",
    ):
        raw_events.append(event)

    payloads = _decode_sse_json_events(raw_events)
    generation_events = [p for p in payloads if p.get("type") == "generation"]

    assert generation_events
    assert generation_events[0]["files_ids"] == ["mock_file_123"]
    assert generation_events[0]["format"] == "docx"
