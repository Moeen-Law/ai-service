"""Unit tests for CONTRACT_ANALYSIS workflow with uploaded files."""

from typing import Any, Dict, List, Optional
from uuid import uuid4

import pytest

from app.core.domain.entities import Context
from app.core.domain.enums import Jurisdiction, Language
from app.core.services.file_text_extractor import ExtractedFileText, FileTextExtractor
from app.core.services.query_pipeline import PipelineResult
from app.core.workflows.contract_analysis import ContractAnalysisWorkflow
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
from app.shared.errors.exceptions import (
    FileExtractionError,
    FilesServiceError,
    PayloadValidationError,
)


class FakeLLMService(LLMServiceInterface):
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.last_request: Optional[LLMRequest] = None

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.last_request = request
        return LLMResponse(
            content=self._response_text,
            model="fake-llm",
            tokens_used=42,
            finish_reason="stop",
        )

    async def generate_with_context(
        self,
        prompt: str,
        context: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        return await self.generate(
            LLMRequest(prompt=prompt, system_prompt=system_prompt)
        )

    async def astream(self, prompt: str, system_prompt: Optional[str] = None):
        yield "token"

    async def generate_with_tools(
        self,
        request: LLMRequest,
        tools: List[Any],
        max_iterations: int = 6,
    ) -> LLMResponse:
        return await self.generate(request)

    async def stream_with_tools(
        self,
        request: LLMRequest,
        tools: List[Any],
        max_iterations: int = 6,
    ):
        yield "", True, {"tool_invocations": []}

    async def health_check(self) -> bool:
        return True


class FakePromptService(PromptServiceInterface):
    def __init__(self) -> None:
        self.last_variables: Optional[Dict[str, Any]] = None

    async def get_template(
        self,
        task_type: str,
        jurisdiction: str,
        language: str,
    ) -> PromptTemplate:
        return PromptTemplate(
            id="contract-analysis-test",
            name="Contract Analysis Test",
            template="Context:\n{context}\n\nContract:\n{contract_text}",
            variables=["context", "contract_text"],
        )

    async def assemble_prompt(
        self,
        template: PromptTemplate,
        variables: Dict[str, Any],
        context_documents: Optional[List[str]] = None,
    ) -> AssembledPrompt:
        self.last_variables = variables
        prompt = template.template.replace(
            "{context}", str(variables.get("context", ""))
        )
        prompt = prompt.replace(
            "{contract_text}", str(variables.get("contract_text", ""))
        )
        return AssembledPrompt(
            prompt=prompt,
            system_prompt="contract-analysis-system",
            template_id=template.id,
            template_version=template.version,
        )

    async def get_system_prompt(
        self,
        task_type: str,
        jurisdiction: str,
        language: str,
    ) -> str:
        return "contract-analysis-system"

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
        self,
        article_number: str,
        domain: Optional[str] = None,
    ) -> List[Document]:
        return []

    def detect_domain_semantic(
        self,
        question: str,
        threshold: float = 0.6,
    ) -> Optional[str]:
        return None

    async def retrieve_by_ids(self, document_ids: List[str]) -> List[Document]:
        return []

    async def search_similar(
        self,
        text: str,
        jurisdiction: str,
        top_k: int = 5,
    ) -> List[Document]:
        return []

    async def health_check(self) -> bool:
        return True


class FakeFileService(ExternalFileServiceInterface):
    def __init__(self, files: List[ExternalFile]) -> None:
        self._files = files
        self.last_file_ids: Optional[List[str]] = None

    async def fetch_file(self, file_id: str) -> ExternalFile:
        return self._files[0]

    async def fetch_files(self, file_ids: List[str]) -> List[ExternalFile]:
        self.last_file_ids = file_ids
        return self._files

    async def health_check(self) -> bool:
        return True


class FakeExtractor(FileTextExtractor):
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
async def test_contract_analysis_with_files_ids_success(
    monkeypatch: pytest.MonkeyPatch,
    egypt_context: Context,
) -> None:
    llm = FakeLLMService(
        response_text='{"risks":[{"clause":"البند 3","risk_level":"high","description":"شرط جزائي مبالغ"}],"summary":"تحليل شامل","recommendations":["تعديل الشرط الجزائي"]}'
    )
    prompt_service = FakePromptService()
    file_service = FakeFileService(
        files=[
            ExternalFile(
                file_id="file_1",
                filename="contract.txt",
                content_type="text/plain",
                content=b"contract",
            )
        ]
    )

    workflow = ContractAnalysisWorkflow(
        rag_service=FakeRAGService(),
        llm_service=llm,
        prompt_service=prompt_service,
        file_service=file_service,
        file_text_extractor=FakeExtractor(
            extracted_files=[
                ExtractedFileText(
                    file_id="file_1",
                    filename="contract.txt",
                    content_type="text/plain",
                    text="uploaded contract full content",
                )
            ]
        ),
    )

    async def fake_pipeline_run(**_: Any) -> PipelineResult:
        return PipelineResult(
            context="legal articles context",
            sources=[{"metadata": {"article_number": "10"}}],
            preferred_domain="commercial",
            question_numbers=[],
        )

    monkeypatch.setattr(workflow._pipeline, "run", fake_pipeline_run)

    result = await workflow.execute(
        task_id=str(uuid4()),
        context=egypt_context,
        payload={"files_ids": ["file_1", "file_1", " "]},
    )

    assert result.is_success
    assert file_service.last_file_ids == ["file_1"]
    assert result.data["summary"] == "تحليل شامل"
    assert result.data["recommendations"] == ["تعديل الشرط الجزائي"]
    assert result.data["sources"] == [{"metadata": {"article_number": "10"}}]
    assert llm.last_request is not None
    assert "uploaded contract full content" in llm.last_request.prompt


@pytest.mark.asyncio
async def test_contract_analysis_files_without_file_service_raises(
    egypt_context: Context,
) -> None:
    workflow = ContractAnalysisWorkflow(
        rag_service=FakeRAGService(),
        llm_service=FakeLLMService(response_text="{}"),
        prompt_service=FakePromptService(),
        file_service=None,
    )

    with pytest.raises(FilesServiceError):
        await workflow.execute(
            task_id=str(uuid4()),
            context=egypt_context,
            payload={"files_ids": ["file_1"]},
        )


@pytest.mark.asyncio
async def test_contract_analysis_file_extraction_error_maps_to_payload_error(
    monkeypatch: pytest.MonkeyPatch,
    egypt_context: Context,
) -> None:
    class FailingExtractor(FileTextExtractor):
        def __init__(self) -> None:
            pass

        def extract_many(self, files: List[ExternalFile]) -> List[ExtractedFileText]:
            raise FileExtractionError(
                message="bad file",
                details={"file_id": "file_1"},
            )

    workflow = ContractAnalysisWorkflow(
        rag_service=FakeRAGService(),
        llm_service=FakeLLMService(response_text="{}"),
        prompt_service=FakePromptService(),
        file_service=FakeFileService(
            files=[
                ExternalFile(
                    file_id="file_1",
                    filename="contract.pdf",
                    content_type="application/pdf",
                    content=b"%PDF",
                )
            ]
        ),
        file_text_extractor=FailingExtractor(),
    )

    async def fake_pipeline_run(**_: Any) -> PipelineResult:
        return PipelineResult(
            context="",
            sources=[],
            preferred_domain=None,
            question_numbers=[],
        )

    monkeypatch.setattr(workflow._pipeline, "run", fake_pipeline_run)

    with pytest.raises(PayloadValidationError) as exc_info:
        await workflow.execute(
            task_id=str(uuid4()),
            context=egypt_context,
            payload={"files_ids": ["file_1"]},
        )

    assert "Unable to process one or more uploaded files" in str(exc_info.value)
