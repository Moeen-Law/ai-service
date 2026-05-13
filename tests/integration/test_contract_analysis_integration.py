"""Integration test for CONTRACT_ANALYSIS using uploaded files.

This test injects fake adapters into `app.infrastructure.adapters` before
creating the FastAPI app so the registry wires workflows with test doubles.
"""

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import app.infrastructure.adapters as adapters
from app.interfaces.external.file_service import ExternalFile
from app.interfaces.ai.llm_service import LLMRequest, LLMResponse, LLMServiceInterface
from app.interfaces.ai.prompt_service import (
    AssembledPrompt,
    PromptServiceInterface,
    PromptTemplate,
)
from app.interfaces.ai.rag_service import (
    RAGServiceInterface,
    RAGResponse,
    RAGQuery,
    Document,
)
from app.interfaces.external.file_service import ExternalFileServiceInterface


class FakeLLMService(LLMServiceInterface):
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.last_request: Optional[LLMRequest] = None

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.last_request = request
        return LLMResponse(
            content=self._response_text,
            model="fake-llm",
            tokens_used=10,
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
        yield "tok"

    async def generate_with_tools(
        self, request: LLMRequest, tools: List[Any], max_iterations: int = 6
    ) -> LLMResponse:
        return await self.generate(request)

    async def stream_with_tools(
        self, request: LLMRequest, tools: List[Any], max_iterations: int = 6
    ):
        yield "", True, {"tool_invocations": []}

    async def health_check(self) -> bool:
        return True


class FakePromptService(PromptServiceInterface):
    async def get_template(
        self, task_type: str, jurisdiction: str, language: str
    ) -> PromptTemplate:
        return PromptTemplate(
            id="t",
            name="t",
            template="{system_prompt}\n{context}\n{contract_text}",
            variables=["system_prompt", "context", "contract_text"],
        )

    async def assemble_prompt(
        self,
        template: PromptTemplate,
        variables: Dict[str, Any],
        context_documents: Optional[List[str]] = None,
    ) -> AssembledPrompt:
        prompt = (
            template.template.replace("{system_prompt}", "system")
            .replace("{context}", str(variables.get("context", "")))
            .replace("{contract_text}", str(variables.get("contract_text", "")))
        )
        return AssembledPrompt(
            prompt=prompt,
            system_prompt="system",
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
        self.last_file_ids = None

    async def fetch_file(self, file_id: str) -> ExternalFile:
        return self._files[0]

    async def fetch_files(self, file_ids: List[str]) -> List[ExternalFile]:
        self.last_file_ids = file_ids
        return self._files

    async def health_check(self) -> bool:
        return True


@pytest.fixture
def client(monkeypatch) -> TestClient:
    # Inject test adapters before app creation so registry picks them up
    fake_llm = FakeLLMService(
        response_text='{"risks":[],"summary":"ok","recommendations":[]}'
    )
    monkeypatch.setattr(adapters, "get_llm_service", lambda: fake_llm)
    monkeypatch.setattr(
        adapters,
        "file_service",
        FakeFileService(
            files=[
                ExternalFile(
                    file_id="file_1",
                    filename="c.txt",
                    content_type="text/plain",
                    content=b"x",
                )
            ]
        ),
    )
    monkeypatch.setattr(adapters, "rag_service", FakeRAGService())
    monkeypatch.setattr(adapters, "prompt_service", FakePromptService())

    from app.api.main import create_app

    app = create_app()
    return TestClient(app)


def test_contract_analysis_post_with_files(client: TestClient) -> None:
    resp = client.post(
        "/v1/ai/tasks",
        json={
            "task_type": "CONTRACT_ANALYSIS",
            "context": {
                "jurisdiction": "EGYPT",
                "language": "ar",
                "domain": "COMMERCIAL",
            },
            "payload": {"files_ids": ["file_1"]},
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["task_type"] == "CONTRACT_ANALYSIS"
    assert data["status"] == "success"
    # CONTRACT_ANALYSIS now returns LEGAL_CHAT-style output: a `message`
    # string plus `sources` and `intent`.
    assert data["result"]["message"].startswith("## تحليل العقد")
    assert data["result"]["sources"] == []
    assert data["result"]["intent"] == "CONTRACT_ANALYSIS"
