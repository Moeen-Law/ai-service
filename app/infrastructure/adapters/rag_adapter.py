"""
Stub RAG Service Adapter

Fake implementation of RAG service for testing and development.
Returns canned document results without calling any real vector store.
"""

from typing import Any, Dict, List, Optional

from app.interfaces.ai.rag_service import (
    Document,
    RAGQuery,
    RAGResponse,
    RAGServiceInterface,
)


class StubRAGService(RAGServiceInterface):
    """
    Stub implementation of RAG service.

    Returns deterministic canned documents for testing.
    Can be easily replaced with real implementation later.
    """

    # Sample documents for different jurisdictions
    _SAMPLE_DOCUMENTS: Dict[str, List[Document]] = {
        "egypt": [
            Document(
                id="eg-civil-89",
                content="المادة 89 من القانون المدني المصري: العقد شريعة المتعاقدين...",
                source="القانون المدني المصري",
                score=0.95,
                metadata={"article": 89, "law": "civil_code"},
            ),
            Document(
                id="eg-civil-147",
                content="المادة 147 من القانون المدني: يجب تنفيذ العقد طبقاً لما اشتمل عليه...",
                source="القانون المدني المصري",
                score=0.88,
                metadata={"article": 147, "law": "civil_code"},
            ),
            Document(
                id="eg-commerce-1",
                content="المادة الأولى من قانون التجارة: تسري أحكام هذا القانون على التجار...",
                source="قانون التجارة المصري",
                score=0.82,
                metadata={"article": 1, "law": "commerce_code"},
            ),
        ],
        "uae": [
            Document(
                id="uae-civil-125",
                content="Article 125 of UAE Civil Code: The contract is the law of the parties...",
                source="UAE Civil Code",
                score=0.93,
                metadata={"article": 125, "law": "civil_code"},
            ),
            Document(
                id="uae-commerce-10",
                content="Article 10 of UAE Commercial Transactions Law...",
                source="UAE Commercial Transactions Law",
                score=0.85,
                metadata={"article": 10, "law": "commerce_code"},
            ),
        ],
    }

    async def retrieve(self, query: RAGQuery) -> RAGResponse:
        """Retrieve stubbed documents based on query."""
        jurisdiction = query.jurisdiction.lower()
        documents = self._SAMPLE_DOCUMENTS.get(jurisdiction, [])

        # Filter by score threshold
        filtered_docs = [d for d in documents if d.score >= query.threshold]

        # Limit to top_k
        limited_docs = filtered_docs[: query.top_k]

        return RAGResponse(
            documents=limited_docs,
            query=query.query,
            total_found=len(limited_docs),
            metadata={"stub": True, "jurisdiction": jurisdiction},
        )

    async def retrieve_by_ids(self, document_ids: List[str]) -> List[Document]:
        """Retrieve specific documents by ID."""
        all_docs = []
        for docs in self._SAMPLE_DOCUMENTS.values():
            all_docs.extend(docs)

        return [doc for doc in all_docs if doc.id in document_ids]

    async def search_similar(
        self,
        text: str,
        jurisdiction: str,
        top_k: int = 5,
    ) -> List[Document]:
        """Search for similar documents."""
        documents = self._SAMPLE_DOCUMENTS.get(jurisdiction.lower(), [])
        return documents[:top_k]

    async def health_check(self) -> bool:
        """Always returns True for stub."""
        return True


# Default instance for dependency injection
stub_rag_service = StubRAGService()
