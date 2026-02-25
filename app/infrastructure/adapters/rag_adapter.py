"""
Qdrant RAG Service Adapter

Production implementation of the RAG service using Qdrant vector DB
with hybrid retrieval (vector + BM25) and cross-domain article caching.
"""

import re
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document as LCDocument
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from app.infrastructure.config.settings import get_settings
from app.infrastructure.logging.logger import get_logger
from app.interfaces.ai.rag_service import (
    Document,
    RAGQuery,
    RAGResponse,
    RAGServiceInterface,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain prototype sentences for semantic classification
# ---------------------------------------------------------------------------
_DOMAIN_PROTOTYPES: Dict[str, str] = {
    "penal": "جريمة عقوبة سجن حبس غرامة إعدام قتل سرقة اغتصاب احتيال رشوة تزوير ضرب اعتداء",
    "civil": "عقد التزام تعويض ضرر ملكية إيجار بيع شراء ميراث مسئولية مدنية فسخ بطلان",
    "labor": "عقد عمل أجر مرتب إجازة فصل تعسفي نقابة عمال صاحب عمل ساعات عمل مكافأة نهاية خدمة",
    "constitution": "حق حرية دستور مساواة كرامة مواطن سلطة تشريعية قضائية تنفيذية ضمانات دستورية",
    "commercial": "شركة تجارة عقد تجاري إفلاس شيك كمبيالة سند أمر سجل تجاري إفلاس تصفية",
    "criminal_procedure": "محكمة نيابة تحقيق قضاء دعوى جنائية استئناف نقض توقيف حبس احتياطي إجراءات",
}


def _lc_to_domain(doc: LCDocument, score: float = 0.0) -> Document:
    """Convert a LangChain Document to the domain Document dataclass."""
    meta = doc.metadata or {}
    art = str(meta.get("article_number", ""))
    domain = meta.get("domain", "")
    return Document(
        id=f"{art}|{domain}" if art else str(id(doc)),
        content=doc.page_content,
        source=meta.get("law_name_ar", meta.get("law_name", "قانون مصري")),
        score=score,
        metadata=meta,
    )


class QdrantRAGService(RAGServiceInterface):
    """
    Production RAG service backed by Qdrant + BM25 hybrid retrieval.

    On ``initialize()`` the service:
    1. Scrolls **all** documents from Qdrant into memory.
    2. Builds a cross-domain article cache for O(1) article look-ups.
    3. Initialises a BM25 retriever for lexical matching.
    4. Pre-computes domain prototype embeddings for semantic classification.
    """

    def __init__(self) -> None:
        settings = get_settings()

        # Qdrant client
        self._client = QdrantClient(
            url=settings.QDRANT_URL,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
        )
        self._collection = settings.QDRANT_COLLECTION_NAME

        # Embeddings
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
        )

        # Weights
        self._vector_weight = settings.VECTOR_WEIGHT
        self._bm25_weight = settings.BM25_WEIGHT
        self._retrieval_k = settings.RETRIEVAL_K

        # Populated on initialize()
        self._all_docs: List[LCDocument] = []
        self._article_cache: Dict[str, Dict[str, LCDocument]] = {}
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._domain_proto_vecs: Dict[str, List[float]] = {}
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load all documents, build caches & indexes. Call once at startup."""
        if self._initialized:
            return

        logger.info("RAG service initializing — loading documents from Qdrant …")
        self._all_docs = self._scroll_all_documents()
        logger.info("rag_docs_loaded", count=len(self._all_docs))

        self._build_article_cache()
        self._init_bm25()
        self._compute_domain_prototypes()

        self._initialized = True
        logger.info("RAG service initialised successfully")

    # ------------------------------------------------------------------
    # Interface: basic retrieve
    # ------------------------------------------------------------------

    async def retrieve(self, query: RAGQuery) -> RAGResponse:
        docs = await self.hybrid_retrieve(
            clean_query=query.query,
            expanded_query=query.query,
            domain=query.domain,
            k=query.top_k,
        )
        return RAGResponse(
            documents=docs,
            query=query.query,
            total_found=len(docs),
        )

    # ------------------------------------------------------------------
    # Interface: hybrid retrieval
    # ------------------------------------------------------------------

    async def hybrid_retrieve(
        self,
        clean_query: str,
        expanded_query: str,
        domain: Optional[str] = None,
        k: int = 15,
    ) -> List[Document]:
        fetch_k = max(k * 3, 15)

        vector_results = self._vector_search(clean_query, k=fetch_k)
        bm25_results = self._bm25_search(expanded_query, k=fetch_k)

        logger.debug(
            "hybrid_search",
            vector=len(vector_results),
            bm25=len(bm25_results),
        )

        merged = self._rrf_merge(
            [vector_results, bm25_results],
            weights=[self._vector_weight, self._bm25_weight],
        )

        docs: List[Document] = []
        for rank, lc_doc in enumerate(merged):
            score = 1.0 / (60 + rank + 1)
            docs.append(_lc_to_domain(lc_doc, score=score))
        return docs

    # ------------------------------------------------------------------
    # Interface: article lookup
    # ------------------------------------------------------------------

    def lookup_article(
        self,
        article_number: str,
        domain: Optional[str] = None,
    ) -> List[Document]:
        results = self._lookup_article_lc(article_number, domain)
        return [_lc_to_domain(d, score=1.0) for d in results]

    # ------------------------------------------------------------------
    # Interface: semantic domain detection
    # ------------------------------------------------------------------

    def detect_domain_semantic(
        self,
        question: str,
        threshold: float = 0.6,
    ) -> Optional[str]:
        if not self._domain_proto_vecs:
            return None
        try:
            q_vec = np.array(
                self._embeddings.embed_query(f"query: {question}"),
                dtype=float,
            )
            q_norm = float(np.linalg.norm(q_vec))
            if q_norm == 0:
                return None

            best_domain: Optional[str] = None
            best_sim: float = 0.0
            for dom, proto_vec in self._domain_proto_vecs.items():
                p_arr = np.array(proto_vec, dtype=float)
                sim = float(
                    np.dot(q_vec, p_arr)
                    / (q_norm * float(np.linalg.norm(p_arr)) + 1e-10)
                )
                if sim > best_sim:
                    best_sim = sim
                    best_domain = dom

            if best_sim >= threshold:
                logger.debug(
                    "domain_classified", domain=best_domain, sim=round(best_sim, 3)
                )
                return best_domain
            logger.debug("domain_below_threshold", best_sim=round(best_sim, 3))
            return None
        except Exception as exc:
            logger.warning("domain_classification_failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Interface: retrieve_by_ids / search_similar / health_check
    # ------------------------------------------------------------------

    async def retrieve_by_ids(self, document_ids: List[str]) -> List[Document]:
        id_set = set(document_ids)
        results: List[Document] = []
        for lc_doc in self._all_docs:
            art = str(lc_doc.metadata.get("article_number", ""))
            dom = lc_doc.metadata.get("domain", "")
            key = f"{art}|{dom}"
            if key in id_set:
                results.append(_lc_to_domain(lc_doc, score=1.0))
        return results

    async def search_similar(
        self,
        text: str,
        jurisdiction: str,
        top_k: int = 5,
    ) -> List[Document]:
        results = self._vector_search(text, k=top_k)
        return [_lc_to_domain(d, score=0.9) for d in results]

    async def health_check(self) -> bool:
        try:
            info = self._client.get_collection(self._collection)
            return info.points_count > 0
        except Exception:
            return False

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _scroll_all_documents(self) -> List[LCDocument]:
        docs: List[LCDocument] = []
        offset = None
        page = 0
        while True:
            pts, nxt = self._client.scroll(
                collection_name=self._collection,
                limit=2250,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            page += 1
            for pt in pts:
                text = pt.payload.get("text", "")
                if text and len(text.strip()) > 20:
                    docs.append(LCDocument(page_content=text, metadata=pt.payload))
            if nxt is None:
                break
            offset = nxt
        logger.info("scroll_complete", pages=page, docs=len(docs))
        return docs

    def _build_article_cache(self) -> None:
        self._article_cache = {}
        for doc in self._all_docs:
            art = str(doc.metadata.get("article_number", "")).strip()
            dom = (doc.metadata.get("domain") or "unknown").strip()
            if art:
                self._article_cache.setdefault(art, {})[dom] = doc
        total = sum(len(v) for v in self._article_cache.values())
        logger.info(
            "article_cache_built",
            unique_numbers=len(self._article_cache),
            total_entries=total,
        )

    def _init_bm25(self) -> None:
        try:
            self._bm25_retriever = BM25Retriever.from_documents(self._all_docs)
            self._bm25_retriever.k = 20
            logger.info("bm25_ready")
        except Exception as exc:
            logger.warning("bm25_setup_failed", error=str(exc))
            self._bm25_retriever = None

    def _compute_domain_prototypes(self) -> None:
        try:
            for domain, proto in _DOMAIN_PROTOTYPES.items():
                self._domain_proto_vecs[domain] = self._embeddings.embed_query(
                    f"passage: {proto}"
                )
            logger.info(
                "domain_prototypes_ready",
                domains=list(self._domain_proto_vecs.keys()),
            )
        except Exception as exc:
            logger.warning("domain_prototype_failed", error=str(exc))
            self._domain_proto_vecs = {}

    def _vector_search(self, query: str, k: int = 20) -> List[LCDocument]:
        vec = self._embeddings.embed_query(f"query: {query}")
        hits = self._client.query_points(
            collection_name=self._collection,
            query=vec,
            limit=k,
            with_payload=True,
        ).points
        docs: List[LCDocument] = []
        for h in hits:
            text = h.payload.get("text", "")
            if text and len(text.strip()) > 20:
                docs.append(LCDocument(page_content=text, metadata=h.payload))
        return docs

    def _bm25_search(self, query: str, k: int = 20) -> List[LCDocument]:
        if self._bm25_retriever is None:
            return []
        self._bm25_retriever.k = k
        return self._bm25_retriever.invoke(query)

    @staticmethod
    def _rrf_merge(
        result_lists: List[List[LCDocument]],
        weights: List[float],
        rrf_k: int = 60,
    ) -> List[LCDocument]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, LCDocument] = {}
        for results, w in zip(result_lists, weights):
            for rank, doc in enumerate(results):
                art = doc.metadata.get("article_number", "")
                dom = doc.metadata.get("domain", "")
                key = f"{art}|{dom}|{doc.page_content[:80]}"
                if key not in doc_map:
                    doc_map[key] = doc
                    scores[key] = 0.0
                scores[key] += w / (rrf_k + rank + 1)
        return [doc_map[k] for k in sorted(scores, key=scores.get, reverse=True)]

    def _lookup_article_lc(
        self,
        art_num: str,
        domain: Optional[str] = None,
    ) -> List[LCDocument]:
        results: List[LCDocument] = []

        bucket = self._article_cache.get(art_num, {})
        if bucket:
            if domain:
                doc = bucket.get(domain)
                if doc:
                    results.append(doc)
            else:
                results.extend(bucket.values())
            if results:
                return results

        for key, doms in self._article_cache.items():
            clean = key.strip().replace(" ", "")
            if (
                clean == art_num
                or key.startswith(f"{art_num} ")
                or key.startswith(f"{art_num}/")
            ):
                if domain:
                    doc = doms.get(domain)
                    if doc:
                        results.append(doc)
                else:
                    results.extend(doms.values())
        return results


# Default instance — initialised lazily via ``await rag_service.initialize()``
rag_service = QdrantRAGService()
