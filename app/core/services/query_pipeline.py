"""
Query Pipeline Service

The "agent brain" that orchestrates the full multi-step retrieval + reasoning
pipeline for legal questions.  This is a **domain service** — it depends only
on the interface ports (RAG, LLM, Prompt) and shared utilities, never on
infrastructure details directly.

Pipeline steps (mirroring the original preprocessing ``app.py``):
  0. LLM query rewriting (domain, keywords, likely articles)
  1. Domain detection (LLM primary, semantic fallback)
  2. Article-only shortcut / full hybrid retrieval
  3. Inject articles mentioned in the question from cache
  4. Rerank by article-number match
  5. Domain-priority filtering
  6. Cross-reference injection from retrieved docs
  7. Build context string + source list
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from app.infrastructure.logging.logger import get_logger
from app.interfaces.ai.llm_service import LLMServiceInterface
from app.interfaces.ai.rag_service import Document, RAGServiceInterface
from app.shared.utils.arabic import DOMAIN_NAMES_AR, extract_article_numbers_from_text

logger = get_logger(__name__)

_VALID_DOMAINS: Set[str] = {
    "penal",
    "civil",
    "labor",
    "constitution",
    "commercial",
    "criminal_procedure",
}


@dataclass
class PipelineResult:
    """Result of the full retrieval + reasoning pipeline."""

    context: str
    """Concatenated legal articles ready for the LLM prompt."""

    sources: List[Dict[str, Any]]
    """Source metadata dicts for each article (frontend payload)."""

    preferred_domain: Optional[str] = None
    """Detected legal domain (may be None)."""

    question_numbers: List[str] = field(default_factory=list)
    """Article numbers extracted from the user question + LLM rewrite."""


class QueryPipeline:
    """
    Orchestrates the full retrieval pipeline.

    Constructed with interface instances (ports) so it can be tested
    with stubs or run with real adapters.
    """

    def __init__(
        self,
        rag_service: RAGServiceInterface,
        llm_service: LLMServiceInterface,
        max_frontend_sources: int = 7,
    ) -> None:
        self._rag = rag_service
        self._llm = llm_service
        self._max_sources = max_frontend_sources

    # ==================================================================
    # Public API
    # ==================================================================

    async def run(
        self,
        question: str,
        retrieval_k: int = 4,
    ) -> PipelineResult:
        """Execute the pipeline end-to-end and return context + sources."""

        # 0. LLM query rewriting
        rewrite = await self._rewrite_legal_query(question)
        llm_domain: Optional[str] = rewrite.get("domain")
        llm_keywords: List[str] = rewrite.get("keywords", [])
        llm_articles: List[str] = rewrite.get("likely_articles", [])
        expanded_query = " ".join(llm_keywords) if llm_keywords else question

        # 1. Domain detection
        if llm_domain:
            preferred_domain = llm_domain
        else:
            preferred_domain = self._rag.detect_domain_semantic(question)

        logger.info("pipeline_domain", domain=preferred_domain)

        # 1b. Merge article numbers
        question_numbers = extract_article_numbers_from_text(question)
        for art in llm_articles:
            art = str(art).strip()
            if art and art not in question_numbers:
                question_numbers.append(art)

        # 2. Article-only shortcut
        stripped = re.sub(r"[^\d\s]", "", question).strip()
        is_article_only = bool(stripped and stripped.isdigit())

        if is_article_only and question_numbers:
            retrieved_docs = self._article_only_lookup(question_numbers)
        else:
            # 2b. Full hybrid retrieval
            retrieved_docs = await self._rag.hybrid_retrieve(
                clean_query=question,
                expanded_query=expanded_query,
                domain=preferred_domain,
                k=max(retrieval_k * 3, 15),
            )

        # 3. Inject articles mentioned in question from cache
        retrieved_docs = self._inject_question_articles(
            retrieved_docs,
            question_numbers,
            preferred_domain,
        )

        # 4. Rerank by article-number match
        if question_numbers:
            retrieved_docs = self._rerank_by_article_match(
                retrieved_docs,
                question_numbers,
            )

        # 5. Domain-priority filtering
        retrieved_docs = self._domain_priority_filter(
            retrieved_docs,
            preferred_domain,
            retrieval_k,
        )

        if not retrieved_docs:
            return PipelineResult(
                context="",
                sources=[],
                preferred_domain=preferred_domain,
                question_numbers=question_numbers,
            )

        # 6. Cross-reference injection
        retrieved_docs = self._inject_cross_references(
            retrieved_docs,
            preferred_domain,
        )

        # 7. Build context + sources
        context, sources = self._build_context_and_sources(retrieved_docs)

        return PipelineResult(
            context=context,
            sources=sources,
            preferred_domain=preferred_domain,
            question_numbers=question_numbers,
        )

    def filter_cited_sources(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return only sources whose article_number appears in the LLM answer."""
        cited_nums: Set[str] = set()
        cited_nums.update(re.findall(r"الماد[ةه]\s+(\d+)", answer))
        cited_nums.update(re.findall(r"ماد[ةه]\s+(\d+)", answer))
        cited_nums.update(re.findall(r"(?:^|\s)(\d+)(?:\s|$|[،,.])", answer))

        if not cited_nums:
            return sources

        filtered = [
            s
            for s in sources
            if str(s.get("metadata", {}).get("article_number", "")).strip()
            in cited_nums
        ]
        return filtered if filtered else sources

    # ==================================================================
    # Private — pipeline steps
    # ==================================================================

    async def _rewrite_legal_query(self, question: str) -> Dict[str, Any]:
        """LLM-based query analysis → {domain, keywords, likely_articles}."""
        from app.infrastructure.adapters.prompt_adapter import QUERY_REWRITE_PROMPT

        prompt = QUERY_REWRITE_PROMPT.replace("{question}", question)
        try:
            from app.interfaces.ai.llm_service import LLMRequest

            resp = await self._llm.generate(LLMRequest(prompt=prompt))
            raw = resp.content.strip()
            
            # Remove markdown code blocks (with or without language specifier)
            raw = re.sub(r"```(?:json)?\s*", "", raw)
            
            # Extract JSON object if wrapped in extra text
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw)
            if json_match:
                raw = json_match.group(0)
            
            raw = raw.strip()

            parsed = json.loads(raw)

            domain = parsed.get("domain") or None
            if domain == "null" or domain not in _VALID_DOMAINS:
                domain = None

            keywords = parsed.get("keywords", [])
            keywords = (
                [str(k).strip() for k in keywords if k]
                if isinstance(keywords, list)
                else []
            )

            likely_articles = parsed.get("likely_articles", [])
            likely_articles = (
                [str(a).strip() for a in likely_articles if a]
                if isinstance(likely_articles, list)
                else []
            )

            logger.info(
                "query_rewrite",
                domain=domain,
                keywords_count=len(keywords),
                articles=likely_articles,
            )
            return {
                "domain": domain,
                "keywords": keywords,
                "likely_articles": likely_articles,
            }
        except Exception as exc:
            logger.warning("query_rewrite_failed", error=str(exc))
            sem_domain = self._rag.detect_domain_semantic(question)
            return {"domain": sem_domain, "keywords": [question], "likely_articles": []}

    def _article_only_lookup(self, question_numbers: List[str]) -> List[Document]:
        """Bypass vector search — fetch articles directly from cache."""
        docs: List[Document] = []
        seen: Set[str] = set()
        for num in question_numbers:
            for doc in self._rag.lookup_article(num):
                meta = doc.metadata or {}
                key = f"{meta.get('article_number')}|{meta.get('domain')}"
                if key not in seen:
                    docs.append(doc)
                    seen.add(key)
        return docs

    def _inject_question_articles(
        self,
        docs: List[Document],
        question_numbers: List[str],
        preferred_domain: Optional[str],
    ) -> List[Document]:
        """Inject articles mentioned in the question that aren't already retrieved."""
        existing_keys: Set[str] = {
            f"{str((d.metadata or {}).get('article_number', '')).strip()}|{(d.metadata or {}).get('domain', '')}"
            for d in docs
        }
        injected = 0
        new_docs = list(docs)
        for num in question_numbers:
            candidates = self._rag.lookup_article(num, domain=preferred_domain)
            if not candidates and preferred_domain:
                candidates = self._rag.lookup_article(num)
            for doc in candidates:
                meta = doc.metadata or {}
                key = f"{str(meta.get('article_number', '')).strip()}|{meta.get('domain', '')}"
                if key not in existing_keys:
                    new_docs.insert(0, doc)
                    existing_keys.add(key)
                    injected += 1
        if injected:
            logger.debug("injected_question_articles", count=injected)
        return new_docs

    @staticmethod
    def _rerank_by_article_match(
        docs: List[Document],
        question_numbers: List[str],
    ) -> List[Document]:
        """Move docs whose article_number matches question numbers to top."""
        num_set = set(question_numbers)
        top, rest = [], []
        for doc in docs:
            art = str((doc.metadata or {}).get("article_number", "")).strip()
            if art in num_set:
                top.append(doc)
            else:
                rest.append(doc)
        return top + rest

    @staticmethod
    def _domain_priority_filter(
        docs: List[Document],
        preferred_domain: Optional[str],
        retrieval_k: int,
    ) -> List[Document]:
        """Prioritise docs from the preferred domain."""
        priority, other = [], []
        for doc in docs:
            content = doc.content or ""
            if len(content.strip()) <= 20:
                continue
            if (
                preferred_domain
                and (doc.metadata or {}).get("domain", "") == preferred_domain
            ):
                priority.append(doc)
            else:
                other.append(doc)

        combined = priority + other
        if preferred_domain and priority:
            return combined[: max(retrieval_k * 4, 15)]
        return combined[: retrieval_k * 2]

    def _inject_cross_references(
        self,
        docs: List[Document],
        preferred_domain: Optional[str],
    ) -> List[Document]:
        """Scan retrieved docs for cross-referenced articles and inject them."""
        referenced: Set[str] = set()
        for doc in docs[:15]:
            content = doc.content or ""
            if not content:
                continue
            for m in re.findall(r"المواد?\s+(\d+(?:\s*[و،,]\s*\d+)*)", content):
                referenced.update(re.findall(r"\d+", m))
            referenced.update(re.findall(r"الماد[ةه]\s+(\d+)", content))

        if not referenced:
            return docs

        existing_keys: Set[str] = {
            f"{str((d.metadata or {}).get('article_number', '')).strip()}|{(d.metadata or {}).get('domain', '')}"
            for d in docs
        }
        new_docs = list(docs)
        added = 0
        for art_num in sorted(referenced):
            candidates = self._rag.lookup_article(art_num, domain=preferred_domain)
            if not candidates:
                candidates = self._rag.lookup_article(art_num)
            for doc in candidates:
                if not doc.content or len(doc.content.strip()) < 30:
                    continue
                meta = doc.metadata or {}
                key = f"{str(meta.get('article_number', '')).strip()}|{meta.get('domain', '')}"
                if key not in existing_keys:
                    new_docs.append(doc)
                    existing_keys.add(key)
                    added += 1
        if added:
            logger.debug("cross_ref_injected", count=added)
        return new_docs

    @staticmethod
    def _build_context_and_sources(
        docs: List[Document],
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Build the LLM context string and the source list for the frontend."""
        context_parts: List[str] = []
        sources: List[Dict[str, Any]] = []

        for doc in docs:
            text = doc.content or ""
            if not text or len(text.strip()) < 20:
                continue

            meta = doc.metadata or {}
            article_num = meta.get("article_number", "غير محدد")
            domain = meta.get("domain", "")
            law_number = meta.get("law_number", "")
            law_year = meta.get("law_year", "")

            # Build law name
            if domain in DOMAIN_NAMES_AR:
                law_name = DOMAIN_NAMES_AR[domain]
                if law_number and law_year:
                    law_name = f"{law_name} رقم {law_number} لسنة {law_year}"
                elif law_number:
                    law_name = f"{law_name} رقم {law_number}"
                elif law_year:
                    law_name = f"{law_name} لسنة {law_year}"
            elif law_number and law_year:
                law_name = f"القانون رقم {law_number} لسنة {law_year}"
            elif law_number:
                law_name = f"القانون رقم {law_number}"
            else:
                law_name = meta.get("law_name_ar", meta.get("law_name", "قانون مصري"))

            context_parts.append(f"المادة {article_num} من {law_name}:\n{text}")
            sources.append(
                {
                    "metadata": {
                        "law_name": law_name,
                        "article_number": str(article_num),
                        "article_text": text,
                        "kitab": meta.get("kitab", ""),
                        "bab": meta.get("bab", ""),
                        "fasl": meta.get("fasl", ""),
                        "law_number": str(meta.get("law_number", "")),
                        "law_year": str(meta.get("law_year", "")),
                        "law_type": meta.get("law_type", meta.get("source_type", "")),
                        "domain": domain,
                    },
                }
            )

        return "\n\n".join(context_parts), sources
