"""Hybrid retrieval (BM25 + vector) for Qdrant collections."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.documents import Document
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag.config import rag_config
from rag.ingestion.vector_store import QdrantManager
from rag.retrieval.bm25_index import BM25Index


@dataclass
class RetrievalResult:
    """Retrieval results from hybrid pipeline."""

    vector_docs: list[Document]
    bm25_docs: list[Document]


class HybridRetriever:
    """Hybrid retriever using vector search and BM25."""

    def __init__(self, qdrant_manager: QdrantManager, bm25_max_docs: int = 20000) -> None:
        self.qdrant_manager = qdrant_manager
        self.bm25 = BM25Index(qdrant_manager, max_docs=bm25_max_docs)

    async def vector_search(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[Document]:
        vector_store = await self.qdrant_manager._ensure_vector_store()
        k = k or rag_config.top_k
        qdrant_filter = _build_qdrant_filter(filters)
        docs_with_scores = await asyncio.to_thread(
            vector_store.similarity_search_with_score, query, k, filter=qdrant_filter
        )
        results: list[Document] = []
        for doc, score in docs_with_scores:
            doc.metadata = {**doc.metadata, "vector_score": float(score)}
            if filters and not _matches_filters(doc.metadata, filters):
                continue
            results.append(doc)
        return results

    async def bm25_search(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[Document]:
        k = k or rag_config.bm25_top_k
        return await self.bm25.search(query, k=k, filters=filters)

    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> RetrievalResult:
        k = k or rag_config.top_k
        vector_task = asyncio.create_task(self.vector_search(query, k=k, filters=filters))
        bm25_task = asyncio.create_task(self.bm25_search(query, k=k, filters=filters))
        vector_docs, bm25_docs = await asyncio.gather(vector_task, bm25_task)
        return RetrievalResult(vector_docs=vector_docs, bm25_docs=bm25_docs)


def _build_qdrant_filter(filters: Optional[dict[str, Any]]) -> Optional[Filter]:
    if not filters:
        return None

    must_conditions = []

    fincode = filters.get("fincode")
    symbol = filters.get("symbol")
    category = filters.get("category")

    if fincode is not None:
        must_conditions.append(
            FieldCondition(key="metadata.fincode", match=MatchValue(value=fincode))
        )
    if symbol:
        must_conditions.append(
            FieldCondition(key="metadata.ticker", match=MatchValue(value=symbol.upper()))
        )
    if category:
        must_conditions.append(
            FieldCondition(key="metadata.category", match=MatchValue(value=category))
        )

    if not must_conditions:
        return None
    return Filter(must=must_conditions)


def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    fincode = filters.get("fincode")
    symbol = filters.get("symbol")
    category = filters.get("category")
    date_from = filters.get("date_from")
    date_to = filters.get("date_to")

    if fincode is not None and metadata.get("fincode") != fincode:
        return False
    if symbol and str(metadata.get("ticker", "")).upper() != symbol.upper():
        return False
    if category and metadata.get("category") != category:
        return False

    doc_date = metadata.get("document_date") or metadata.get("date")
    if date_from and doc_date and doc_date < date_from:
        return False
    if date_to and doc_date and doc_date > date_to:
        return False

    return True
