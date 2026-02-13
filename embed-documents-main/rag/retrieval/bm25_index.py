"""BM25 index built from Qdrant payloads for hybrid retrieval."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag.ingestion.vector_store import QdrantManager

logger = logging.getLogger(__name__)


@dataclass
class IndexedDoc:
    """Document stored in BM25 index."""

    doc_id: str
    content: str
    metadata: dict[str, Any]


class BM25Index:
    """BM25 index for Qdrant documents."""

    def __init__(self, qdrant_manager: QdrantManager, max_docs: int = 20000) -> None:
        self.qdrant_manager = qdrant_manager
        self.max_docs = max_docs
        self._bm25: Optional[BM25Okapi] = None
        self._indexed_docs: list[IndexedDoc] = []
        self._lock = asyncio.Lock()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in text.lower().split() if t]

    async def _load_docs(self) -> list[IndexedDoc]:
        client = await self.qdrant_manager._ensure_client()
        indexed_docs: list[IndexedDoc] = []
        offset = None

        while True:
            points, next_offset = await asyncio.to_thread(
                client.scroll,
                collection_name=self.qdrant_manager.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}
                content = payload.get("page_content") or ""
                metadata = payload.get("metadata") or {}
                if not content:
                    continue
                indexed_docs.append(
                    IndexedDoc(doc_id=str(point.id), content=content, metadata=metadata)
                )
                if len(indexed_docs) >= self.max_docs:
                    break

            if next_offset is None or len(indexed_docs) >= self.max_docs:
                break
            offset = next_offset

        return indexed_docs

    async def ensure_index(self) -> None:
        """Build index if not already built."""
        if self._bm25 is not None:
            return

        async with self._lock:
            if self._bm25 is not None:
                return

            logger.info("Building BM25 index from Qdrant payloads...")
            self._indexed_docs = await self._load_docs()
            tokenized = [self._tokenize(doc.content) for doc in self._indexed_docs]
            self._bm25 = BM25Okapi(tokenized) if tokenized else None
            logger.info("BM25 index ready with %s docs", len(self._indexed_docs))

    async def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[Document]:
        await self.ensure_index()
        if not self._bm25 or not self._indexed_docs:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        results: list[Document] = []
        for idx, score in ranked:
            doc = self._indexed_docs[idx]
            if filters and not _matches_filters(doc.metadata, filters):
                continue
            metadata = {**doc.metadata, "bm25_score": float(score)}
            results.append(Document(page_content=doc.content, metadata=metadata))
            if len(results) >= k:
                break

        return results


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
