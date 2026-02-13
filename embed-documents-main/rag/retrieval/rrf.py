"""Reciprocal Rank Fusion (RRF) implementation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document


@dataclass
class RankedDoc:
    """Ranked document representation."""

    doc: Document
    rank: int
    source: str


def rrf_fuse(
    ranked_lists: Iterable[list[Document]],
    k: int = 60,
) -> list[Document]:
    """Fuse multiple ranked lists using RRF."""
    scores = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for docs in ranked_lists:
        for rank, doc in enumerate(docs, start=1):
            key = _doc_key(doc)
            scores[key] += 1.0 / (k + rank)
            doc_map[key] = doc

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in sorted_docs]


def _doc_key(doc: Document) -> str:
    metadata = doc.metadata or {}
    if "chunk_id" in metadata:
        return str(metadata["chunk_id"])
    return f"{metadata.get('source', '')}:{hash(doc.page_content)}"
