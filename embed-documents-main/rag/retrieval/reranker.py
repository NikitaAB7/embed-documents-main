"""Cross-encoder reranker for retrieved documents."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Reranked documents and scores."""

    documents: list[Document]
    scores: list[float]


class CrossEncoderReranker:
    """Reranker using a sentence-transformers cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def _ensure_model(self) -> CrossEncoder:
        if self._model is None:
            logger.info("Loading reranker model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: Iterable[Document], top_k: int = 10) -> RerankResult:
        docs = list(documents)
        if not docs:
            return RerankResult(documents=[], scores=[])

        model = self._ensure_model()
        pairs = [(query, doc.page_content) for doc in docs]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(docs, scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        ranked_docs, ranked_scores = zip(*ranked) if ranked else ([], [])
        return RerankResult(documents=list(ranked_docs), scores=list(ranked_scores))
