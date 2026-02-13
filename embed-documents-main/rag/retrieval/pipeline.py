"""End-to-end retrieval pipeline: router -> hybrid -> rerank -> RRF -> HYDE."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document

from rag.config import rag_config
from rag.ingestion.vector_store import QdrantManager
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.hyde import HydeGenerator
from rag.retrieval.query_router import QueryRouter, RouteDecision
from rag.retrieval.rrf import rrf_fuse
from rag.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Final retrieval pipeline result."""

    route: RouteDecision
    documents: list[Document]


class RetrievalPipeline:
    """Pipeline for hybrid retrieval with HYDE, RRF, and reranking."""

    def __init__(
        self,
        qdrant_manager: Optional[QdrantManager] = None,
        use_hyde: bool = True,
    ) -> None:
        self.qdrant_manager = qdrant_manager or QdrantManager()
        self.router = QueryRouter()
        self.retriever = HybridRetriever(
            self.qdrant_manager,
            bm25_max_docs=rag_config.bm25_max_docs,
        )
        self.reranker = CrossEncoderReranker()
        self.use_hyde = use_hyde
        self.hyde = HydeGenerator() if use_hyde else None

    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[dict[str, object]] = None,
    ) -> PipelineResult:
        route = self.router.should_use_rag(query)
        if not route.use_rag:
            return PipelineResult(route=route, documents=[])

        k = k or rag_config.top_k

        base = await self.retriever.retrieve(query, k=k, filters=filters)
        fusion_lists = [base.vector_docs, base.bm25_docs]

        if self.use_hyde and self.hyde:
            try:
                hyde_result = await self.hyde.generate(query)
                hyde_docs = await self.retriever.vector_search(
                    hyde_result.hypothetical_answer, k=k, filters=filters
                )
                fusion_lists.append(hyde_docs)
            except Exception as exc:
                logger.warning("HYDE failed: %s", exc)

        fused = rrf_fuse(fusion_lists, k=rag_config.rrf_k)

        reranked = self.reranker.rerank(
            query, fused, top_k=min(k, rag_config.rerank_top_k)
        )
        return PipelineResult(route=route, documents=reranked.documents)
