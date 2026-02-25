"""End-to-end retrieval pipeline: router -> hybrid -> rerank -> RRF -> HYDE.

Supports dynamic LLM-based filter extraction for intelligent query routing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.documents import Document

from rag.config import rag_config
from rag.ingestion.vector_store import QdrantManager
from rag.retrieval.dynamic_filter_router import DynamicFilterRouter, ExtractedFilters
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.hyde import HydeGenerator
from rag.retrieval.query_router import QueryRouter, RouteDecision
from rag.retrieval.reranker import CrossEncoderReranker
from rag.retrieval.rrf import rrf_fuse

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Final retrieval pipeline result."""

    route: RouteDecision
    documents: list[Document]
    extracted_filters: Optional[ExtractedFilters] = field(default=None)


class RetrievalPipeline:
    """Pipeline for hybrid retrieval with HYDE, RRF, and reranking.

    Supports dynamic LLM-based filter extraction when use_dynamic_filters=True.
    """

    def __init__(
        self,
        qdrant_manager: Optional[QdrantManager] = None,
        use_hyde: bool = True,
        use_dynamic_filters: bool = False,
        symbol_lookup: Optional[dict[str, int]] = None,
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
        self.use_dynamic_filters = use_dynamic_filters
        self.dynamic_router = (
            DynamicFilterRouter(symbol_lookup=symbol_lookup)
            if use_dynamic_filters
            else None
        )
        self.symbol_lookup = symbol_lookup

    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        use_dynamic_filters: Optional[bool] = None,
    ) -> PipelineResult:
        """Retrieve documents with optional dynamic filter extraction.

        Args:
            query: User query
            k: Number of results to retrieve
            filters: Explicit filters (merged with dynamic filters)
            use_dynamic_filters: Override instance-level setting

        Returns:
            PipelineResult with documents and extracted filters
        """
        route = self.router.should_use_rag(query)
        if not route.use_rag:
            return PipelineResult(route=route, documents=[])

        k = k or rag_config.top_k

        # Determine if we should use dynamic filtering
        should_use_dynamic = (
            use_dynamic_filters
            if use_dynamic_filters is not None
            else self.use_dynamic_filters
        )

        extracted_filters: Optional[ExtractedFilters] = None
        final_filters = filters

        # Extract filters dynamically using LLM if enabled
        if should_use_dynamic:
            dynamic_router = self.dynamic_router or DynamicFilterRouter(
                symbol_lookup=self.symbol_lookup
            )
            extracted_filters = await dynamic_router.route_query(query, filters)
            final_filters = extracted_filters.to_filter_dict()
            logger.info(
                f"Dynamic filters extracted: {final_filters} "
                f"(confidence: {extracted_filters.confidence:.2f})"
            )

        base = await self.retriever.retrieve(query, k=k, filters=final_filters)
        fusion_lists = [base.vector_docs, base.bm25_docs]

        if self.use_hyde and self.hyde:
            try:
                hyde_result = await self.hyde.generate(query)
                hyde_docs = await self.retriever.vector_search(
                    hyde_result.hypothetical_answer, k=k, filters=final_filters
                )
                fusion_lists.append(hyde_docs)
            except Exception as exc:
                logger.warning("HYDE failed: %s", exc)

        fused = rrf_fuse(fusion_lists, k=rag_config.rrf_k)

        reranked = self.reranker.rerank(
            query, fused, top_k=min(k, rag_config.rerank_top_k)
        )
        return PipelineResult(
            route=route,
            documents=reranked.documents,
            extracted_filters=extracted_filters,
        )
