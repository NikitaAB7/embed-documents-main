"""Qdrant Vector Store Manager for RAG Pipeline.

Handles all Qdrant operations including:
- Collection initialization and validation
- Document existence checking
- Batch embedding and upsert operations
- Metadata filtering and retrieval
"""

import asyncio
import logging
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

from rag.config import rag_config
from rag.ingestion.docling_processor import estimate_embedding_cost
from rag.ingestion.document_tracker import DocumentTracker

logger = logging.getLogger(__name__)

# Qdrant's gRPC API uses protobuf int64 which has a max value of 2^63-1
MAX_INT64 = 2**63 - 1
MIN_INT64 = -(2**63)


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Recursively sanitize metadata to handle values that exceed Qdrant's limits.

    Qdrant's gRPC API uses protobuf int64 which has a range of [-2^63, 2^63-1].
    Large integers (like nanosecond timestamps) need to be converted to strings.

    Args:
        metadata: Dictionary potentially containing problematic values

    Returns:
        Sanitized metadata dictionary with large integers converted to strings
    """

    def sanitize_value(value: Any) -> Any:
        """Recursively sanitize a single value."""
        if isinstance(value, dict):
            return {k: sanitize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [sanitize_value(item) for item in value]
        elif isinstance(value, int):
            # Convert integers outside int64 range to strings
            if value > MAX_INT64 or value < MIN_INT64:
                return str(value)
            return value
        else:
            return value

    return sanitize_value(metadata)


class QdrantManager:
    """Manager for Qdrant vector store operations."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        prefer_grpc: bool = True,
    ):
        """Initialize Qdrant manager.

        Args:
            url: Qdrant server URL (defaults to config)
            api_key: Qdrant API key (defaults to config)
            collection_name: Collection name (defaults to config)
            prefer_grpc: Use gRPC for communication (default: True for Qdrant Cloud)
        """
        self.url = url or rag_config.qdrant_url
        self.api_key = api_key or rag_config.qdrant_api_key
        self.collection_name = collection_name or rag_config.qdrant_collection_name
        self.prefer_grpc = prefer_grpc

        if not self.url:
            raise ValueError(
                "Qdrant URL not provided. Set QDRANT_URL environment variable."
            )
        if not self.api_key:
            raise ValueError(
                "Qdrant API key not provided. Set QDRANT_API_KEY environment variable."
            )

        # Initialize embeddings with timeout and retry configuration
        self.embeddings = OpenAIEmbeddings(
            model=rag_config.embedding_model,
            timeout=60.0,  # 60 second timeout for embedding requests
            max_retries=3,  # Retry up to 3 times on failure
        )

        # Don't initialize vector store yet - will be created on first access
        # This allows initialize_collection() to create it if needed
        self._vector_store: Optional[QdrantVectorStore] = None
        self._client: Optional[QdrantClient] = None

        # Initialize document tracker for fast existence checks
        self.doc_tracker = DocumentTracker(collection_name=self.collection_name)

        logger.info(
            f"Initialized QdrantManager with collection '{self.collection_name}'"
        )

    async def _ensure_client(self) -> QdrantClient:
        """Ensure Qdrant client is initialized (async-safe).

        Returns:
            QdrantClient instance
        """
        if self._client is None:
            # Initialize client in thread pool to avoid blocking
            def _create_client():
                return QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    prefer_grpc=self.prefer_grpc,
                )

            self._client = await asyncio.to_thread(_create_client)
        return self._client

    async def _ensure_vector_store(self) -> QdrantVectorStore:
        """Ensure vector store is initialized (async-safe).

        Returns:
            QdrantVectorStore instance
        """
        if self._vector_store is None:
            # Ensure client exists first
            client = await self._ensure_client()

            # Create vector store in thread pool to avoid blocking
            def _create_vector_store():
                return QdrantVectorStore(
                    client=client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings,
                )

            self._vector_store = await asyncio.to_thread(_create_vector_store)

        return self._vector_store

    async def initialize_collection(self) -> dict[str, Any]:
        """Initialize or validate Qdrant collection.

        Creates collection if it doesn't exist, validates if it does.
        Also initializes the SQLite document tracker database.

        Returns:
            Dictionary with collection info and status

        Raises:
            Exception: If collection initialization fails
        """
        try:
            # Initialize document tracker database
            await self.doc_tracker.initialize()

            # Ensure client is initialized
            client = await self._ensure_client()

            # Check if collection exists (wrap blocking call)
            collections = await asyncio.to_thread(client.get_collections)
            collection_names = [col.name for col in collections.collections]

            if self.collection_name in collection_names:
                # Collection exists - get info (wrap blocking call)
                collection_info = await asyncio.to_thread(
                    client.get_collection, self.collection_name
                )
                logger.info(
                    f"Collection '{self.collection_name}' exists with "
                    f"{collection_info.points_count} points"
                )
                return {
                    "status": "exists",
                    "collection_name": self.collection_name,
                    "points_count": collection_info.points_count,
                    "vectors_config": collection_info.config.params.vectors,
                }
            else:
                # Create new collection (wrap blocking call)
                logger.info(f"Creating new collection '{self.collection_name}'")
                await asyncio.to_thread(
                    client.create_collection,
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=rag_config.embedding_dimensions,
                        distance=Distance.COSINE,
                    ),
                )
                return {
                    "status": "created",
                    "collection_name": self.collection_name,
                    "points_count": 0,
                }

        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise

    async def get_embedded_sources(self) -> set[str]:
        """Get set of unique source filenames already embedded in Qdrant.

        Returns:
            Set of filenames that are already embedded

        Raises:
            Exception: If retrieval fails
        """
        try:
            # Ensure client is initialized
            client = await self._ensure_client()

            # Scroll through all points to get unique sources (wrap blocking call)
            # Note: For large collections, consider using filters or pagination
            scroll_result = await asyncio.to_thread(
                client.scroll,
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your collection size
                with_payload=True,
                with_vectors=False,
            )

            sources = set()
            for point in scroll_result[0]:
                if point.payload and "metadata" in point.payload:
                    metadata = point.payload["metadata"]
                    if "source" in metadata:
                        sources.add(metadata["source"])

            logger.info(f"Found {len(sources)} unique sources in Qdrant")
            return sources

        except Exception as e:
            logger.error(f"Failed to get embedded sources: {e}")
            raise

    async def check_documents_exist(self, filenames: list[str]) -> dict[str, bool]:
        """Check which documents are already embedded using SQLite tracker.

        This is significantly faster than querying Qdrant directly, using
        local indexed lookups instead of remote collection scans.

        Args:
            filenames: List of document filenames to check

        Returns:
            Dictionary mapping filename to existence status

        Raises:
            Exception: If check fails
        """
        try:
            # Use SQLite for fast local lookup (100-500x faster than Qdrant)
            return await self.doc_tracker.check_documents_exist(filenames)

        except Exception as e:
            logger.error(f"Failed to check document existence: {e}")
            raise

    async def embed_documents(
        self,
        documents: list[Document],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Embed documents in batches and add to Qdrant.

        Args:
            documents: List of Document objects to embed
            batch_size: Batch size (defaults to config)
            show_progress: Log progress during embedding

        Returns:
            Dictionary with embedding stats and costs

        Raises:
            Exception: If embedding fails
        """
        batch_size = batch_size or rag_config.batch_size

        try:
            # Estimate cost before embedding
            cost_info = estimate_embedding_cost(
                documents, model=rag_config.embedding_model
            )
            logger.info(
                f"Embedding {len(documents)} documents "
                f"(~{cost_info['total_tokens']} tokens, "
                f"${cost_info['estimated_cost_usd']:.4f} estimated cost)"
            )

            # Split into batches
            total_embedded = 0
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(documents) + batch_size - 1) // batch_size

                if show_progress:
                    logger.info(
                        f"Embedding batch {batch_num}/{total_batches} "
                        f"({len(batch)} documents)"
                    )

                # Sanitize metadata to handle large integers that exceed int64 range
                sanitized_batch = []
                for doc in batch:
                    sanitized_doc = Document(
                        page_content=doc.page_content,
                        metadata=sanitize_metadata(doc.metadata),
                    )
                    sanitized_batch.append(sanitized_doc)

                # Add documents to vector store
                # QdrantVectorStore handles embedding internally
                vector_store = await self._ensure_vector_store()
                await asyncio.to_thread(
                    vector_store.add_documents,
                    documents=sanitized_batch,
                )

                total_embedded += len(batch)

            logger.info(f"Successfully embedded {total_embedded} documents")

            # Track embedded documents in SQLite for fast future lookups
            # Group documents by source to get counts
            source_counts = {}
            for doc in documents:
                source = doc.metadata.get("source", "unknown")
                source_counts[source] = source_counts.get(source, 0) + 1

            # Mark each unique source as embedded in tracker
            for source, count in source_counts.items():
                # Calculate per-source cost proportionally
                source_cost = (
                    cost_info["estimated_cost_usd"] * count / len(documents)
                )
                source_tokens = cost_info["total_tokens"] * count // len(documents)

                await self.doc_tracker.mark_embedded(
                    filename=source,
                    document_count=count,
                    total_tokens=source_tokens,
                    embedding_cost_usd=source_cost,
                )

            logger.info(
                f"Tracked {len(source_counts)} unique sources in SQLite tracker"
            )

            return {
                "total_documents": len(documents),
                "total_embedded": total_embedded,
                "unique_sources": len(source_counts),
                "batches_processed": (len(documents) + batch_size - 1) // batch_size,
                "estimated_cost_usd": cost_info["estimated_cost_usd"],
                "total_tokens": cost_info["total_tokens"],
            }

        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            raise

    async def get_collection_info(self) -> dict[str, Any]:
        """Get information about the Qdrant collection.

        Returns:
            Dictionary with collection statistics

        Raises:
            Exception: If retrieval fails
        """
        try:
            # Ensure client is initialized
            client = await self._ensure_client()

            collection_info = await asyncio.to_thread(
                client.get_collection, self.collection_name
            )

            return {
                "collection_name": self.collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status.value,
            }

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise

    async def delete_by_source(self, source: str) -> dict[str, Any]:
        """Delete all documents from a specific source.

        Args:
            source: Source filename to delete

        Returns:
            Dictionary with deletion stats

        Raises:
            Exception: If deletion fails
        """
        try:
            # Ensure client is initialized
            client = await self._ensure_client()

            # Get points matching the source
            scroll_result = await asyncio.to_thread(
                client.scroll,
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source",
                            match=MatchValue(value=source),
                        )
                    ]
                ),
                with_payload=False,
                with_vectors=False,
            )

            point_ids = [point.id for point in scroll_result[0]]

            if point_ids:
                await asyncio.to_thread(
                    client.delete,
                    collection_name=self.collection_name,
                    points_selector=point_ids,
                )

            logger.info(f"Deleted {len(point_ids)} points for source '{source}'")

            # Also remove from SQLite tracker to keep in sync
            removed = await self.doc_tracker.remove_document(source)
            if removed:
                logger.info(f"Removed '{source}' from SQLite tracker")

            return {
                "source": source,
                "deleted_count": len(point_ids),
                "removed_from_tracker": removed,
            }

        except Exception as e:
            logger.error(f"Failed to delete by source: {e}")
            raise
