"""RAG Pipeline Configuration.

Centralized configuration for all RAG components including:
- Qdrant vector store settings
- Embedding model configuration
- Retrieval parameters
- Cost tracking limits
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

class RAGConfig(BaseModel):
    """Configuration for RAG pipeline."""

    # Qdrant Settings
    qdrant_url: str = Field(
        default_factory=lambda: os.getenv("QDRANT_URL", ""),
        description="Qdrant server URL",
    )
    qdrant_api_key: str = Field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY", ""),
        description="Qdrant API key",
    )
    qdrant_collection_name: str = Field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "company_files"),
        description="Qdrant collection name for document embeddings",
    )
    qdrant_prefer_grpc: bool = Field(
        default=True,
        description="Use gRPC for communication with Qdrant Cloud (recommended)",
    )

    # Embedding Model Settings
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model name",
    )
    embedding_dimensions: int = Field(
        default=3072,
        description="Embedding vector dimensions (3072 for text-embedding-3-large)",
    )

    # Retrieval Settings
    top_k: int = Field(
        default=10,
        description="Number of top results to retrieve per intent",
    )
    bm25_top_k: int = Field(
        default=10,
        description="Number of BM25 results to retrieve",
    )
    rerank_top_k: int = Field(
        default=10,
        description="Number of documents to keep after reranking",
    )
    rrf_k: int = Field(
        default=60,
        description="RRF constant used for fusion",
    )
    bm25_max_docs: int = Field(
        default=20000,
        description="Max docs to load into BM25 index",
    )

    # Synthesis Settings
    synthesis_model: str = Field(
        default="gpt-4.1",
        description="Model for response synthesis",
    )
    synthesis_temperature: float = Field(
        default=0.1,
        description="Temperature for synthesis model (lower = more focused)",
    )

    # Batch Processing
    batch_size: int = Field(
        default=100,
        description="Batch size for embedding operations",
    )
    max_parallel_fetches: int = Field(
        default=5,
        description="Maximum parallel document fetches",
    )

    # Cost Tracking
    embedding_cost_per_million: float = Field(
        default=0.02,
        description="Cost per million tokens for embedding model (USD)",
    )
    synthesis_cost_per_million_input: float = Field(
        default=2.50,
        description="Cost per million input tokens for synthesis model (USD)",
    )
    synthesis_cost_per_million_output: float = Field(
        default=10.00,
        description="Cost per million output tokens for synthesis model (USD)",
    )


# Global config instance
rag_config = RAGConfig()
