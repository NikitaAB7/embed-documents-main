"""RAG schemas for document processing and retrieval."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class DocumentCategory(str):
    """Document category enum."""

    ANNUAL_REPORT = "annual-report"
    CONCALL = "concall"
    INVESTOR_PRESENTATION = "investor-presentation"


class DocumentMetadata(BaseModel):
    """Metadata for a document without full content."""

    filename: str = Field(..., description="Document filename")
    fincode: int = Field(..., description="Company financial code")
    company_name: Optional[str] = Field(None, description="Company name")
    category: Literal["annual-report", "concall", "investor-presentation"] = Field(
        ..., description="Document category"
    )
    document_date: str = Field(..., description="Document date in YYYY-MM-DD format")
    is_embedded: bool = Field(
        False, description="Whether this document is already embedded in vector DB"
    )
    page_count: Optional[int] = Field(None, description="Total pages in document")


class QueryRequest(BaseModel):
    """Query request for retrieval pipeline."""

    query: str = Field(..., description="User query")
    top_k: Optional[int] = Field(None, description="Number of results to return")
    use_hyde: bool = Field(True, description="Whether to use HYDE for retrieval")
    use_dynamic_filters: bool = Field(
        False,
        description="Use LLM to dynamically extract filters from query",
    )
    synthesize: bool = Field(True, description="Whether to synthesize an answer")
    strict_citations: bool = Field(
        False, description="Validate sentence-level citations"
    )
    structured_output: bool = Field(
        False, description="Return structured answer with claims"
    )
    evaluate_faithfulness: bool = Field(
        False, description="Run an LLM-based faithfulness check"
    )
    filters: Optional["QueryFilters"] = Field(
        None, description="Metadata filters for retrieval (merged with dynamic filters)"
    )


class QueryFilters(BaseModel):
    """Metadata filters for retrieval."""

    fincode: Optional[int] = Field(None, description="Filter by fincode")
    symbol: Optional[str] = Field(None, description="Filter by stock symbol")
    category: Optional[str] = Field(None, description="Filter by document category")
    date_from: Optional[str] = Field(
        None, description="Filter by document_date >= YYYY-MM-DD"
    )
    date_to: Optional[str] = Field(
        None, description="Filter by document_date <= YYYY-MM-DD"
    )


class RetrievedChunk(BaseModel):
    """Retrieved chunk with metadata for citations."""

    content: str = Field(..., description="Chunk content")
    source: Optional[str] = Field(None, description="Source filename")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier")
    reference_chunk_id: Optional[str] = Field(
        None, description="Referenced chunk identifier (for table refs)"
    )
    metadata: Optional[dict] = Field(None, description="Raw metadata")


class ExtractedFiltersResponse(BaseModel):
    """Response containing LLM-extracted filters from query."""

    fincode: Optional[int] = Field(None, description="Extracted company fincode")
    symbol: Optional[str] = Field(None, description="Extracted stock symbol")
    company_name: Optional[str] = Field(None, description="Extracted company name")
    category: Optional[str] = Field(None, description="Extracted document category")
    date_from: Optional[str] = Field(None, description="Extracted start date")
    date_to: Optional[str] = Field(None, description="Extracted end date")
    confidence: float = Field(0.0, description="Confidence in extracted filters")
    reasoning: str = Field("", description="LLM's reasoning for filter extraction")


class QueryResponse(BaseModel):
    """Query response from retrieval pipeline."""

    use_rag: bool
    confidence: float
    reason: str
    results: list[RetrievedChunk]
    answer: Optional[str] = None
    citations: Optional[list[dict]] = None
    validation: Optional[dict] = None
    structured_answer: Optional[dict] = None
    faithfulness_score: Optional[float] = None
    extracted_filters: Optional[ExtractedFiltersResponse] = Field(
        None, description="Filters extracted dynamically from query by LLM"
    )
