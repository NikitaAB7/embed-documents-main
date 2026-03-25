"""Standardized schemas for the unified parser module.

These schemas define the output format that ALL parsers must produce,
regardless of their underlying implementation. This ensures that downstream
consumers (embedders, retrievers, etc.) can work with any parser backend
without modification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ParserBackend(str, Enum):
    """Available parser backends."""

    LLAMAPARSE = "llamaparse"
    DOCLING = "docling"
    DEEPSEEK = "deepseek"
    GLOCR = "glocr"
    PYPDF = "pypdf"
    UNSTRUCTURED = "unstructured"


class ChunkType(str, Enum):
    """Types of content chunks that can be extracted from documents."""

    TEXT = "text"
    TABLE = "table"
    HEADING = "heading"
    LIST = "list"
    IMAGE = "image"
    CODE = "code"
    FORMULA = "formula"
    FIGURE = "figure"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_BREAK = "page_break"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box for a chunk in the source document.

    All coordinates are in absolute PDF points (1 point = 1/72 inch).
    Origin is at top-left unless otherwise specified.

    Attributes:
        x: X coordinate of the top-left corner
        y: Y coordinate of the top-left corner
        width: Width of the bounding box
        height: Height of the bounding box
        page_width: Width of the page containing this bbox
        page_height: Height of the page containing this bbox
        coord_origin: Coordinate origin ("TOPLEFT" or "BOTTOMLEFT")
    """

    x: float
    y: float
    width: float
    height: float
    page_width: Optional[float] = None
    page_height: Optional[float] = None
    coord_origin: str = "TOPLEFT"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "x": self.x,
            "y": self.y,
            "w": self.width,
            "h": self.height,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "coord_origin": self.coord_origin,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Optional["BoundingBox"]:
        """Create BoundingBox from dictionary with various formats."""
        if not data:
            return None

        try:
            # Format: x/y/w/h (standard)
            if "x" in data and "w" in data:
                return cls(
                    x=float(data["x"]),
                    y=float(data["y"]),
                    width=float(data["w"]),
                    height=float(data["h"]),
                    page_width=data.get("page_width"),
                    page_height=data.get("page_height"),
                    coord_origin=data.get("coord_origin", "TOPLEFT"),
                )

            # Format: x/y/width/height
            if "x" in data and "width" in data:
                return cls(
                    x=float(data["x"]),
                    y=float(data["y"]),
                    width=float(data["width"]),
                    height=float(data["height"]),
                    page_width=data.get("page_width"),
                    page_height=data.get("page_height"),
                    coord_origin=data.get("coord_origin", "TOPLEFT"),
                )

            # Format: x1/y1/x2/y2
            if "x1" in data and "y1" in data:
                return cls(
                    x=float(data["x1"]),
                    y=float(data["y1"]),
                    width=float(data["x2"]) - float(data["x1"]),
                    height=float(data["y2"]) - float(data["y1"]),
                    page_width=data.get("page_width"),
                    page_height=data.get("page_height"),
                    coord_origin=data.get("coord_origin", "TOPLEFT"),
                )

            # Format: l/t/r/b (normalized)
            if "l" in data:
                return cls(
                    x=float(data["l"]),
                    y=float(data["t"]),
                    width=float(data["r"]) - float(data["l"]),
                    height=float(data["b"]) - float(data["t"]),
                    page_width=data.get("page_width"),
                    page_height=data.get("page_height"),
                    coord_origin=data.get("coord_origin", "TOPLEFT"),
                )
        except (KeyError, TypeError, ValueError):
            return None

        return None

    def normalize(self) -> dict[str, float]:
        """Return normalized coordinates (0-1 range) if page dimensions are known."""
        if not self.page_width or not self.page_height:
            return self.to_dict()

        return {
            "x": self.x / self.page_width,
            "y": self.y / self.page_height,
            "w": self.width / self.page_width,
            "h": self.height / self.page_height,
        }


@dataclass
class DocumentMetadata:
    """Metadata about the parsed document.

    Attributes:
        source: Original filename or identifier
        fincode: Financial code (company identifier)
        ticker: Stock ticker symbol
        company_name: Full company name
        category: Document category (e.g., "annual-report", "concall")
        document_date: Document date in YYYY-MM-DD format
        page_count: Total number of pages
        language: Detected language code (e.g., "en", "hi")
        parser_backend: Which parser was used
        parse_time_seconds: Time taken to parse
        extra: Any additional metadata from the parser
    """

    source: str
    fincode: Optional[int] = None
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    category: Optional[str] = None
    document_date: Optional[str] = None
    page_count: Optional[int] = None
    language: Optional[str] = None
    parser_backend: Optional[str] = None
    parse_time_seconds: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "source": self.source,
            "fincode": self.fincode,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "category": self.category,
            "document_date": self.document_date,
            "page_count": self.page_count,
            "language": self.language,
            "parser_backend": self.parser_backend,
            "parse_time_seconds": self.parse_time_seconds,
        }
        result.update(self.extra)
        return result


@dataclass
class ParsedChunk:
    """A single chunk of content extracted from a document.

    This is the fundamental unit of parsed content that can be embedded
    and stored in a vector database.

    Attributes:
        text: The extracted text content
        chunk_id: Unique identifier for this chunk
        chunk_type: Type of content (text, table, heading, etc.)
        page: Page number (1-indexed)
        bbox: Bounding box coordinates for visual highlighting
        table_caption: Caption for table chunks
        table_summary: Auto-generated summary for table chunks
        reference_chunk_id: ID of a related chunk (e.g., table reference)
        confidence: Parser's confidence in this extraction (0-1)
        raw_content: Original content before any processing
        extra: Any additional metadata from the parser
    """

    text: str
    chunk_id: str
    chunk_type: ChunkType = ChunkType.TEXT
    page: Optional[int] = None
    bbox: Optional[BoundingBox] = None
    table_caption: Optional[str] = None
    table_summary: Optional[str] = None
    reference_chunk_id: Optional[str] = None
    confidence: Optional[float] = None
    raw_content: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type.value,
            "page": self.page,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "table_caption": self.table_caption,
            "table_summary": self.table_summary,
            "reference_chunk_id": self.reference_chunk_id,
            "confidence": self.confidence,
            "extra": self.extra,
        }

    def to_langchain_document(self) -> "Document":
        """Convert to LangChain Document format."""
        from langchain_core.documents import Document

        metadata = {
            "source": self.chunk_id.split("::")[0] if "::" in self.chunk_id else "",
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type.value,
            "page": self.page,
            "reference_chunk_id": self.reference_chunk_id,
            **self.extra,
        }

        if self.bbox:
            metadata["bbox"] = self.bbox.to_dict()
        if self.table_caption:
            metadata["table_caption"] = self.table_caption
        if self.table_summary:
            metadata["table_summary"] = self.table_summary

        return Document(page_content=self.text, metadata=metadata)


@dataclass
class ParsedDocument:
    """A fully parsed document with all its chunks and metadata.

    This is the primary output of any parser implementation. It contains
    all extracted chunks in a standardized format, along with document-level
    metadata.

    Attributes:
        source: Original filename or identifier
        chunks: List of parsed chunks
        metadata: Document-level metadata
        raw_output: Raw output from the parser (for debugging)
        success: Whether parsing was successful
        error: Error message if parsing failed
    """

    source: str
    chunks: list[ParsedChunk] = field(default_factory=list)
    metadata: Optional[DocumentMetadata] = None
    raw_output: Optional[Any] = field(default=None, repr=False)
    success: bool = True
    error: Optional[str] = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = DocumentMetadata(source=self.source)

    @property
    def text_chunks(self) -> list[ParsedChunk]:
        """Get only text chunks."""
        return [c for c in self.chunks if c.chunk_type == ChunkType.TEXT]

    @property
    def table_chunks(self) -> list[ParsedChunk]:
        """Get only table chunks."""
        return [c for c in self.chunks if c.chunk_type == ChunkType.TABLE]

    @property
    def total_text(self) -> str:
        """Get all text concatenated."""
        return "\n\n".join(c.text for c in self.chunks)

    @property
    def page_count(self) -> int:
        """Get the number of unique pages."""
        pages = {c.page for c in self.chunks if c.page is not None}
        return len(pages) if pages else 0

    def to_langchain_documents(self) -> list["Document"]:
        """Convert all chunks to LangChain Document format."""
        docs = []
        for chunk in self.chunks:
            doc = chunk.to_langchain_document()
            # Add document-level metadata
            if self.metadata:
                doc.metadata.update({
                    "fincode": self.metadata.fincode,
                    "ticker": self.metadata.ticker,
                    "category": self.metadata.category,
                    "document_date": self.metadata.document_date,
                })
            docs.append(doc)
        return docs

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "source": self.source,
            "chunks": [c.to_dict() for c in self.chunks],
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "success": self.success,
            "error": self.error,
        }


# Type aliases for convenience
ChunkList = list[ParsedChunk]
DocumentList = list[ParsedDocument]
