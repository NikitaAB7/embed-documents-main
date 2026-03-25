"""Base abstract class for all document parsers.

All parser implementations must inherit from BaseDocumentParser and implement
the required abstract methods. This ensures a consistent interface regardless
of the underlying parsing engine.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

from parser.schemas import (
    DocumentMetadata,
    ParsedChunk,
    ParsedDocument,
    ParserBackend,
)

logger = logging.getLogger(__name__)


# Type for document input - can be file path, binary stream, or bytes
DocumentInput = Union[str, Path, BinaryIO, bytes]


class BaseDocumentParser(ABC):
    """Abstract base class for document parsers.

    All parser implementations must:
    1. Implement `_parse_impl()` to do the actual parsing
    2. Implement `backend` property to identify the parser type
    3. Optionally override `_enrich_metadata()` for custom metadata handling

    The base class provides:
    - Unified `parse()` and `parse_batch()` methods
    - Timing and error handling
    - Metadata enrichment hooks
    - Conversion to LangChain documents
    """

    def __init__(
        self,
        chunk_prefix_template: Optional[str] = None,
        metadata_resolver: Optional[callable] = None,
    ):
        """Initialize the parser.

        Args:
            chunk_prefix_template: Optional template for chunk text prefix.
                Example: "Ticker: {ticker}; Type: {category}; Date: {document_date}"
            metadata_resolver: Optional async function to resolve metadata.
                Signature: async (source_name: str) -> dict
        """
        self.chunk_prefix_template = chunk_prefix_template
        self.metadata_resolver = metadata_resolver

    @property
    @abstractmethod
    def backend(self) -> ParserBackend:
        """Return the parser backend type."""
        pass

    @property
    def name(self) -> str:
        """Human-readable name for the parser."""
        return self.backend.value

    @abstractmethod
    async def _parse_impl(
        self,
        document: DocumentInput,
        source_name: str,
        **kwargs: Any,
    ) -> ParsedDocument:
        """Implementation-specific parsing logic.

        Args:
            document: The document to parse (file path, bytes, or stream)
            source_name: Name/identifier for the document
            **kwargs: Additional parser-specific options

        Returns:
            ParsedDocument with all extracted chunks
        """
        pass

    async def parse(
        self,
        document: DocumentInput,
        source_name: Optional[str] = None,
        enrich_metadata: bool = True,
        add_prefix: bool = True,
        **kwargs: Any,
    ) -> ParsedDocument:
        """Parse a document and return standardized output.

        This is the main entry point for parsing. It handles:
        - Source name resolution
        - Timing
        - Error handling
        - Metadata enrichment
        - Chunk prefix addition

        Args:
            document: The document to parse
            source_name: Optional name for the document (inferred if not provided)
            enrich_metadata: Whether to resolve additional metadata
            add_prefix: Whether to add prefix to chunk text
            **kwargs: Additional parser-specific options

        Returns:
            ParsedDocument with all extracted chunks and metadata
        """
        # Resolve source name
        if source_name is None:
            source_name = self._resolve_source_name(document)

        logger.info(f"[{self.name}] Parsing document: {source_name}")
        start_time = time.time()

        try:
            # Call implementation-specific parsing
            result = await self._parse_impl(document, source_name, **kwargs)

            # Record timing
            parse_time = time.time() - start_time
            if result.metadata:
                result.metadata.parse_time_seconds = parse_time
                result.metadata.parser_backend = self.backend.value

            # Enrich metadata if requested
            if enrich_metadata and self.metadata_resolver:
                await self._enrich_metadata(result)

            # Add prefix to chunks if requested
            if add_prefix and self.chunk_prefix_template and result.metadata:
                self._add_chunk_prefixes(result)

            logger.info(
                f"[{self.name}] Parsed {source_name}: "
                f"{len(result.chunks)} chunks in {parse_time:.2f}s"
            )

            return result

        except Exception as e:
            parse_time = time.time() - start_time
            logger.error(f"[{self.name}] Failed to parse {source_name}: {e}")

            return ParsedDocument(
                source=source_name,
                chunks=[],
                metadata=DocumentMetadata(
                    source=source_name,
                    parser_backend=self.backend.value,
                    parse_time_seconds=parse_time,
                ),
                success=False,
                error=str(e),
            )

    async def parse_batch(
        self,
        documents: list[tuple[DocumentInput, Optional[str]]],
        **kwargs: Any,
    ) -> list[ParsedDocument]:
        """Parse multiple documents.

        Args:
            documents: List of (document, source_name) tuples
            **kwargs: Additional parser-specific options

        Returns:
            List of ParsedDocument results
        """
        results = []
        for doc, source_name in documents:
            result = await self.parse(doc, source_name, **kwargs)
            results.append(result)
        return results

    def _resolve_source_name(self, document: DocumentInput) -> str:
        """Resolve source name from document input."""
        if isinstance(document, (str, Path)):
            return Path(document).name
        if hasattr(document, "name"):
            return document.name
        return "unknown_document"

    async def _enrich_metadata(self, result: ParsedDocument) -> None:
        """Enrich document metadata using the metadata resolver.

        Override this method for custom metadata handling.
        """
        if not self.metadata_resolver or not result.metadata:
            return

        try:
            extra_meta = await self.metadata_resolver(result.source)
            if extra_meta:
                result.metadata.fincode = extra_meta.get("fincode")
                result.metadata.ticker = extra_meta.get("ticker")
                result.metadata.company_name = extra_meta.get("company_name")
                result.metadata.category = extra_meta.get("category")
                result.metadata.document_date = extra_meta.get("document_date")
                result.metadata.extra.update(
                    {k: v for k, v in extra_meta.items() if k not in [
                        "fincode", "ticker", "company_name", "category", "document_date"
                    ]}
                )
        except Exception as e:
            logger.warning(f"Failed to enrich metadata for {result.source}: {e}")

    def _add_chunk_prefixes(self, result: ParsedDocument) -> None:
        """Add prefix to each chunk's text based on metadata."""
        if not self.chunk_prefix_template or not result.metadata:
            return

        try:
            prefix = self.chunk_prefix_template.format(
                ticker=result.metadata.ticker or "",
                fincode=result.metadata.fincode or "",
                category=result.metadata.category or "",
                document_date=result.metadata.document_date or "",
                company_name=result.metadata.company_name or "",
                source=result.source,
            )

            for chunk in result.chunks:
                if prefix and chunk.text:
                    # Include page number in prefix if available
                    page_suffix = f"; page: {chunk.page}" if chunk.page else ""
                    full_prefix = prefix.rstrip() + page_suffix + "\n"
                    chunk.text = full_prefix + chunk.text

        except KeyError as e:
            logger.warning(f"Failed to format chunk prefix: {e}")

    def to_langchain_documents(
        self,
        result: ParsedDocument,
    ) -> list:
        """Convert ParsedDocument to LangChain Document format.

        Args:
            result: ParsedDocument to convert

        Returns:
            List of LangChain Document objects
        """
        return result.to_langchain_documents()


class ParserConfig:
    """Configuration container for parser settings.

    Use this to configure parser behavior without subclassing.
    """

    def __init__(
        self,
        backend: ParserBackend = ParserBackend.LLAMAPARSE,
        chunk_prefix_template: Optional[str] = (
            "Ticker: {ticker}; Type: {category}; Date: {document_date}; "
            "fincode: {fincode}"
        ),
        max_pages: Optional[int] = None,
        language: str = "en",
        extract_tables: bool = True,
        extract_images: bool = False,
        **extra_options: Any,
    ):
        """Initialize parser configuration.

        Args:
            backend: Which parser backend to use
            chunk_prefix_template: Template for chunk text prefix
            max_pages: Maximum pages to parse (None for all)
            language: Expected document language
            extract_tables: Whether to extract tables
            extract_images: Whether to extract images
            **extra_options: Backend-specific options
        """
        self.backend = backend
        self.chunk_prefix_template = chunk_prefix_template
        self.max_pages = max_pages
        self.language = language
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.extra_options = extra_options

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend": self.backend.value,
            "chunk_prefix_template": self.chunk_prefix_template,
            "max_pages": self.max_pages,
            "language": self.language,
            "extract_tables": self.extract_tables,
            "extract_images": self.extract_images,
            **self.extra_options,
        }
