"""Docling adapter for the unified parser module.

This adapter wraps the Docling/langchain-docling library to provide
standardized output compatible with the parser module's schemas.

Docling offers:
- 97.9% accuracy on complex financial tables
- Better integration with LangChain ecosystem
- HybridChunker that respects table boundaries
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from parser.base import BaseDocumentParser, DocumentInput
from parser.schemas import (
    BoundingBox,
    ChunkType,
    DocumentMetadata,
    ParsedChunk,
    ParsedDocument,
    ParserBackend,
)

logger = logging.getLogger(__name__)


class DoclingAdapter(BaseDocumentParser):
    """Adapter for Docling document parsing.

    Supports:
    - PDF parsing with high accuracy table extraction
    - Rich metadata extraction
    - Bounding box information from Docling elements
    """

    def __init__(
        self,
        use_gpu: bool = True,
        num_threads: int = 8,
        max_pages: Optional[int] = 100,
        **kwargs: Any,
    ):
        """Initialize Docling adapter.

        Args:
            use_gpu: Use GPU acceleration if available
            num_threads: Number of CPU threads for processing
            max_pages: Maximum pages to process (None for all)
            **kwargs: Additional BaseDocumentParser arguments
        """
        super().__init__(**kwargs)
        self.use_gpu = use_gpu
        self.num_threads = num_threads
        self.max_pages = max_pages

    @property
    def backend(self) -> ParserBackend:
        return ParserBackend.DOCLING

    async def _parse_impl(
        self,
        document: DocumentInput,
        source_name: str,
        **kwargs: Any,
    ) -> ParsedDocument:
        """Parse document using Docling."""
        try:
            from docling.datamodel.accelerator_options import (
                AcceleratorDevice,
                AcceleratorOptions,
            )
            from docling.datamodel.base_models import DocumentStream, InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from langchain_docling.loader import DoclingLoader
        except ImportError as e:
            logger.error(f"Docling not installed: {e}")
            return ParsedDocument(
                source=source_name,
                success=False,
                error="Docling library not installed. Install with: pip install docling langchain-docling",
            )

        try:
            # Create document stream
            if isinstance(document, (str, Path)):
                file_path = Path(document)
                source = DocumentStream(
                    name=source_name,
                    stream=open(file_path, "rb"),
                )
            elif isinstance(document, bytes):
                import io
                source = DocumentStream(
                    name=source_name,
                    stream=io.BytesIO(document),
                )
            else:
                source = DocumentStream(
                    name=source_name,
                    stream=document,
                )

            # Configure pipeline options
            accelerator_device = (
                AcceleratorDevice.AUTO if self.use_gpu else AcceleratorDevice.CPU
            )
            pipeline_options = PdfPipelineOptions(
                accelerator_options=AcceleratorOptions(
                    device=accelerator_device,
                    num_threads=self.num_threads,
                ),
            )

            # Create converter
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                },
            )

            # Create loader
            page_range = [1, self.max_pages] if self.max_pages else None
            loader = DoclingLoader(
                file_path=source_name,
                source=source,
                converter=converter,
                convert_kwargs={"page_range": page_range} if page_range else {},
                meta_extractor=self._create_meta_extractor(),
            )

            # Load documents
            chunks: list[ParsedChunk] = []
            text_index = 0

            async for doc in loader.alazy_load():
                page_content = doc.page_content
                meta = doc.metadata or {}

                # Extract page number from Docling metadata
                page_no = self._extract_page_number(meta)

                # Extract bounding box if available
                bbox = self._extract_bbox_from_metadata(meta)

                # Determine chunk type
                chunk_type = self._determine_chunk_type(page_content, meta)

                chunk_id = f"{source_name}::{'table' if chunk_type == ChunkType.TABLE else 'text'}::{text_index}"
                text_index += 1

                chunks.append(ParsedChunk(
                    text=page_content,
                    chunk_id=chunk_id,
                    chunk_type=chunk_type,
                    page=page_no,
                    bbox=bbox,
                    extra={"dl_meta": meta.get("dl_meta")},
                ))

            # Build metadata
            metadata = DocumentMetadata(
                source=source_name,
                page_count=self._count_pages(chunks),
            )

            return ParsedDocument(
                source=source_name,
                chunks=chunks,
                metadata=metadata,
                success=True,
            )

        except Exception as e:
            logger.error(f"Docling parsing failed for {source_name}: {e}")
            return ParsedDocument(
                source=source_name,
                success=False,
                error=str(e),
            )

    def _create_meta_extractor(self):
        """Create a meta extractor for Docling."""
        try:
            from langchain_docling.loader import BaseMetaExtractor
            from docling.chunking import BaseChunk
            from docling.datamodel.document import DoclingDocument
        except ImportError:
            return None

        class SimpleMetaExtractor(BaseMetaExtractor):
            def extract_chunk_meta(self, file_path: str, chunk: BaseChunk) -> dict:
                return {
                    "source": chunk.meta.origin.filename if chunk.meta.origin else file_path,
                    "dl_meta": chunk.meta.export_json_dict() if chunk.meta else {},
                }

            def extract_dl_doc_meta(self, file_path: str, dl_doc: DoclingDocument) -> dict:
                return {"source": file_path}

        return SimpleMetaExtractor()

    def _extract_page_number(self, meta: dict) -> Optional[int]:
        """Extract page number from Docling metadata."""
        dl_meta = meta.get("dl_meta", {})
        doc_items = dl_meta.get("doc_items", [])
        
        if doc_items:
            prov = doc_items[0].get("prov", [])
            if prov:
                return prov[0].get("page_no")
        
        return None

    def _extract_bbox_from_metadata(self, meta: dict) -> Optional[BoundingBox]:
        """Extract bounding box from Docling metadata."""
        dl_meta = meta.get("dl_meta", {})
        doc_items = dl_meta.get("doc_items", [])

        if not doc_items:
            return None

        prov = doc_items[0].get("prov", [])
        if not prov:
            return None

        bbox_data = prov[0].get("bbox")
        if not bbox_data:
            return None

        try:
            # Docling uses l/t/r/b format
            if "l" in bbox_data:
                return BoundingBox(
                    x=float(bbox_data["l"]),
                    y=float(bbox_data["t"]),
                    width=float(bbox_data["r"]) - float(bbox_data["l"]),
                    height=float(bbox_data["b"]) - float(bbox_data["t"]),
                )
            # Also support x/y/w/h format
            if "x" in bbox_data:
                return BoundingBox(
                    x=float(bbox_data.get("x", 0)),
                    y=float(bbox_data.get("y", 0)),
                    width=float(bbox_data.get("w", bbox_data.get("width", 0))),
                    height=float(bbox_data.get("h", bbox_data.get("height", 0))),
                )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to extract bbox: {e}")

        return None

    def _determine_chunk_type(self, content: str, meta: dict) -> ChunkType:
        """Determine the type of chunk from content and metadata."""
        dl_meta = meta.get("dl_meta", {})
        doc_items = dl_meta.get("doc_items", [])

        if doc_items:
            item_type = doc_items[0].get("type", "").lower()
            if item_type == "table":
                return ChunkType.TABLE
            if item_type == "heading":
                return ChunkType.HEADING
            if item_type == "list":
                return ChunkType.LIST
            if item_type == "figure":
                return ChunkType.FIGURE

        # Heuristic: check for table-like content
        lines = content.split("\n")
        table_indicators = sum(1 for line in lines if "|" in line)
        if table_indicators >= 3:
            return ChunkType.TABLE

        return ChunkType.TEXT

    def _count_pages(self, chunks: list[ParsedChunk]) -> int:
        """Count unique pages from chunks."""
        pages = {c.page for c in chunks if c.page is not None}
        return len(pages) if pages else 0
