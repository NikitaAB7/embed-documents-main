"""LlamaParse adapter for the unified parser module.

This adapter wraps the LlamaParse library to provide standardized output
compatible with the parser module's schemas.
"""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

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

# Regex patterns for table detection
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{2,}\s*\|")
_HEADER_RE = re.compile(r"^#{1,6}\s+")


class LlamaParseAdapter(BaseDocumentParser):
    """Adapter for LlamaParse document parsing.

    Supports:
    - PDF parsing with JSON mode (includes bounding boxes)
    - Fallback to markdown mode
    - Table extraction with captions and summaries
    """

    def __init__(
        self,
        use_json_mode: bool = True,
        extract_layout: bool = True,
        precise_bbox: bool = True,
        split_by_page: bool = True,
        **kwargs: Any,
    ):
        """Initialize LlamaParse adapter.

        Args:
            use_json_mode: Use JSON mode for bbox extraction (default: True)
            extract_layout: Extract page layout information
            precise_bbox: Use precise bounding boxes
            split_by_page: Split results by page
            **kwargs: Additional BaseDocumentParser arguments
        """
        super().__init__(**kwargs)
        self.use_json_mode = use_json_mode
        self.extract_layout = extract_layout
        self.precise_bbox = precise_bbox
        self.split_by_page = split_by_page

    @property
    def backend(self) -> ParserBackend:
        return ParserBackend.LLAMAPARSE

    async def _parse_impl(
        self,
        document: DocumentInput,
        source_name: str,
        **kwargs: Any,
    ) -> ParsedDocument:
        """Parse document using LlamaParse."""
        from llama_parse import LlamaParse

        # Write to temp file if needed
        temp_file = None
        try:
            if isinstance(document, (str, Path)):
                file_path = Path(document)
            else:
                # Write bytes/stream to temp file
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                )
                if isinstance(document, bytes):
                    temp_file.write(document)
                else:
                    temp_file.write(self._read_stream_bytes(document))
                temp_file.close()
                file_path = Path(temp_file.name)

            # Parse with LlamaParse
            chunks = []
            if self.use_json_mode:
                chunks = await self._load_with_json_mode(file_path, source_name)

            # Fallback to markdown if JSON fails
            if not chunks:
                logger.info(f"Falling back to markdown mode for {source_name}")
                chunks = await self._load_with_markdown_mode(file_path, source_name)

            # Build parsed document
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

        finally:
            # Cleanup temp file
            if temp_file:
                try:
                    Path(temp_file.name).unlink()
                except Exception:
                    pass

    async def _load_with_json_mode(
        self,
        file_path: Path,
        source: str,
    ) -> list[ParsedChunk]:
        """Load document with LlamaParse JSON mode (includes bbox)."""
        from llama_parse import LlamaParse

        parser = LlamaParse(
            result_type="json",
            extract_layout=self.extract_layout,
            precise_bounding_box=self.precise_bbox,
            line_level_bounding_box=True,
            split_by_page=self.split_by_page,
        )

        try:
            json_results = await asyncio.to_thread(
                parser.get_json_result, str(file_path)
            )
        except Exception as e:
            logger.error(f"LlamaParse JSON mode failed for {source}: {e}")
            return []

        if not json_results:
            logger.warning(f"LlamaParse returned no results for {source}")
            return []

        chunks: list[ParsedChunk] = []
        for result in json_results:
            if isinstance(result, dict):
                chunks.extend(self._parse_json_elements(result, source))

        return chunks

    async def _load_with_markdown_mode(
        self,
        file_path: Path,
        source: str,
    ) -> list[ParsedChunk]:
        """Load document with LlamaParse markdown mode (fallback)."""
        from llama_parse import LlamaParse

        parser = LlamaParse(result_type="markdown")
        docs = await parser.aload_data(str(file_path))

        if not docs:
            return []

        md_text = "\n\n".join(
            doc.text if hasattr(doc, "text") else str(doc) for doc in docs
        )

        return self._extract_markdown_chunks(md_text, source)

    def _parse_json_elements(
        self,
        json_data: dict,
        source: str,
    ) -> list[ParsedChunk]:
        """Parse LlamaParse JSON response into ParsedChunks."""
        chunks: list[ParsedChunk] = []
        text_index = 0
        table_index = 0

        pages = json_data.get("pages", [])
        if not pages and "elements" in json_data:
            pages = [{"page": 1, "items": json_data["elements"]}]

        if not pages:
            return chunks

        buf_texts: list[str] = []
        buf_bboxes: list[Optional[BoundingBox]] = []
        buf_page: Optional[int] = None
        buf_page_width: Optional[float] = None
        buf_page_height: Optional[float] = None

        def flush_text_buffer():
            nonlocal buf_texts, buf_bboxes, buf_page, text_index
            if not buf_texts:
                return
            merged_text = "\n".join(buf_texts).strip()
            if not merged_text:
                buf_texts, buf_bboxes = [], []
                return

            merged_bbox = self._merge_bboxes(buf_bboxes, buf_page_width, buf_page_height)
            chunk_id = f"{source}::text::{text_index}"
            text_index += 1

            chunks.append(ParsedChunk(
                text=merged_text,
                chunk_id=chunk_id,
                chunk_type=ChunkType.TEXT,
                page=buf_page,
                bbox=merged_bbox,
            ))
            buf_texts, buf_bboxes = [], []

        for page_data in pages:
            page_num = page_data.get("page", page_data.get("page_number", 1))
            page_width = page_data.get("width", 595.0)
            page_height = page_data.get("height", 842.0)

            # Try to get dimensions from images
            for img in page_data.get("images", []):
                if img.get("type") == "full_page_screenshot":
                    page_width = img.get("width", page_width)
                    page_height = img.get("height", page_height)
                    break

            flush_text_buffer()
            buf_page = page_num
            buf_page_width = page_width
            buf_page_height = page_height

            for item in page_data.get("items", page_data.get("elements", [])):
                item_type = item.get("type", "text").lower()
                text = item.get("value", item.get("md", item.get("text", item.get("content", ""))))

                if not text or not text.strip():
                    continue

                raw_bbox = item.get("bBox", item.get("bbox", item.get("bounding_box")))
                bbox = self._extract_bbox(raw_bbox, page_width, page_height)

                if item_type in ("table", "structured_table"):
                    flush_text_buffer()
                    buf_page = page_num
                    buf_page_width = page_width
                    buf_page_height = page_height

                    chunk_id = f"{source}::table::{table_index}"
                    table_index += 1

                    chunks.append(ParsedChunk(
                        text=text.strip(),
                        chunk_id=chunk_id,
                        chunk_type=ChunkType.TABLE,
                        page=page_num,
                        bbox=bbox,
                        table_caption=item.get("caption", item.get("title")),
                        table_summary=self._summarize_table(text),
                    ))
                elif item_type == "heading":
                    flush_text_buffer()
                    buf_page = page_num
                    buf_page_width = page_width
                    buf_page_height = page_height
                    buf_texts.append(text.strip())
                    buf_bboxes.append(bbox)
                else:
                    buf_texts.append(text.strip())
                    buf_bboxes.append(bbox)

        flush_text_buffer()
        return chunks

    def _extract_markdown_chunks(
        self,
        md_text: str,
        source: str,
    ) -> list[ParsedChunk]:
        """Extract chunks from markdown text."""
        lines = md_text.splitlines()
        chunks: list[ParsedChunk] = []
        buffer: list[str] = []
        current_header: Optional[str] = None
        table_index = 0

        def flush_text_buffer():
            nonlocal buffer
            text = "\n".join(buffer).strip()
            if text:
                chunk_id = f"{source}::text::{len(chunks)}"
                if current_header:
                    text = f"{current_header}\n{text}"
                chunks.append(ParsedChunk(
                    text=text,
                    chunk_id=chunk_id,
                    chunk_type=ChunkType.TEXT,
                ))
            buffer = []

        i = 0
        while i < len(lines):
            line = lines[i]

            if _HEADER_RE.match(line):
                flush_text_buffer()
                current_header = line.strip()
                i += 1
                continue

            if "|" in line and i + 1 < len(lines) and _TABLE_SEPARATOR_RE.match(lines[i + 1]):
                flush_text_buffer()

                table_lines = [line, lines[i + 1]]
                i += 2
                while i < len(lines) and "|" in lines[i]:
                    table_lines.append(lines[i])
                    i += 1

                table_md = "\n".join(table_lines).strip()
                chunk_id = f"{source}::table::{table_index}"
                table_index += 1

                if current_header:
                    table_md = f"{current_header}\n{table_md}"

                chunks.append(ParsedChunk(
                    text=table_md,
                    chunk_id=chunk_id,
                    chunk_type=ChunkType.TABLE,
                    table_summary=self._summarize_markdown_table(table_lines),
                ))
                continue

            buffer.append(line)
            i += 1

        flush_text_buffer()
        return chunks

    def _extract_bbox(
        self,
        raw_bbox: Optional[dict],
        page_width: float,
        page_height: float,
    ) -> Optional[BoundingBox]:
        """Extract bounding box from raw data."""
        if not raw_bbox:
            return None

        try:
            if "x" in raw_bbox and "w" in raw_bbox:
                return BoundingBox(
                    x=float(raw_bbox["x"]),
                    y=float(raw_bbox["y"]),
                    width=float(raw_bbox["w"]),
                    height=float(raw_bbox["h"]),
                    page_width=page_width,
                    page_height=page_height,
                )
            if "x" in raw_bbox and "width" in raw_bbox:
                return BoundingBox(
                    x=float(raw_bbox["x"]),
                    y=float(raw_bbox["y"]),
                    width=float(raw_bbox["width"]),
                    height=float(raw_bbox["height"]),
                    page_width=page_width,
                    page_height=page_height,
                )
            if "x1" in raw_bbox:
                return BoundingBox(
                    x=float(raw_bbox["x1"]),
                    y=float(raw_bbox["y1"]),
                    width=float(raw_bbox["x2"]) - float(raw_bbox["x1"]),
                    height=float(raw_bbox["y2"]) - float(raw_bbox["y1"]),
                    page_width=page_width,
                    page_height=page_height,
                )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to extract bbox: {e}")

        return None

    def _merge_bboxes(
        self,
        bboxes: list[Optional[BoundingBox]],
        page_width: Optional[float],
        page_height: Optional[float],
    ) -> Optional[BoundingBox]:
        """Merge multiple bounding boxes into one encompassing box."""
        valid = [b for b in bboxes if b is not None]
        if not valid:
            return None

        x_min = min(b.x for b in valid)
        y_min = min(b.y for b in valid)
        x_max = max(b.x + b.width for b in valid)
        y_max = max(b.y + b.height for b in valid)

        return BoundingBox(
            x=x_min,
            y=y_min,
            width=x_max - x_min,
            height=y_max - y_min,
            page_width=page_width,
            page_height=page_height,
        )

    def _summarize_table(self, text: str) -> str:
        """Create a summary for table content."""
        lines = text.strip().split("\n")
        row_count = len([l for l in lines if l.strip()])
        return f"Table summary: approximately {row_count} rows."

    def _summarize_markdown_table(self, table_lines: Iterable[str]) -> str:
        """Create a summary for a markdown table."""
        lines = list(table_lines)
        if not lines:
            return "Table summary: (empty table)"

        header = lines[0]
        columns = [c.strip() for c in header.strip("|").split("|") if c.strip()]
        body_rows = max(len(lines) - 2, 0)
        columns_text = ", ".join(columns) if columns else "unknown columns"
        return f"Table summary: {body_rows} rows, columns: {columns_text}."

    def _count_pages(self, chunks: list[ParsedChunk]) -> int:
        """Count unique pages from chunks."""
        pages = {c.page for c in chunks if c.page is not None}
        return len(pages) if pages else 0

    def _read_stream_bytes(self, stream) -> bytes:
        """Read bytes from a stream-like object."""
        if hasattr(stream, "getvalue"):
            return stream.getvalue()
        if hasattr(stream, "read"):
            return stream.read()
        return bytes(stream)
