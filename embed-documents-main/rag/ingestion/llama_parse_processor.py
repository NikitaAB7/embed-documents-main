"""LlamaParse-based document processor with table-aware chunking.

Supports:
- PDF parsing via LlamaParse
- XML parsing via lxml fallback
- Table chunking with caption/summary reference chunks
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from docling.datamodel.base_models import DocumentStream
from langchain_core.documents import Document
from llama_parse import LlamaParse
from lxml import etree

from utils.data_helpers import (
    fincode_to_symbol,
    get_metadata_item_by_attachment_name,
    is_metadata_initialized,
    is_stock_data_initialized,
)

logger = logging.getLogger(__name__)


@dataclass
class ChunkArtifact:
    """Chunk artifact for parsed content."""

    text: str
    chunk_type: str
    chunk_id: str
    page: Optional[int] = None
    bbox: Optional[dict] = None  # {"l": 0.1, "t": 0.2, "r": 0.9, "b": 0.8}
    coord_origin: str = "TOPLEFT"  # "TOPLEFT" or "BOTTOMLEFT"
    table_caption: Optional[str] = None
    table_summary: Optional[str] = None


_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{2,}\s*\|")
_HEADER_RE = re.compile(r"^#{1,6}\s+")


def _resolve_metadata(source_name: str) -> dict:
    """Resolve metadata for a document source using cached helpers."""
    file_meta = (
        get_metadata_item_by_attachment_name(source_name)
        if is_metadata_initialized()
        else None
    ) or {}

    fincode = file_meta.get("fincode")
    ticker = None
    if fincode and is_stock_data_initialized():
        ticker = fincode_to_symbol(fincode)

    return {
        "fincode": fincode,
        "ticker": ticker,
        "category": file_meta.get("subcatname", ""),
        "document_date": (file_meta.get("newsDt", "") or "")[:10],
    }


def _extract_markdown_tables(md_text: str, source: str) -> list[ChunkArtifact]:
    """Extract table and text chunks from markdown.

    Tables become dedicated chunks with optional caption/summary.
    """
    lines = md_text.splitlines()
    chunks: list[ChunkArtifact] = []
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
            chunks.append(
                ChunkArtifact(text=text, chunk_type="text", chunk_id=chunk_id)
            )
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

            # Capture table block
            table_lines = [line, lines[i + 1]]
            i += 2
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1

            table_md = "\n".join(table_lines).strip()
            caption = _find_table_caption(lines, i - len(table_lines) - 1)
            summary = _summarize_table_from_markdown(table_lines)
            chunk_id = f"{source}::table::{table_index}"
            table_index += 1

            if current_header:
                table_md = f"{current_header}\n{table_md}"

            chunks.append(
                ChunkArtifact(
                    text=table_md,
                    chunk_type="table",
                    chunk_id=chunk_id,
                    table_caption=caption,
                    table_summary=summary,
                )
            )
            continue

        buffer.append(line)
        i += 1

    flush_text_buffer()
    return chunks


def _find_table_caption(lines: list[str], start_idx: int) -> Optional[str]:
    """Find a caption for a table by scanning upward from start_idx."""
    idx = start_idx
    while idx >= 0:
        candidate = lines[idx].strip()
        if not candidate:
            idx -= 1
            continue
        if candidate.startswith("|"):
            idx -= 1
            continue
        if re.match(r"^(Table|TABLE|Exhibit|Figure)\b", candidate):
            return candidate
        if candidate.endswith(":"):
            return candidate
        return candidate
    return None


def _summarize_table_from_markdown(table_lines: Iterable[str]) -> str:
    """Create a light-weight summary for a markdown table."""
    lines = list(table_lines)
    if not lines:
        return "Table summary: (empty table)"

    header = lines[0]
    columns = [c.strip() for c in header.strip("|").split("|") if c.strip()]
    body_rows = max(len(lines) - 2, 0)
    columns_text = ", ".join(columns) if columns else "unknown columns"
    return f"Table summary: {body_rows} rows, columns: {columns_text}."


def _get_llamaparse_text(doc) -> str:
    """Extract raw text from LlamaParse response doc."""
    if hasattr(doc, "text"):
        return doc.text
    if hasattr(doc, "get_content"):
        return doc.get_content()
    return str(doc)


def _normalize_bbox(raw_bbox: dict, page_width: float = 1.0, page_height: float = 1.0) -> Optional[dict]:
    """Normalize bounding box to 0-1 range with l/t/r/b keys.
    
    LlamaParse can return bbox in various formats:
    - {"x1", "y1", "x2", "y2"} - absolute coordinates
    - {"x", "y", "width", "height"} - position + size
    - {"left", "top", "right", "bottom"} - already normalized
    """
    if not raw_bbox:
        return None
    
    try:
        # Format: x1/y1/x2/y2 (common in LlamaParse)
        if "x1" in raw_bbox and "y1" in raw_bbox:
            return {
                "l": raw_bbox["x1"] / page_width if page_width > 1 else raw_bbox["x1"],
                "t": raw_bbox["y1"] / page_height if page_height > 1 else raw_bbox["y1"],
                "r": raw_bbox["x2"] / page_width if page_width > 1 else raw_bbox["x2"],
                "b": raw_bbox["y2"] / page_height if page_height > 1 else raw_bbox["y2"],
            }
        
        # Format: x/y/width/height
        if "x" in raw_bbox and "width" in raw_bbox:
            x = raw_bbox["x"] / page_width if page_width > 1 else raw_bbox["x"]
            y = raw_bbox["y"] / page_height if page_height > 1 else raw_bbox["y"]
            w = raw_bbox["width"] / page_width if page_width > 1 else raw_bbox["width"]
            h = raw_bbox["height"] / page_height if page_height > 1 else raw_bbox["height"]
            return {"l": x, "t": y, "r": x + w, "b": y + h}
        
        # Format: left/top/right/bottom (already normalized)
        if "left" in raw_bbox:
            return {
                "l": raw_bbox["left"],
                "t": raw_bbox["top"],
                "r": raw_bbox["right"],
                "b": raw_bbox["bottom"],
            }
        
        # Format: l/t/r/b (already correct)
        if "l" in raw_bbox:
            return raw_bbox
            
    except (KeyError, TypeError, ZeroDivisionError) as e:
        logger.warning(f"Failed to normalize bbox {raw_bbox}: {e}")
    
    return None


def _parse_llamaparse_json_elements(json_data: dict, source: str) -> list[ChunkArtifact]:
    """Parse LlamaParse JSON response into ChunkArtifacts with bbox info.
    
    Handles various LlamaParse JSON structures:
    - pages[].items[] structure
    - pages[].elements[] structure
    - Direct elements[] array
    """
    chunks: list[ChunkArtifact] = []
    text_index = 0
    table_index = 0
    
    # Try to find pages array
    pages = json_data.get("pages", [])
    if not pages and "elements" in json_data:
        # Wrap single-page structure
        pages = [{"page": 1, "items": json_data["elements"]}]
    
    for page_data in pages:
        page_num = page_data.get("page", page_data.get("page_number", 1))
        page_width = page_data.get("width", 1.0)
        page_height = page_data.get("height", 1.0)
        
        # Get items/elements array
        items = page_data.get("items", page_data.get("elements", []))
        
        for item in items:
            item_type = item.get("type", "text").lower()
            text = item.get("value", item.get("text", item.get("content", "")))
            
            if not text or not text.strip():
                continue
            
            # Extract and normalize bbox
            raw_bbox = item.get("bbox", item.get("bounding_box", item.get("boundingBox")))
            bbox = _normalize_bbox(raw_bbox, page_width, page_height)
            
            # Determine coord origin (LlamaParse typically uses top-left)
            coord_origin = item.get("coord_origin", "TOPLEFT")
            
            # Classify chunk type
            if item_type in ("table", "structured_table"):
                chunk_type = "table"
                chunk_id = f"{source}::table::{table_index}"
                table_index += 1
                
                # Try to extract table caption/summary
                caption = item.get("caption", item.get("title"))
                summary = _summarize_table_text(text)
                
                chunks.append(ChunkArtifact(
                    text=text.strip(),
                    chunk_type=chunk_type,
                    chunk_id=chunk_id,
                    page=page_num,
                    bbox=bbox,
                    coord_origin=coord_origin,
                    table_caption=caption,
                    table_summary=summary,
                ))
            else:
                chunk_type = "text"
                chunk_id = f"{source}::text::{text_index}"
                text_index += 1
                
                chunks.append(ChunkArtifact(
                    text=text.strip(),
                    chunk_type=chunk_type,
                    chunk_id=chunk_id,
                    page=page_num,
                    bbox=bbox,
                    coord_origin=coord_origin,
                ))
    
    return chunks


def _summarize_table_text(text: str) -> str:
    """Create a summary for table text content."""
    lines = text.strip().split("\n")
    if not lines:
        return "Table summary: (empty table)"
    
    # Count rows (rough estimate)
    row_count = len([l for l in lines if l.strip()])
    return f"Table summary: approximately {row_count} rows."


async def _load_with_llamaparse_json(file_path: Path, source: str) -> list[ChunkArtifact]:
    """Load document with LlamaParse JSON mode and extract elements with bbox.
    
    Returns list of ChunkArtifacts with page numbers and bounding boxes.
    """
    # Enable bbox extraction with these parameters:
    # - result_type="json" for structured output
    # - extract_layout=True for layout information
    # - precise_bounding_box=True for accurate bbox coordinates
    # - line_level_bounding_box=True for line-level coordinates
    parser = LlamaParse(
        result_type="json",
        extract_layout=True,
        precise_bounding_box=True,
        line_level_bounding_box=True,
        split_by_page=True,  # Split output by page for better page-level tracking
    )
    
    try:
        docs = await parser.aload_data(str(file_path))
    except Exception as e:
        logger.error(f"LlamaParse failed for {file_path}: {e}")
        return []
    
    if not docs:
        return []
    
    all_chunks: list[ChunkArtifact] = []
    
    for doc in docs:
        # Get the raw response - could be JSON string or dict
        raw_content = _get_llamaparse_text(doc)
        
        # Try to parse as JSON
        try:
            if isinstance(raw_content, str):
                json_data = json.loads(raw_content)
            elif isinstance(raw_content, dict):
                json_data = raw_content
            else:
                logger.warning(f"Unexpected LlamaParse response type: {type(raw_content)}")
                continue
                
            chunks = _parse_llamaparse_json_elements(json_data, source)
            all_chunks.extend(chunks)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LlamaParse JSON response: {e}")
            # Fallback: treat as plain text without bbox
            all_chunks.append(ChunkArtifact(
                text=raw_content,
                chunk_type="text",
                chunk_id=f"{source}::text::fallback",
                page=1,
            ))
    
    return all_chunks


async def _load_with_llamaparse_markdown(file_path: Path) -> str:
    """Load document with LlamaParse markdown mode (fallback, no bbox)."""
    parser = LlamaParse(result_type="markdown")
    docs = await parser.aload_data(str(file_path))
    if not docs:
        return ""
    return "\n\n".join(_get_llamaparse_text(doc) for doc in docs)


def _read_stream_bytes(stream) -> bytes:
    """Read bytes from a stream-like object."""
    if hasattr(stream, "getvalue"):
        return stream.getvalue()
    if hasattr(stream, "read"):
        try:
            current_pos = stream.tell()
        except Exception:
            current_pos = None
        data = stream.read()
        if current_pos is not None:
            try:
                stream.seek(current_pos)
            except Exception:
                pass
        return data
    return bytes(stream)


def _xml_to_chunks(xml_bytes: bytes, source: str) -> list[ChunkArtifact]:
    """Parse XML into chunks using element paths."""
    root = etree.fromstring(xml_bytes)
    chunks: list[ChunkArtifact] = []
    index = 0

    for elem in root.iter():
        text = " ".join(t.strip() for t in elem.itertext() if t.strip())
        if not text:
            continue
        path = _get_xml_path(elem)
        chunk_id = f"{source}::xml::{index}"
        index += 1
        chunks.append(
            ChunkArtifact(
                text=f"{path}\n{text}",
                chunk_type="xml",
                chunk_id=chunk_id,
            )
        )

    return chunks


def _get_xml_path(elem: etree._Element) -> str:
    """Create a simple XPath-like path for an XML element."""
    parts = []
    current = elem
    while current is not None:
        tag = current.tag if isinstance(current.tag, str) else "node"
        parts.append(tag)
        current = current.getparent()
    return "/" + "/".join(reversed(parts))


def _build_documents(
    chunks: list[ChunkArtifact],
    source_name: str,
) -> list[Document]:
    """Build LangChain Documents with table reference chunks and bbox metadata."""
    resolved = _resolve_metadata(source_name)
    docs: list[Document] = []

    for chunk in chunks:
        base_meta = {
            "source": source_name,
            "chunk_id": chunk.chunk_id,
            "chunk_type": chunk.chunk_type,
            **resolved,
        }
        
        # Add page number if available
        if chunk.page is not None:
            base_meta["page"] = chunk.page
        
        # Add bounding box if available (for PDF viewer highlighting)
        if chunk.bbox is not None:
            base_meta["bbox"] = chunk.bbox
            base_meta["coord_origin"] = chunk.coord_origin
        
        docs.append(Document(page_content=chunk.text, metadata=base_meta))

        if chunk.chunk_type == "table":
            if chunk.table_caption:
                docs.append(
                    Document(
                        page_content=f"Table caption: {chunk.table_caption}",
                        metadata={
                            **base_meta,
                            "chunk_role": "table_caption",
                            "reference_chunk_id": chunk.chunk_id,
                        },
                    )
                )
            if chunk.table_summary:
                docs.append(
                    Document(
                        page_content=f"Table summary: {chunk.table_summary}",
                        metadata={
                            **base_meta,
                            "chunk_role": "table_summary",
                            "reference_chunk_id": chunk.chunk_id,
                        },
                    )
                )

    return docs


async def process_document(
    doc_stream: DocumentStream,
    file_type: str = "pdf",
    use_json_mode: bool = True,
) -> list[Document]:
    """Process a document using LlamaParse (PDF) or XML parser.

    Args:
        doc_stream: DocumentStream with file name and bytes
        file_type: "pdf" or "xml"
        use_json_mode: If True, use JSON mode for bbox extraction (default).
                       If False, use markdown mode (faster but no bbox).

    Returns:
        List of LangChain Documents with metadata including page, bbox, coord_origin
    """
    if file_type.lower() == "xml":
        xml_bytes = _read_stream_bytes(doc_stream.stream)
        return _build_documents(_xml_to_chunks(xml_bytes, doc_stream.name), doc_stream.name)

    # Default to PDF via LlamaParse
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(doc_stream.name).suffix) as tmp:
        tmp.write(_read_stream_bytes(doc_stream.stream))
        tmp_path = Path(tmp.name)

    try:
        if use_json_mode:
            # JSON mode: extracts page numbers and bounding boxes
            chunks = await _load_with_llamaparse_json(tmp_path, doc_stream.name)
            if not chunks:
                logger.warning(f"JSON mode returned no chunks for {doc_stream.name}, trying markdown fallback")
                md_text = await _load_with_llamaparse_markdown(tmp_path)
                if not md_text:
                    return []
                chunks = _extract_markdown_tables(md_text, doc_stream.name)
        else:
            # Markdown mode: faster but no bbox info
            md_text = await _load_with_llamaparse_markdown(tmp_path)
            if not md_text:
                return []
            chunks = _extract_markdown_tables(md_text, doc_stream.name)
        
        return _build_documents(chunks, doc_stream.name)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to delete temp file: %s", tmp_path)
