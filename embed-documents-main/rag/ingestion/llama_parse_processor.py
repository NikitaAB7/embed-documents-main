"""LlamaParse-based document processor with table-aware chunking.

Supports:
- PDF parsing via LlamaParse
- XML parsing via lxml fallback
- Table chunking with caption/summary reference chunks
"""

from __future__ import annotations

import asyncio
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
    bbox: Optional[dict] = None  # {"x": 257, "y": 62, "w": 81, "h": 21} in PDF points
    page_width: Optional[float] = None  # page width in PDF points
    page_height: Optional[float] = None  # page height in PDF points
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


def _extract_bbox(raw_bbox: dict) -> Optional[dict]:
    """Extract bounding box as {x, y, w, h} in absolute PDF points.

    Stores raw coordinates without normalization so the consumer (PDF viewer)
    can convert to whatever coordinate system it needs using page_width/page_height
    stored alongside in metadata.

    LlamaParse returns bbox as {"x", "y", "w", "h"} in PDF points with
    top-left origin. Other formats are converted to the same structure.
    """
    if not raw_bbox:
        return None

    try:
        # Format: x/y/w/h (LlamaParse JSON bBox format) — pass through
        if "x" in raw_bbox and "w" in raw_bbox:
            return {
                "x": raw_bbox["x"],
                "y": raw_bbox["y"],
                "w": raw_bbox["w"],
                "h": raw_bbox["h"],
            }

        # Format: x/y/width/height
        if "x" in raw_bbox and "width" in raw_bbox:
            return {
                "x": raw_bbox["x"],
                "y": raw_bbox["y"],
                "w": raw_bbox["width"],
                "h": raw_bbox["height"],
            }

        # Format: x1/y1/x2/y2
        if "x1" in raw_bbox and "y1" in raw_bbox:
            return {
                "x": raw_bbox["x1"],
                "y": raw_bbox["y1"],
                "w": raw_bbox["x2"] - raw_bbox["x1"],
                "h": raw_bbox["y2"] - raw_bbox["y1"],
            }

        # Format: l/t/r/b (normalized — store as-is, no page dims needed)
        if "l" in raw_bbox:
            return raw_bbox

        # Format: left/top/right/bottom
        if "left" in raw_bbox:
            return {
                "l": raw_bbox["left"],
                "t": raw_bbox["top"],
                "r": raw_bbox["right"],
                "b": raw_bbox["bottom"],
            }

    except (KeyError, TypeError) as e:
        logger.warning(f"Failed to extract bbox {raw_bbox}: {e}")

    return None


def _merge_bboxes(bboxes: list[dict]) -> Optional[dict]:
    """Compute the union (enclosing) bounding box for a list of bboxes.

    All bboxes must be in {x, y, w, h} format (absolute PDF points, top-left origin).
    Returns None if the list is empty or all entries are None.
    """
    valid = [b for b in bboxes if b is not None]
    if not valid:
        return None

    x_min = min(b["x"] for b in valid)
    y_min = min(b["y"] for b in valid)
    x_max = max(b["x"] + b["w"] for b in valid)
    y_max = max(b["y"] + b["h"] for b in valid)

    return {"x": x_min, "y": y_min, "w": x_max - x_min, "h": y_max - y_min}


def _parse_llamaparse_json_elements(json_data: dict, source: str) -> list[ChunkArtifact]:
    """Parse LlamaParse JSON response into ChunkArtifacts with bbox info.

    Merges consecutive text/heading items on the same page into section-level
    chunks, splitting at headings and tables. Each merged chunk gets a union
    bounding box encompassing all its source items.

    Tables are always emitted as standalone chunks.
    """
    chunks: list[ChunkArtifact] = []
    text_index = 0
    table_index = 0

    # Try to find pages array
    pages = json_data.get("pages", [])
    logger.info(f"[PARSE] Found {len(pages)} pages in json_data")

    if not pages and "elements" in json_data:
        logger.info(f"[PARSE] No 'pages' key, using 'elements' directly ({len(json_data['elements'])} elements)")
        pages = [{"page": 1, "items": json_data["elements"]}]

    if not pages:
        logger.warning(f"[PARSE] No 'pages' or 'elements' found in json_data. "
                       f"Available keys: {list(json_data.keys()) if isinstance(json_data, dict) else type(json_data).__name__}")
        return chunks

    # --- per-page accumulator state ---
    buf_texts: list[str] = []
    buf_bboxes: list[Optional[dict]] = []
    buf_page: Optional[int] = None
    buf_page_width: Optional[float] = None
    buf_page_height: Optional[float] = None
    buf_coord_origin: str = "TOPLEFT"

    def flush_text_buffer():
        nonlocal buf_texts, buf_bboxes, buf_page, text_index
        if not buf_texts:
            return
        merged_text = "\n".join(buf_texts).strip()
        if not merged_text:
            buf_texts, buf_bboxes = [], []
            return
        merged_bbox = _merge_bboxes(buf_bboxes)
        chunk_id = f"{source}::text::{text_index}"
        text_index += 1
        chunks.append(ChunkArtifact(
            text=merged_text,
            chunk_type="text",
            chunk_id=chunk_id,
            page=buf_page,
            bbox=merged_bbox,
            page_width=buf_page_width,
            page_height=buf_page_height,
            coord_origin=buf_coord_origin,
        ))
        buf_texts, buf_bboxes = [], []

    for page_idx, page_data in enumerate(pages):
        page_num = page_data.get("page", page_data.get("page_number", 1))
        page_width = page_data.get("width", 1.0)
        page_height = page_data.get("height", 1.0)

        if page_idx == 0:
            logger.info(f"[PARSE] Page {page_num} keys: {list(page_data.keys())}")

        items = page_data.get("items", page_data.get("elements", []))

        if page_idx == 0:
            logger.info(f"[PARSE] Page {page_num}: {len(items)} items/elements")
            if items:
                logger.info(f"[PARSE] First item keys: {list(items[0].keys()) if isinstance(items[0], dict) else type(items[0]).__name__}")
                logger.info(f"[PARSE] First item preview: {str(items[0])[:300]}")

        # Flush any leftover buffer from the previous page
        flush_text_buffer()
        buf_page = page_num
        buf_page_width = page_width
        buf_page_height = page_height

        for item in items:
            item_type = item.get("type", "text").lower()
            text = item.get("value", item.get("md", item.get("text", item.get("content", ""))))

            if not text or not text.strip():
                continue

            raw_bbox = item.get("bBox", item.get("bbox", item.get("bounding_box", item.get("boundingBox"))))
            bbox = _extract_bbox(raw_bbox)
            coord_origin = item.get("coord_origin", "TOPLEFT")

            if item_type in ("table", "structured_table"):
                # Flush accumulated text before the table
                flush_text_buffer()
                buf_page = page_num
                buf_page_width = page_width
                buf_page_height = page_height

                chunk_id = f"{source}::table::{table_index}"
                table_index += 1
                caption = item.get("caption", item.get("title"))
                summary = _summarize_table_text(text)

                chunks.append(ChunkArtifact(
                    text=text.strip(),
                    chunk_type="table",
                    chunk_id=chunk_id,
                    page=page_num,
                    bbox=bbox,
                    page_width=page_width,
                    page_height=page_height,
                    coord_origin=coord_origin,
                    table_caption=caption,
                    table_summary=summary,
                ))
            elif item_type == "heading":
                # Headings start a new section — flush previous text
                flush_text_buffer()
                buf_page = page_num
                buf_page_width = page_width
                buf_page_height = page_height
                buf_coord_origin = coord_origin
                buf_texts.append(text.strip())
                buf_bboxes.append(bbox)
            else:
                # Accumulate consecutive text items
                buf_coord_origin = coord_origin
                buf_texts.append(text.strip())
                buf_bboxes.append(bbox)

    # Flush any remaining buffer from the last page
    flush_text_buffer()

    logger.info(f"[PARSE] Total chunks parsed: {len(chunks)} (text={text_index}, tables={table_index})")
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

    Uses get_json_result() to get structured JSON with page-level items and
    bounding boxes. aload_data() does not work with result_type="json".

    Returns list of ChunkArtifacts with page numbers and bounding boxes.
    """
    parser = LlamaParse(
        result_type="json",
        extract_layout=True,
        precise_bounding_box=True,
        line_level_bounding_box=True,
        split_by_page=True,
    )

    try:
        # get_json_result returns the raw JSON structure (list of result dicts)
        # aload_data() is broken for result_type="json" — returns 0 docs
        json_results = await asyncio.to_thread(parser.get_json_result, str(file_path))
    except Exception as e:
        logger.error(f"[JSON] LlamaParse get_json_result failed for {source}: {e}")
        return []

    if not json_results:
        logger.warning(f"[JSON] LlamaParse returned no results for {source}")
        return []

    logger.info(f"[JSON] LlamaParse returned {len(json_results)} result(s) for {source}")

    all_chunks: list[ChunkArtifact] = []

    for result_idx, result in enumerate(json_results):
        if not isinstance(result, dict):
            logger.warning(f"[JSON] Result {result_idx}: unexpected type {type(result).__name__}")
            continue

        logger.info(f"[JSON] Result {result_idx} top-level keys: {list(result.keys())}")

        chunks = _parse_llamaparse_json_elements(result, source)
        logger.info(f"[JSON] Result {result_idx}: parsed {len(chunks)} chunks")
        if chunks:
            with_bbox = sum(1 for c in chunks if c.bbox is not None)
            logger.info(f"[JSON] Result {result_idx}: {with_bbox}/{len(chunks)} chunks have bbox")
        all_chunks.extend(chunks)

    logger.info(f"[JSON] Total chunks for {source}: {len(all_chunks)}")
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
        # bbox is in absolute PDF points; page_width/page_height let the
        # consumer normalize or scale as needed for any page size.
        if chunk.bbox is not None:
            base_meta["bbox"] = chunk.bbox
            base_meta["coord_origin"] = chunk.coord_origin
            if chunk.page_width is not None:
                base_meta["page_width"] = chunk.page_width
            if chunk.page_height is not None:
                base_meta["page_height"] = chunk.page_height
        
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
            logger.info(f"[PROCESS] Starting JSON mode for {doc_stream.name}")
            chunks = await _load_with_llamaparse_json(tmp_path, doc_stream.name)
            if not chunks:
                logger.warning(f"[PROCESS] JSON mode returned no chunks for {doc_stream.name}, trying markdown fallback")
                md_text = await _load_with_llamaparse_markdown(tmp_path)
                if not md_text:
                    logger.warning(f"[PROCESS] Markdown fallback also returned nothing for {doc_stream.name}")
                    return []
                chunks = _extract_markdown_tables(md_text, doc_stream.name)
                logger.info(f"[PROCESS] Markdown fallback produced {len(chunks)} chunks (no bbox)")
            else:
                with_bbox = sum(1 for c in chunks if c.bbox is not None)
                with_page = sum(1 for c in chunks if c.page is not None)
                logger.info(f"[PROCESS] JSON mode produced {len(chunks)} chunks: "
                            f"{with_bbox} with bbox, {with_page} with page number")
        else:
            # Markdown mode: faster but no bbox info
            logger.info(f"[PROCESS] Starting Markdown mode for {doc_stream.name}")
            md_text = await _load_with_llamaparse_markdown(tmp_path)
            if not md_text:
                return []
            chunks = _extract_markdown_tables(md_text, doc_stream.name)

        docs = _build_documents(chunks, doc_stream.name)
        docs_with_bbox = sum(1 for d in docs if d.metadata.get("bbox") is not None)
        logger.info(f"[PROCESS] Final result for {doc_stream.name}: {len(docs)} documents, "
                    f"{docs_with_bbox} with bbox metadata")
        return docs
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to delete temp file: %s", tmp_path)
