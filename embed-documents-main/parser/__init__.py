"""Unified Document Parser Module.

This module provides a unified, backend-agnostic interface for parsing documents.
It abstracts away the differences between various parsing engines (LlamaParse,
Docling, DeepSeek, GLOCR, etc.) and produces a standardized output format.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Parser Factory                            │
    │  get_parser("llamaparse") -> LlamaParseAdapter              │
    │  get_parser("docling")    -> DoclingAdapter                 │
    │  get_parser("deepseek")   -> DeepSeekAdapter                │
    │  get_parser("glocr")      -> GLOCRAdapter (GOT-OCR)         │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                 BaseDocumentParser (ABC)                     │
    │  - parse(document) -> ParsedDocument                        │
    │  - parse_batch(documents) -> list[ParsedDocument]           │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   ParsedDocument                             │
    │  - source: str                                              │
    │  - chunks: list[ParsedChunk]                                │
    │  - metadata: DocumentMetadata                               │
    │  - raw_output: Any (optional, for debugging)                │
    └─────────────────────────────────────────────────────────────┘

Available Backends:
    - llamaparse: LlamaParse (JSON mode with bboxes, markdown fallback)
    - docling: IBM Docling (high table accuracy, GPU support)
    - deepseek: DeepSeek-VL2 (vision-language OCR, API or local)
    - glocr: GOT-OCR2.0 (state-of-the-art OCR, formula recognition)
    - pypdf: PyPDF (basic text extraction, no OCR)

Usage:
    from parser import get_parser, ParserBackend

    # Get a parser instance
    parser = get_parser(ParserBackend.LLAMAPARSE)

    # Or use DeepSeek for OCR
    parser = get_parser(ParserBackend.DEEPSEEK, api_key="...")

    # Or GLM-OCR with vLLM server
    parser = get_parser(
        ParserBackend.GLOCR,
        mode=GLOCRMode.VLLM,
        server_url="http://localhost:8080",
    )

    # Or with Ollama
    parser = get_parser(
        ParserBackend.GLOCR,
        mode=GLOCRMode.OLLAMA,
        ollama_model="glm-ocr",
    )

    # Parse a document
    result = await parser.parse(pdf_stream)

    # Access chunks in a standardized format
    for chunk in result.chunks:
        print(chunk.text, chunk.chunk_type, chunk.page)
"""

from parser.schemas import (
    BoundingBox,
    ChunkType,
    DocumentMetadata,
    ParsedChunk,
    ParsedDocument,
    ParserBackend,
)
from parser.base import BaseDocumentParser, ParserConfig
from parser.factory import get_parser, list_available_parsers, is_parser_available
from parser.glocr_parser import GLOCRMode, GLOCRModel

__all__ = [
    # Schemas
    "BoundingBox",
    "ChunkType",
    "DocumentMetadata",
    "ParsedChunk",
    "ParsedDocument",
    "ParserBackend",
    # Base class
    "BaseDocumentParser",
    "ParserConfig",
    # Factory
    "get_parser",
    "list_available_parsers",
    "is_parser_available",
    # GLOCR enums
    "GLOCRMode",
    "GLOCRModel",
]
