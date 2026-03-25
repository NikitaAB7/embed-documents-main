"""Parser factory for instantiating document parsers.

This module provides a factory function to create parser instances based on
the desired backend. It also handles lazy loading of parser implementations
to avoid import errors when optional dependencies are not installed.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type

from parser.base import BaseDocumentParser, ParserConfig
from parser.schemas import ParserBackend

logger = logging.getLogger(__name__)

# Registry of available parsers (populated lazily)
_PARSER_REGISTRY: dict[ParserBackend, Type[BaseDocumentParser]] = {}


def _register_parser(backend: ParserBackend, parser_class: Type[BaseDocumentParser]):
    """Register a parser implementation."""
    _PARSER_REGISTRY[backend] = parser_class


def _load_llamaparse_adapter() -> Optional[Type[BaseDocumentParser]]:
    """Load LlamaParse adapter if available."""
    try:
        from parser.llama_parser import LlamaParseAdapter
        return LlamaParseAdapter
    except ImportError as e:
        logger.warning(f"LlamaParse adapter not available: {e}")
        return None


def _load_docling_adapter() -> Optional[Type[BaseDocumentParser]]:
    """Load Docling adapter if available."""
    try:
        from parser.docling_parser import DoclingAdapter
        return DoclingAdapter
    except ImportError as e:
        logger.warning(f"Docling adapter not available: {e}")
        return None


def _load_pypdf_adapter() -> Optional[Type[BaseDocumentParser]]:
    """Load PyPDF adapter if available."""
    try:
        from parser.pypdf_parser import PyPDFAdapter
        return PyPDFAdapter
    except ImportError as e:
        logger.debug(f"PyPDF adapter not available: {e}")
        return None


def _load_deepseek_adapter() -> Optional[Type[BaseDocumentParser]]:
    """Load DeepSeek adapter if available (future)."""
    try:
        from parser.deepseek_parser import DeepSeekAdapter
        return DeepSeekAdapter
    except ImportError:
        return None


def _load_glocr_adapter() -> Optional[Type[BaseDocumentParser]]:
    """Load GLOCR adapter if available (future)."""
    try:
        from parser.glocr_parser import GLOCRAdapter
        return GLOCRAdapter
    except ImportError:
        return None


# Loader functions for each backend
_BACKEND_LOADERS = {
    ParserBackend.LLAMAPARSE: _load_llamaparse_adapter,
    ParserBackend.DOCLING: _load_docling_adapter,
    ParserBackend.PYPDF: _load_pypdf_adapter,
    ParserBackend.DEEPSEEK: _load_deepseek_adapter,
    ParserBackend.GLOCR: _load_glocr_adapter,
}


def get_parser(
    backend: ParserBackend | str = ParserBackend.LLAMAPARSE,
    config: Optional[ParserConfig] = None,
    **kwargs: Any,
) -> BaseDocumentParser:
    """Get a parser instance for the specified backend.

    Args:
        backend: Parser backend to use (string or ParserBackend enum)
        config: Optional ParserConfig with settings
        **kwargs: Additional arguments passed to the parser constructor

    Returns:
        BaseDocumentParser instance

    Raises:
        ValueError: If the backend is not available or not installed

    Example:
        >>> from parser import get_parser, ParserBackend
        >>> parser = get_parser(ParserBackend.LLAMAPARSE)
        >>> result = await parser.parse("document.pdf")
    """
    # Convert string to enum if needed
    if isinstance(backend, str):
        try:
            backend = ParserBackend(backend.lower())
        except ValueError:
            available = [b.value for b in ParserBackend]
            raise ValueError(
                f"Unknown parser backend: {backend}. "
                f"Available backends: {available}"
            )

    # Check registry first
    if backend in _PARSER_REGISTRY:
        parser_class = _PARSER_REGISTRY[backend]
    else:
        # Try to load the adapter
        loader = _BACKEND_LOADERS.get(backend)
        if not loader:
            raise ValueError(f"No loader for backend: {backend}")

        parser_class = loader()
        if parser_class is None:
            raise ValueError(
                f"Parser backend '{backend.value}' is not available. "
                f"Please install the required dependencies."
            )

        # Cache in registry
        _PARSER_REGISTRY[backend] = parser_class

    # Merge config into kwargs
    if config:
        kwargs.setdefault("chunk_prefix_template", config.chunk_prefix_template)
        kwargs.update(config.extra_options)

    return parser_class(**kwargs)


def list_available_parsers() -> list[ParserBackend]:
    """List all available parser backends.

    Returns:
        List of ParserBackend enums that are currently available
    """
    available = []
    for backend, loader in _BACKEND_LOADERS.items():
        try:
            parser_class = loader()
            if parser_class is not None:
                available.append(backend)
        except Exception:
            pass
    return available


def is_parser_available(backend: ParserBackend | str) -> bool:
    """Check if a specific parser backend is available.

    Args:
        backend: Parser backend to check

    Returns:
        True if the parser is available, False otherwise
    """
    if isinstance(backend, str):
        try:
            backend = ParserBackend(backend.lower())
        except ValueError:
            return False

    loader = _BACKEND_LOADERS.get(backend)
    if not loader:
        return False

    try:
        return loader() is not None
    except Exception:
        return False
