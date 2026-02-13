"""Document processor using LangChain Docling loader for enhanced PDF extraction.

Based on research findings (PDF Parsing Libraries for Financial RAG Pipelines):
- Docling: 97.9% accuracy on complex financial tables
- langchain-docling: Better integration with LangChain ecosystem
- HybridChunker: Respects table boundaries (critical for financial data)
- DOC_CHUNKS export: Preserves table structure as HTML
"""

import logging
from typing import Any

import tiktoken
from docling.chunking import BaseChunk
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import DoclingDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from langchain_core.documents import Document

from langchain_docling.loader import BaseMetaExtractor, DoclingLoader
from utils.data_helpers import (
    fincode_to_symbol,
    get_metadata_item_by_attachment_name,
    is_metadata_initialized,
    is_stock_data_initialized,
)

logger = logging.getLogger(__name__)


class MDTableSerializerProvider(ChunkingSerializerProvider):
    """Markdown Table Serializer Provider."""

    def get_serializer(self, doc):
        """Get serializer."""
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),  # configuring a different table serializer
        )


# Chunker initialization removed - not currently used (see line 137)
# To re-enable: uncomment below and pass chunker=_chunker to DoclingLoader
# _chunker = HybridChunker(
#     serializer_provider=MDTableSerializerProvider(),
# )


class MetaExtractor(BaseMetaExtractor):
    """MetaExtractor with graceful fallback when metadata not initialized."""

    def extract_chunk_meta(self, file_path: str, chunk: BaseChunk) -> dict[str, Any]:
        """Extract chunk meta with graceful fallback."""
        file_name = chunk.meta.origin.filename

        # Try to get metadata if initialized, otherwise use defaults
        if is_metadata_initialized():
            file_meta_data = get_metadata_item_by_attachment_name(file_name) or {}
            logger.info(f"Extracted metadata for {file_name}: {file_meta_data}")
        else:
            logger.warning(
                "Metadata not initialized. Using filename for metadata extraction. "
                "Call 'await initialize_metadata_data()' for full metadata."
            )
            file_meta_data = {}

        fincode = file_meta_data.get("fincode", None)

        # Try to get ticker if stock data initialized
        ticker = None
        if fincode and is_stock_data_initialized():
            ticker = fincode_to_symbol(fincode)

        return {
            "source": file_name,
            "category": file_meta_data.get("subcatname", ""),
            "document_date": file_meta_data.get("newsDt", ""),
            "dl_meta": chunk.meta.export_json_dict(),
            "fincode": fincode,
            "ticker": ticker,
        }

    def extract_dl_doc_meta(
        self, file_path: str, dl_doc: DoclingDocument
    ) -> dict[str, Any]:
        """Extract Docling document meta."""
        logger.info(f"extract_dl_doc_meta called for: {file_path}")

        # Try to get metadata if initialized
        file_name = file_path if isinstance(file_path, str) else getattr(file_path, 'name', str(file_path))

        if is_metadata_initialized():
            file_meta_data = get_metadata_item_by_attachment_name(file_name) or {}
            logger.info(f"Extracted metadata in extract_dl_doc_meta for {file_name}: {file_meta_data}")
        else:
            logger.warning(
                "Metadata not initialized in extract_dl_doc_meta. Using filename for metadata extraction. "
                "Call 'await initialize_metadata_data()' for full metadata."
            )
            file_meta_data = {}

        fincode = file_meta_data.get("fincode", None)

        # Try to get ticker if stock data initialized
        ticker = None
        if fincode and is_stock_data_initialized():
            ticker = fincode_to_symbol(fincode)

        result = {
            "source": file_name,
            "category": file_meta_data.get("subcatname", ""),
            "document_date": file_meta_data.get("newsDt", ""),
            "fincode": fincode,
            "ticker": ticker,
        }
        logger.info(f"extract_dl_doc_meta returning: {result}")
        return result


async def process_pdf(
    pdf: DocumentStream,
) -> list[Document]:
    """Process a PDF using langchain-docling loader with HybridChunker.

    Args:
        pdf: DocumentStream object containing PDF data

    Returns:
        List of Document objects with rich metadata

    Raises:
        Exception: If PDF processing fails
    """
    try:
        logger.info(f"Processing PDF with langchain-docling: {pdf.name}")
        # Create custom document converter with our pipeline options
        pipeline_options = PdfPipelineOptions(
            accelerator_options=AcceleratorOptions(
                device=AcceleratorDevice.AUTO, num_threads=8
            ),
            # do_picture_description=True,
            # picture_description_options=(smolvlm_picture_description),
            # images_scale=2.0,
            # generate_picture_images=True,
        )
        # pipeline_options.picture_description_options.prompt = (
        #     "Describe the image in three sentences. Be consise and accurate."
        # )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            },
        )

        # Create loader WITHOUT chunker - we'll do page-level aggregation ourselves
        # This avoids tokenizer issues and gives us direct access to Docling's elements
        # Note: Pass pdf (DocumentStream) as file_path - DoclingLoader handles both str and DocumentStream
        loader = DoclingLoader(
            file_path=pdf.name,
            source=pdf,
            converter=converter,
            convert_kwargs={
                "page_range": [1, 100],
            },
            # chunker=_chunker,
            meta_extractor=MetaExtractor(),
        )

        # Load and process document
        docs = []
        async for doc in loader.alazy_load():
            page_content = doc.page_content
            meta = doc.metadata
            page_no = (
                meta.get("dl_meta", {})
                .get("doc_items", [])[0]
                .get("prov", [])[0]
                .get("page_no", "")
            )
            prefix = f"Ticker: {meta.get('ticker', '')}; Type: {meta.get('category', '')}; Date: {meta.get('document_date', '')}; fincode: {meta.get('fincode', '')}; page no: {page_no}\n"
            doc.page_content = prefix + page_content
            docs.append(doc)

        return docs

    except Exception as e:
        logger.error(f"Failed to process PDF {pdf.name}: {e}")
        raise


def estimate_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Estimate token count for given text using tiktoken.

    Args:
        text: Text to count tokens for
        encoding_name: Tiktoken encoding name (default: cl100k_base for text-embedding-3-*)

    Returns:
        Estimated token count
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(
            f"Failed to count tokens with tiktoken: {e}. Using approximation."
        )
        # Fallback: rough approximation (1 token ≈ 4 chars)
        return len(text) // 4


def estimate_documents_tokens(documents: list[Document]) -> int:
    """Calculate total token count for a list of documents.

    Args:
        documents: List of Document objects from process_pdf

    Returns:
        Total estimated token count across all documents
    """
    return sum(estimate_tokens(doc.page_content) for doc in documents)


def estimate_embedding_cost(
    documents: list[Document],
    model: str = "text-embedding-3-small",
) -> dict[str, Any]:
    """Estimate embedding cost for a list of documents.

    Pricing (as of Jan 2025):
    - text-embedding-3-small: $0.02 per 1M tokens
    - text-embedding-3-large: $0.13 per 1M tokens

    Args:
        documents: List of Document objects from process_pdf
        model: Embedding model name

    Returns:
        Dictionary with token count, cost estimate, and model info
    """
    total_tokens = estimate_documents_tokens(documents)

    # Pricing per 1M tokens
    pricing = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
        "text-embedding-ada-002": 0.10,  # Legacy model
    }

    cost_per_million = pricing.get(model, 0.02)  # Default to small model
    estimated_cost = (total_tokens / 1_000_000) * cost_per_million

    return {
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 4),
        "model": model,
        "cost_per_million_tokens": cost_per_million,
        "total_documents": len(documents),
    }
