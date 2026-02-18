"""Debug script to test LlamaParse JSON mode with bbox extraction.

Tests the fixed process_document flow for first 3 pages.

Usage:
    python debug_llamaparse_json.py
"""

import asyncio
import logging

from dotenv import load_dotenv

from rag.ingestion.document_fetcher import DocumentFetcher
from rag.ingestion.llama_parse_processor import process_document
from utils.data_helpers import initialize_metadata_data, initialize_stock_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main():
    load_dotenv()
    await initialize_stock_data()
    await initialize_metadata_data()

    # Fetch a test document
    fetcher = DocumentFetcher()
    test_fincode = 100325  # Reliance Industries
    docs = await fetcher.get_available_documents(fincode=test_fincode)
    doc = docs[7]  # concall document
    logger.info(f"Using document: {doc.filename} ({doc.category}) - {doc.document_date}")

    pdf = await fetcher.get_pdf_doc_stream(doc.filename, doc.category)
    logger.info(f"PDF size: {len(pdf.stream.getvalue()) / 1024 / 1024:.2f} MB")

    # Process with the fixed code
    result_docs = await process_document(pdf, file_type="pdf")

    logger.info(f"\n{'='*80}")
    logger.info(f"RESULTS: {len(result_docs)} documents")
    logger.info(f"{'='*80}")

    with_bbox = sum(1 for d in result_docs if d.metadata.get("bbox") is not None)
    with_page = sum(1 for d in result_docs if d.metadata.get("page") is not None)
    logger.info(f"  With bbox: {with_bbox}/{len(result_docs)}")
    logger.info(f"  With page: {with_page}/{len(result_docs)}")

    # Show a few samples
    for i, d in enumerate(result_docs[:5]):
        logger.info(f"\n--- Document {i} ---")
        logger.info(f"  chunk_type: {d.metadata.get('chunk_type')}")
        logger.info(f"  page: {d.metadata.get('page')}")
        logger.info(f"  bbox: {d.metadata.get('bbox')}")
        logger.info(f"  content preview: {d.page_content[:150]}")


if __name__ == "__main__":
    asyncio.run(main())
