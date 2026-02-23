"""Embed all documents for a specific category (e.g., all concalls for a company)."""

import asyncio
import logging

from dotenv import load_dotenv
from langchain_core.documents import Document

from rag.ingestion.document_fetcher import DocumentFetcher
from rag.ingestion.llama_parse_processor import process_document
from rag.ingestion.vector_store import QdrantManager
from utils.data_helpers import initialize_metadata_data, initialize_stock_data

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============ CONFIGURATION ============
# Set FINCODE to None to embed all companies, or specify a fincode for one company
FINCODE = None  # None = all companies, or e.g. 100325 for Reliance
TARGET_CATEGORY = "concall"  # Options: "annual-report", "concall", "investor-presentation"
MAX_DOCS = 10  # Limit number of documents to process (None for all)
# =======================================


async def main():
    """Embed all documents for a specific category."""
    # Initialize data
    await initialize_stock_data()
    await initialize_metadata_data()

    fetcher = DocumentFetcher()
    qdrant_manager = QdrantManager()
    doc_tracker = qdrant_manager.doc_tracker

    # Fetch all documents for this category (optionally filtered by company)
    fincode_desc = f"fincode={FINCODE}" if FINCODE else "all companies"
    logger.info(f"Fetching documents for {fincode_desc}, category={TARGET_CATEGORY}")
    docs = await fetcher.get_available_documents(
        fincode=FINCODE,
        category=TARGET_CATEGORY,
    )

    logger.info(f"Found {len(docs)} documents of category '{TARGET_CATEGORY}'")

    if not docs:
        logger.warning("No documents found. Exiting.")
        return

    # Check which documents are already embedded
    exists = await qdrant_manager.check_documents_exist([doc.filename for doc in docs])
    docs_to_process = [doc for doc in docs if not exists[doc.filename]]
    
    # Apply MAX_DOCS limit
    if MAX_DOCS and len(docs_to_process) > MAX_DOCS:
        logger.info(f"Limiting to {MAX_DOCS} documents (out of {len(docs_to_process)} available)")
        docs_to_process = docs_to_process[:MAX_DOCS]
    
    logger.info(
        f"Processing {len(docs_to_process)} new documents "
        f"(skipping {len(docs) - len(docs_to_process)} already embedded or over limit)"
    )

    for doc in docs_to_process:
        logger.info(f"  - {doc.filename} (fincode={doc.fincode}, date={doc.document_date})")

    if not docs_to_process:
        logger.info("All documents already embedded. Nothing to do.")
        return

    # Process documents
    processed_docs: list[Document] = []

    for i, doc in enumerate(docs_to_process, 1):
        try:
            logger.info(f"[{i}/{len(docs_to_process)}] Processing: {doc.filename}")

            # Try cache first (no download needed)
            cached = await doc_tracker.get_cached_documents(doc.filename)
            if cached is not None:
                logger.info(f"  Cache hit: {len(cached)} chunks")
                processed_docs.extend(cached)
                continue

            # Cache miss: download, parse, cache
            logger.info(f"  Cache miss — downloading and parsing via LlamaParse")
            pdf = await fetcher.get_pdf_doc_stream(doc.filename, doc.category)

            file_size = len(pdf.stream.getbuffer()) / 1024 / 1024
            logger.info(f"  Document size: {file_size:.2f} MB")

            parsed = await process_document(pdf, file_type="pdf")
            if parsed:
                await doc_tracker.cache_parsed_documents(doc.filename, parsed)
                logger.info(f"  Parsed into {len(parsed)} chunks")
            processed_docs.extend(parsed)
        except Exception as e:
            logger.error(f"Error processing document {doc.filename}: {e}")
            continue

    logger.info(f"Total chunks ready to embed: {len(processed_docs)}")

    # Filter to only docs that don't exist yet (double-check)
    docs_to_embed: list[Document] = []
    for doc in processed_docs:
        source = doc.metadata.get("source", "")
        if source and not exists.get(source, False):
            docs_to_embed.append(doc)

    logger.info(f"Documents to embed: {len(docs_to_embed)} chunks")

    # Embed documents
    if docs_to_embed:
        result = await qdrant_manager.embed_documents(docs_to_embed)
        logger.info(f"Embedding complete. Cost: ${result['estimated_cost_usd']:.4f}")
    else:
        logger.info("No documents to embed.")


if __name__ == "__main__":
    asyncio.run(main())
