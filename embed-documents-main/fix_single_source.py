"""Fix embedded documents for a specific source file.

This script:
1. Queries Qdrant for all chunks from a specific source file
2. Looks up correct metadata using the source filename
3. Rebuilds page_content with correct prefix
4. Deletes old embeddings (if not dry run)
5. Re-embeds with corrected data (if not dry run)

Usage:
    # Dry run (default - no changes)
    python fix_single_source.py --source "b437b7ef-89ba-40b8-9e0e-25dcbb671e48.pdf"

    # Actually fix the document
    python fix_single_source.py --source "b437b7ef-89ba-40b8-9e0e-25dcbb671e48.pdf" --no-dry-run
"""

import argparse
import asyncio
import logging
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag.ingestion.vector_store import QdrantManager
from utils.data_helpers import (
    fincode_to_symbol,
    get_metadata_item_by_attachment_name,
    initialize_metadata_data,
    initialize_stock_data,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def get_documents_by_source(
    qdrant_manager: QdrantManager, source: str
) -> List[Dict[str, Any]]:
    """Query Qdrant for all documents from a specific source.

    Args:
        qdrant_manager: QdrantManager instance
        source: Source filename to query

    Returns:
        List of documents with their point_id, metadata, and page_content
    """
    client = await qdrant_manager._ensure_client()

    logger.info(f"Querying Qdrant for source: {source}")

    # Scroll through all points with the specific source
    offset = None
    all_points = []

    while True:
        scroll_result = await asyncio.to_thread(
            client.scroll,
            collection_name=qdrant_manager.collection_name,
            limit=100,
            offset=offset,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.source",
                        match=MatchValue(value=source),
                    )
                ]
            ),
            with_payload=True,
            with_vectors=False,
        )

        points, next_offset = scroll_result
        all_points.extend(points)

        if next_offset is None:
            break
        offset = next_offset

    logger.info(f"Found {len(all_points)} chunks for source '{source}'")

    # Convert to dict format
    documents = []
    for point in all_points:
        if point.payload:
            documents.append({
                "point_id": point.id,
                "metadata": point.payload.get("metadata", {}),
                "page_content": point.payload.get("page_content", ""),
            })

    return documents


def analyze_document(doc_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a document to check its current metadata status.

    Args:
        doc_data: Document data with metadata and page_content

    Returns:
        Dictionary with analysis results
    """
    metadata = doc_data["metadata"]
    page_content = doc_data["page_content"]

    # Check current state
    has_missing = (
        metadata.get("fincode") is None
        or metadata.get("ticker") is None
        or metadata.get("category") == ""
        or metadata.get("document_date") == ""
    )

    # Extract current prefix (first line)
    first_line = page_content.split("\n", 1)[0] if "\n" in page_content else ""

    return {
        "has_missing_metadata": has_missing,
        "current_fincode": metadata.get("fincode"),
        "current_ticker": metadata.get("ticker"),
        "current_category": metadata.get("category"),
        "current_document_date": metadata.get("document_date"),
        "current_prefix": first_line,
    }


def fix_document(doc_data: Dict[str, Any]) -> Document:
    """Fix a single document's metadata and page_content prefix.

    Args:
        doc_data: Document data with metadata and page_content

    Returns:
        Fixed Document with corrected metadata and prefix
    """
    metadata = doc_data["metadata"].copy()
    page_content = doc_data["page_content"]
    source = metadata.get("source", "")

    # Look up correct metadata
    file_meta = get_metadata_item_by_attachment_name(source) or {}

    # Extract fincode, category, document_date from file metadata
    fincode = file_meta.get("fincode", None)
    category = file_meta.get("subcatname", "")
    document_date = (
        file_meta.get("newsDt", "")[:10] if file_meta.get("newsDt") else ""
    )

    # Get ticker from fincode
    ticker = None
    if fincode:
        ticker = fincode_to_symbol(fincode)

    # Extract page number from old prefix
    page_no_match = re.search(r"page no: (\d+)", page_content)
    page_no = page_no_match.group(1) if page_no_match else ""

    # Remove old prefix (everything before first newline)
    content_parts = page_content.split("\n", 1)
    actual_content = content_parts[1] if len(content_parts) > 1 else page_content

    # Build new prefix
    new_prefix = f"Ticker: {ticker}; Type: {category}; Date: {document_date}; fincode: {fincode}; page no: {page_no}\n"
    new_page_content = new_prefix + actual_content

    # Update metadata
    metadata["fincode"] = fincode
    metadata["ticker"] = ticker
    metadata["category"] = category
    metadata["document_date"] = document_date

    return Document(page_content=new_page_content, metadata=metadata)


async def fix_source(source: str, dry_run: bool = True):
    """Fix all documents from a specific source.

    Args:
        source: Source filename to fix
        dry_run: If True, only show what would be done without making changes
    """
    load_dotenv()

    # Initialize data helpers
    logger.info("Initializing stock data and metadata...")
    await initialize_stock_data()
    await initialize_metadata_data()

    # Initialize Qdrant manager
    qdrant_manager = QdrantManager()

    # Get all documents for this source
    docs = await get_documents_by_source(qdrant_manager, source)

    if not docs:
        logger.warning(f"No documents found for source '{source}'")
        return

    logger.info(f"\n{'='*80}")
    logger.info(f"Found {len(docs)} chunks for source: {source}")
    logger.info(f"{'='*80}\n")

    # Analyze first document
    logger.info("Current state of first chunk:")
    analysis = analyze_document(docs[0])
    logger.info(f"  Has missing metadata: {analysis['has_missing_metadata']}")
    logger.info(f"  Current fincode: {analysis['current_fincode']}")
    logger.info(f"  Current ticker: {analysis['current_ticker']}")
    logger.info(f"  Current category: {analysis['current_category']}")
    logger.info(f"  Current date: {analysis['current_document_date']}")
    logger.info(f"  Current prefix: {analysis['current_prefix']}")

    # Fix all documents
    logger.info(f"\nFixing {len(docs)} documents...")
    fixed_docs = [fix_document(doc_data) for doc_data in docs]

    # Show what the fix will look like
    logger.info("\nFixed state of first chunk:")
    sample = fixed_docs[0]
    logger.info(f"  Fincode: {sample.metadata.get('fincode')}")
    logger.info(f"  Ticker: {sample.metadata.get('ticker')}")
    logger.info(f"  Category: {sample.metadata.get('category')}")
    logger.info(f"  Date: {sample.metadata.get('document_date')}")
    new_prefix = sample.page_content.split("\n", 1)[0]
    logger.info(f"  New prefix: {new_prefix}")
    logger.info(f"\n  Content preview: {sample.page_content[:200]}...")

    if dry_run:
        logger.warning(f"\n{'='*80}")
        logger.warning("DRY RUN MODE - No changes will be made")
        logger.warning(f"{'='*80}")
        logger.info(f"\nWould delete {len(docs)} old chunks and re-embed {len(fixed_docs)} fixed chunks")
        logger.info("Run with --no-dry-run to actually fix these documents")
        return

    # Actually fix the documents
    logger.warning(f"\n{'='*80}")
    logger.warning("LIVE MODE - Making actual changes!")
    logger.warning(f"{'='*80}\n")

    # Delete old embeddings
    logger.info(f"Deleting old embeddings for '{source}'...")
    delete_result = await qdrant_manager.delete_by_source(source)
    logger.info(f"✓ Deleted {delete_result['deleted_count']} points")

    # Re-embed with correct data
    logger.info(f"\nRe-embedding {len(fixed_docs)} documents...")
    embed_result = await qdrant_manager.embed_documents(fixed_docs, show_progress=True)
    logger.info(
        f"✓ Embedded {embed_result['total_embedded']} documents\n"
        f"  Total tokens: {embed_result['total_tokens']}\n"
        f"  Estimated cost: ${embed_result['estimated_cost_usd']:.4f}"
    )

    logger.info(f"\n{'='*80}")
    logger.info(f"✓ Successfully fixed source: {source}")
    logger.info(f"{'='*80}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fix embedded documents for a specific source file"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source filename to fix (e.g., 'document.pdf')",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually fix the documents (default is dry run)",
    )

    args = parser.parse_args()

    # Run async function
    asyncio.run(fix_source(args.source, dry_run=not args.no_dry_run))


if __name__ == "__main__":
    main()
