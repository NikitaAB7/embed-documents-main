"""Test script to re-embed a few documents and verify bbox extraction.

This script:
1. Clears a test collection (or deletes specific documents)
2. Embeds 2-3 documents using Docling (local, free, has bbox)
3. Prints the metadata to verify bbox is captured
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[0]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv()

from qdrant_client import QdrantClient
from rag.config import rag_config
from rag.ingestion.document_fetcher import DocumentFetcher
from rag.ingestion.docling_processor import process_pdf  # Use Docling instead
from rag.ingestion.vector_store import QdrantManager
from utils.data_helpers import (
    initialize_metadata_data,
    initialize_stock_data,
    symbol_to_fincode,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TEST_COLLECTION = "bbox_test"  # Use a separate test collection


async def clear_test_collection():
    """Clear the test collection if it exists."""
    client = QdrantClient(
        url=rag_config.qdrant_url,
        api_key=rag_config.qdrant_api_key,
        prefer_grpc=True,
    )
    
    collections = client.get_collections()
    collection_names = [col.name for col in collections.collections]
    
    if TEST_COLLECTION in collection_names:
        logger.info(f"Deleting test collection '{TEST_COLLECTION}'...")
        client.delete_collection(TEST_COLLECTION)
        logger.info("Deleted.")
    else:
        logger.info(f"Test collection '{TEST_COLLECTION}' doesn't exist.")


async def embed_test_documents(symbol: str = "TCS", max_docs: int = 2):
    """Embed a few documents for testing.
    
    Args:
        symbol: Stock symbol to fetch documents for
        max_docs: Max number of documents to embed
    """
    logger.info("Initializing data helpers...")
    await initialize_stock_data()
    await initialize_metadata_data()
    
    fincode = symbol_to_fincode(symbol)
    if not fincode:
        logger.error(f"Could not find fincode for symbol {symbol}")
        return
    
    logger.info(f"Fetching documents for {symbol} (fincode: {fincode})...")
    
    fetcher = DocumentFetcher()
    documents = await fetcher.get_available_documents(fincode=fincode)
    
    logger.info(f"Found {len(documents)} documents")
    
    if not documents:
        logger.error("No documents found!")
        return
    
    # Take only first N documents
    documents = documents[:max_docs]
    logger.info(f"Will embed {len(documents)} documents")
    
    # Initialize Qdrant manager with test collection
    qdrant = QdrantManager(collection_name=TEST_COLLECTION)
    await qdrant.initialize_collection()
    
    for i, doc_meta in enumerate(documents):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing document {i+1}/{len(documents)}: {doc_meta.filename}")
        
        try:
            # Fetch the document
            doc_stream = await fetcher.get_pdf_doc_stream(doc_meta.filename, doc_meta.category)
            if not doc_stream:
                logger.error(f"Failed to fetch document: {doc_meta.filename}")
                continue
            
            # Process with Docling (local, free, captures bbox in dl_meta)
            logger.info("Processing with Docling...")
            chunks = await process_pdf(doc_stream)
            
            logger.info(f"Got {len(chunks)} chunks")
            
            # Print metadata from first few chunks to verify bbox
            logger.info("\n=== CHUNK METADATA SAMPLE ===")
            for j, chunk in enumerate(chunks[:5]):
                meta = chunk.metadata
                logger.info(f"\nChunk {j+1}:")
                logger.info(f"  source: {meta.get('source')}")
                
                # Extract page and bbox from dl_meta
                dl_meta = meta.get('dl_meta', {})
                if dl_meta:
                    doc_items = dl_meta.get('doc_items', [])
                    if doc_items:
                        prov = doc_items[0].get('prov', [])
                        if prov:
                            page_no = prov[0].get('page_no')
                            bbox = prov[0].get('bbox')
                            coord_origin = prov[0].get('coord_origin')
                            logger.info(f"  page (from dl_meta): {page_no}")
                            logger.info(f"  bbox (from dl_meta): {bbox}")
                            logger.info(f"  coord_origin: {coord_origin}")
                        else:
                            logger.info("  No prov in doc_items")
                    else:
                        logger.info("  No doc_items in dl_meta")
                else:
                    logger.info("  No dl_meta found")
                
                logger.info(f"  chunk_type: {meta.get('chunk_type', 'unknown')}")
            
            # Check if any chunk has bbox in dl_meta
            has_bbox = False
            for chunk in chunks:
                dl_meta = chunk.metadata.get('dl_meta', {})
                doc_items = dl_meta.get('doc_items', [])
                if doc_items:
                    prov = doc_items[0].get('prov', [])
                    if prov and prov[0].get('bbox'):
                        has_bbox = True
                        break
            
            logger.info(f"\n>>> HAS BBOX IN DL_META: {has_bbox} <<<")
            
            if not has_bbox:
                logger.warning("No bbox found in chunks!")
            
            # Embed the chunks
            if chunks:
                logger.info(f"Embedding {len(chunks)} chunks...")
                result = await qdrant.add_documents(chunks)
                logger.info(f"Embedded: {result}")
            
        except Exception as e:
            logger.error(f"Failed to process document: {e}", exc_info=True)
    
    logger.info("\n" + "="*60)
    logger.info("DONE! Now test the app by querying for this document.")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test bbox extraction with new LlamaParse JSON mode")
    parser.add_argument("--symbol", default="TCS", help="Stock symbol to test with")
    parser.add_argument("--max-docs", type=int, default=2, help="Max documents to embed")
    parser.add_argument("--clear-only", action="store_true", help="Only clear the test collection")
    
    args = parser.parse_args()
    
    if args.clear_only:
        await clear_test_collection()
    else:
        await clear_test_collection()
        await embed_test_documents(args.symbol, args.max_docs)


if __name__ == "__main__":
    asyncio.run(main())
