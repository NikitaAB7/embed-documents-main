#!/usr/bin/env python3
"""Chunk documents using LlamaParse and embed them in Qdrant in parallel.

Usage:
    python chunk_and_embed_parallel.py --fincodes 100325 100348 --collection compliance_docs
    python chunk_and_embed_parallel.py --group "Nifty 50 Index" --collection nifty50_docs
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
from joblib import Parallel, delayed
from langchain_core.documents import Document

from rag.ingestion.llama_parse_processor import process_document
from rag.ingestion.document_fetcher import DocumentFetcher
from rag.ingestion.document_tracker import DocumentTracker
from rag.ingestion.vector_store import QdrantManager
from utils.data_helpers import (
    initialize_metadata_data,
    initialize_stock_data,
    symbol_to_fincode,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATE_FMT = "%Y-%m-%d"
DEFAULT_DATE_WINDOW_DAYS = 730
NIFTY_50_GROUP_NAME = "Nifty 50 Index"


class ChunkAndEmbedPipeline:
    """Pipeline to chunk documents with LlamaParse and embed them in Qdrant."""

    def __init__(self, collection_name: str = "compliance_docs", max_workers: Optional[int] = None):
        """Initialize the pipeline.
        
        Args:
            collection_name: Name of the Qdrant collection
        """
        load_dotenv()
        self.collection_name = collection_name
        self.fetcher = DocumentFetcher()
        self.tracker = DocumentTracker(collection_name=collection_name)
        self.qdrant = QdrantManager(collection_name=collection_name)
        self._group_fincode_cache: dict[str, list[int]] = {}

        cpu_count = os.cpu_count() or 1
        default_workers = max(1, int(cpu_count * 0.75))
        if max_workers is not None:
            self.max_workers = max(1, min(max_workers, cpu_count))
            logger.info(
                "Using %s parallel workers (user-specified, max cores %s)",
                self.max_workers,
                cpu_count,
            )
        else:
            self.max_workers = default_workers
            logger.info(
                "Using %s parallel workers (75%% of %s cores)",
                self.max_workers,
                cpu_count,
            )

    async def initialize(self):
        """Initialize Qdrant collection and data helpers."""
        logger.info("Initializing Qdrant collection...")
        await self.qdrant.initialize_collection()
        
        logger.info("Initializing stock data...")
        await initialize_stock_data()
        
        logger.info("Initializing metadata...")
        await initialize_metadata_data()

    def _process_document_sync(self, pdf_stream, filename: str) -> tuple[list[Document], dict]:
        """Process a single document synchronously for joblib parallelization.
        
        Args:
            pdf_stream: The PDF document stream
            filename: The filename for logging/identification
            
        Returns:
            Tuple of (documents, diagnostics)
        """
        diagnostics = {
            "filename": filename,
            "error": None,
            "chunk_count": 0,
            "with_bbox": 0,
        }

        try:
            # Initialize in worker process
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Process the document with LlamaParse (JSON mode with bbox)
            result_docs = loop.run_until_complete(
                process_document(pdf_stream, file_type="pdf", use_json_mode=True)
            )
            
            diagnostics["chunk_count"] = len(result_docs)
            diagnostics["with_bbox"] = sum(
                1 for d in result_docs if d.metadata.get("bbox") is not None
            )
            
            logger.info(
                f"✓ {filename}: {len(result_docs)} chunks "
                f"({diagnostics['with_bbox']} with bbox)"
            )
            
            return result_docs, diagnostics

        except Exception as e:
            diagnostics["error"] = str(e)
            logger.error(f"✗ {filename}: {e}")
            return [], diagnostics
        finally:
            loop.close()

    async def fetch_documents_for_fincodes(
        self,
        fincodes: list[int],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[dict]:
        """Fetch documents for given financial codes.
        
        Args:
            fincodes: List of financial codes
            
        Returns:
            List of document metadata
        """
        docs = []
        for fincode in fincodes:
            logger.info(f"Fetching documents for fincode {fincode}...")
            available_docs = await self.fetcher.get_available_documents(
                fincode=fincode,
                start_date=start_date,
                end_date=end_date,
            )
            docs.extend(available_docs)
        
        logger.info(f"Total documents fetched: {len(docs)}")
        return docs

    async def _get_group_fincodes(self, group_name: str) -> list[int]:
        """Resolve fincodes for a predefined group and cache the result."""
        if group_name in self._group_fincode_cache:
            return self._group_fincode_cache[group_name]

        groups = await self.fetcher.api.get_predefined_groups()

        if group_name not in groups:
            raise ValueError(
                f"Group '{group_name}' not found. Available: {list(groups.keys())}"
            )

        symbols = groups[group_name]
        logger.info("Group '%s' has %s symbols", group_name, len(symbols))

        fincodes: list[int] = []
        for symbol in symbols:
            try:
                fincode = symbol_to_fincode(symbol)
                if fincode:
                    fincodes.append(fincode)
            except Exception as e:
                logger.warning("Could not convert symbol %s to fincode: %s", symbol, e)

        unique_fincodes = sorted(set(fincodes))
        self._group_fincode_cache[group_name] = unique_fincodes
        logger.info("Resolved group '%s' to %s fincodes", group_name, len(unique_fincodes))
        return unique_fincodes

    async def fetch_documents_for_group(
        self,
        group_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[dict]:
        """Fetch documents for all companies in a predefined group.
        
        Args:
            group_name: Name of the predefined group (e.g., "Nifty 50 Index")
            
        Returns:
            List of document metadata
        """
        fincodes = await self._get_group_fincodes(group_name)
        return await self.fetch_documents_for_fincodes(
            fincodes,
            start_date=start_date,
            end_date=end_date,
        )

    async def fetch_all_documents(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[dict]:
        """Fetch all available documents within an optional date range."""
        logger.info("Fetching all available documents...")
        docs = await self.fetcher.get_available_documents(
            start_date=start_date,
            end_date=end_date,
        )
        logger.info(f"Total documents fetched: {len(docs)}")
        return docs

    async def filter_documents_for_group(
        self,
        documents: list,
        group_name: str,
    ) -> list:
        """Filter already-fetched documents down to a specific predefined group."""
        allowed_fincodes = set(await self._get_group_fincodes(group_name))
        filtered = [doc for doc in documents if doc.fincode in allowed_fincodes]
        logger.info(
            "Filtered documents for group '%s': kept %s of %s",
            group_name,
            len(filtered),
            len(documents),
        )
        return filtered

    def _resolve_date_range(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> tuple[str, str]:
        """Resolve date inputs to a concrete window (defaults to 2 years)."""
        today = datetime.utcnow().date()
        resolved_end = (
            datetime.strptime(end_date, DATE_FMT).date() if end_date else today
        )
        resolved_start = (
            datetime.strptime(start_date, DATE_FMT).date()
            if start_date
            else resolved_end - timedelta(days=DEFAULT_DATE_WINDOW_DAYS)
        )

        if resolved_start > resolved_end:
            logger.warning(
                "Start date %s is after end date %s; swapping values.",
                resolved_start,
                resolved_end,
            )
            resolved_start, resolved_end = resolved_end, resolved_start

        return resolved_start.strftime(DATE_FMT), resolved_end.strftime(DATE_FMT)

    def _generate_date_batches(
        self,
        start_date: str,
        end_date: str,
        batch_days: Optional[int],
    ) -> list[tuple[str, str]]:
        """Generate inclusive date batches for processing."""
        if not batch_days or batch_days <= 0:
            return [(start_date, end_date)]

        start = datetime.strptime(start_date, DATE_FMT).date()
        end = datetime.strptime(end_date, DATE_FMT).date()

        batches: list[tuple[str, str]] = []
        cursor = start
        while cursor <= end:
            batch_end = min(cursor + timedelta(days=batch_days - 1), end)
            batches.append(
                (cursor.strftime(DATE_FMT), batch_end.strftime(DATE_FMT))
            )
            cursor = batch_end + timedelta(days=1)

        logger.info(
            "Prepared %s date batches spanning %s to %s (batch size %s days)",
            len(batches),
            start_date,
            end_date,
            batch_days,
        )
        return batches

    async def process_and_embed(self, documents: list[dict]):
        """Process documents with LlamaParse and embed them in Qdrant in parallel.
        
        Args:
            documents: List of document metadata from fetcher
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {len(documents)} documents with LlamaParse...")
        logger.info(f"{'='*80}\n")
        
        # Skip documents that are already embedded in this collection
        filenames = [doc.filename for doc in documents]
        existence_map = await self.tracker.check_documents_exist(
            filenames, collection_name=self.collection_name
        )
        documents_to_process = [
            doc for doc in documents if not existence_map.get(doc.filename, False)
        ]
        skipped = len(documents) - len(documents_to_process)

        if skipped:
            logger.info(
                f"Skipping {skipped} documents already embedded in '{self.collection_name}'."
            )

        if not documents_to_process:
            logger.info("All documents are already embedded. Nothing to do.")
            return
        
        logger.info(f"{len(documents_to_process)} documents remain to be processed.")
        
        # Fetch all PDFs
        pdf_streams = []
        
        for doc in documents_to_process:
            try:
                logger.info(f"Fetching {doc.filename}...")
                pdf_stream = await self.fetcher.get_pdf_doc_stream(
                    doc.filename, doc.category
                )
                pdf_streams.append((pdf_stream, doc.filename))
            except Exception as e:
                logger.error(f"Failed to fetch {doc.filename}: {e}")
        
        logger.info(f"Successfully fetched {len(pdf_streams)} PDFs\n")
        
        # Process PDFs in parallel with joblib
        logger.info(f"Processing {len(pdf_streams)} PDFs with LlamaParse in parallel...")
        results = Parallel(n_jobs=self.max_workers, backend="threading")(
            delayed(self._process_document_sync)(stream, filename)
            for stream, filename in pdf_streams
        )
        
        # Collect all documents and diagnostic info
        all_chunks = []
        total_chunks = 0
        total_with_bbox = 0
        errors = 0
        
        for result_docs, diagnostics in results:
            all_chunks.extend(result_docs)
            total_chunks += diagnostics["chunk_count"]
            total_with_bbox += diagnostics["with_bbox"]
            if diagnostics["error"]:
                errors += 1
        
        logger.info(f"\n{'='*80}")
        logger.info(f"LlamaParse Processing Summary:")
        logger.info(f"  Total chunks: {total_chunks}")
        logger.info(f"  Chunks with bbox: {total_with_bbox}")
        logger.info(f"  Processing errors: {errors}")
        logger.info(f"{'='*80}\n")
        
        if not all_chunks:
            logger.warning("No chunks produced! Exiting.")
            return
        
        # Embed chunks in Qdrant
        logger.info(f"Embedding {len(all_chunks)} chunks in Qdrant collection '{self.collection_name}'...")
        embed_result = await self.qdrant.embed_documents(all_chunks)
        logger.info(f"Embedding result: {embed_result}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✓ Pipeline complete!")
        logger.info(f"  Processed: {len(pdf_streams)} documents")
        logger.info(f"  Generated: {len(all_chunks)} chunks")
        logger.info(f"  Collection: {self.collection_name}")
        logger.info(f"{'='*80}")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Chunk and embed documents using LlamaParse and Qdrant"
    )
    parser.add_argument(
        "--fincodes",
        nargs="+",
        type=int,
        help="Financial codes to process (e.g., 100325 100348)"
    )
    parser.add_argument(
        "--group",
        type=str,
        help="Predefined group name to process (e.g., 'Nifty 50 Index')"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available documents (optionally filtered by date range)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="compliance_docs",
        help="Qdrant collection name (default: compliance_docs)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date filter (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date filter (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Override the default parallel worker count (<= CPU cores)."
    )
    parser.add_argument(
        "--batch-days",
        type=int,
        help="Split the date range into sequential batches of N days."
    )
    parser.add_argument(
        "--nifty50-only",
        action="store_true",
        help="Filter fetched documents down to the Nifty 50 Index companies."
    )
    
    args = parser.parse_args()
    
    if not args.fincodes and not args.group and not args.all:
        parser.print_help()
        return
    
    # Initialize pipeline
    if args.nifty50_only and args.group and args.group != NIFTY_50_GROUP_NAME:
        parser.error("--nifty50-only cannot be combined with a different --group")

    pipeline = ChunkAndEmbedPipeline(
        collection_name=args.collection,
        max_workers=args.max_workers,
    )
    await pipeline.initialize()
    resolved_start, resolved_end = pipeline._resolve_date_range(
        args.start_date, args.end_date
    )
    date_batches = pipeline._generate_date_batches(
        resolved_start, resolved_end, args.batch_days
    )

    total_documents = 0
    nifty_filter_group = (
        NIFTY_50_GROUP_NAME if args.nifty50_only and args.group != NIFTY_50_GROUP_NAME else None
    )

    for idx, (batch_start, batch_end) in enumerate(date_batches, start=1):
        logger.info(
            "Processing date batch %s/%s: %s → %s",
            idx,
            len(date_batches),
            batch_start,
            batch_end,
        )

        if args.group:
            logger.info(f"Processing group: {args.group}")
            documents = await pipeline.fetch_documents_for_group(
                args.group,
                start_date=batch_start,
                end_date=batch_end,
            )
        elif args.fincodes:
            logger.info(f"Processing fincodes: {args.fincodes}")
            documents = await pipeline.fetch_documents_for_fincodes(
                args.fincodes,
                start_date=batch_start,
                end_date=batch_end,
            )
        else:
            logger.info("Processing all available documents")
            documents = await pipeline.fetch_all_documents(
                start_date=batch_start,
                end_date=batch_end,
            )

        if not documents:
            logger.warning("No documents found to process for this batch!")
            continue

        if nifty_filter_group:
            documents = await pipeline.filter_documents_for_group(
                documents,
                nifty_filter_group,
            )
            if not documents:
                logger.warning(
                    "No documents matched the Nifty 50 filter in this batch; skipping."
                )
                continue

        await pipeline.process_and_embed(documents)
        total_documents += len(documents)

    if not total_documents:
        logger.warning("No documents found to process across all batches!")


if __name__ == "__main__":
    asyncio.run(main())
