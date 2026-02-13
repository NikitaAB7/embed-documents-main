"""Migration script to sync existing Qdrant data to SQLite tracker.

This one-time migration script scans your Qdrant collection and populates
the SQLite tracking database with all existing embedded documents.

Usage:
    python -m rag.ingestion.migrations.sync_qdrant_to_sqlite

    Or from code:
    from rag.ingestion.migrations.sync_qdrant_to_sqlite import migrate
    await migrate()
"""

import argparse
import asyncio
import logging
from collections import defaultdict
from typing import Any

from rag.config import rag_config
from rag.ingestion.document_tracker import DocumentTracker
from rag.ingestion.vector_store import QdrantManager
from utils.data_helpers import initialize_metadata_data, initialize_stock_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def migrate(
    collection_name: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Migrate Qdrant collection data to SQLite tracker.

    Args:
        collection_name: Qdrant collection to migrate (defaults to config)
        dry_run: If True, only report what would be migrated without writing

    Returns:
        Dictionary with migration statistics
    """
    collection = collection_name or rag_config.qdrant_collection_name

    logger.info(f"Starting migration for collection '{collection}'")
    logger.info(f"Dry run mode: {dry_run}")

    # Initialize managers
    qdrant_manager = QdrantManager(collection_name=collection)
    doc_tracker = DocumentTracker(collection_name=collection)

    # Initialize tracker database (needed even in dry-run to check existing data)
    await doc_tracker.initialize()
    logger.info("Initialized SQLite database")

    # Initialize data helpers for fincode/symbol auto-resolution
    logger.info("Initializing stock and metadata data for fincode/symbol resolution...")
    await initialize_stock_data()
    await initialize_metadata_data()
    logger.info("Data helpers initialized successfully")

    # Scan Qdrant collection once to get sources AND counts
    logger.info("Scanning Qdrant collection for sources and document counts...")
    logger.info("(Processing in batches - this may take a while for large collections)")

    source_counts = defaultdict(int)
    client = await qdrant_manager._ensure_client()

    # Scroll through all points in batches
    offset = None
    batch_num = 0
    total_points = 0

    while True:
        batch_num += 1
        logger.info(f"Processing batch {batch_num}...")

        scroll_result = await asyncio.to_thread(
            client.scroll,
            collection_name=collection,
            limit=1000,  # Process 1000 points at a time
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        points, next_offset = scroll_result

        if not points:
            break

        # Count documents by source
        for point in points:
            total_points += 1
            if point.payload and "metadata" in point.payload:
                metadata = point.payload["metadata"]
                if "source" in metadata:
                    source_counts[metadata["source"]] += 1

        # Check if there are more points
        if next_offset is None:
            break

        offset = next_offset

    logger.info(f"Processed {total_points} total points")
    logger.info(f"Found {len(source_counts)} unique sources with document counts")

    # Check if we found anything
    if len(source_counts) == 0:
        logger.warning("No sources found in Qdrant. Nothing to migrate.")
        return {
            "collection_name": collection,
            "sources_found": 0,
            "sources_migrated": 0,
            "sources_skipped": 0,
            "total_points": total_points,
            "dry_run": dry_run,
        }

    # Check what's already in SQLite
    already_tracked = await doc_tracker.get_all_sources()
    logger.info(f"Already tracked in SQLite: {len(already_tracked)} sources")

    # Migrate to SQLite
    sources_migrated = 0
    sources_skipped = 0

    for source, count in source_counts.items():
        if source in already_tracked:
            logger.debug(f"Skipping '{source}' (already tracked)")
            sources_skipped += 1
            continue

        if not dry_run:
            # Mark as embedded without cost info (we don't have historical cost data)
            await doc_tracker.mark_embedded(
                filename=source,
                document_count=count,
                total_tokens=None,  # Unknown
                embedding_cost_usd=None,  # Unknown
                metadata={"migrated": True, "migration_date": asyncio.get_event_loop().time()},
            )

        sources_migrated += 1
        logger.debug(f"Migrated '{source}' ({count} documents)")

    # Log summary
    logger.info("=" * 60)
    logger.info("Migration Summary:")
    logger.info(f"  Collection: {collection}")
    logger.info(f"  Sources in Qdrant: {len(source_counts)}")
    logger.info(f"  Sources migrated: {sources_migrated}")
    logger.info(f"  Sources skipped (already tracked): {sources_skipped}")
    logger.info(f"  Total points processed: {total_points}")
    logger.info(f"  Dry run: {dry_run}")
    logger.info("=" * 60)

    if dry_run:
        logger.info("This was a dry run. No data was written to SQLite.")
        logger.info("Run without --dry-run to perform the actual migration.")
    else:
        logger.info("Migration completed successfully!")
        logger.info("SQLite tracker is now in sync with Qdrant.")

    return {
        "collection_name": collection,
        "sources_found": len(source_counts),
        "sources_migrated": sources_migrated,
        "sources_skipped": sources_skipped,
        "total_points": total_points,
        "dry_run": dry_run,
    }


async def verify_sync(collection_name: str | None = None) -> dict[str, Any]:
    """Verify that SQLite and Qdrant are in sync.

    Args:
        collection_name: Collection to verify (defaults to config)

    Returns:
        Dictionary with verification results
    """
    collection = collection_name or rag_config.qdrant_collection_name

    logger.info(f"Verifying sync for collection '{collection}'")

    # Initialize managers
    qdrant_manager = QdrantManager(collection_name=collection)
    doc_tracker = DocumentTracker(collection_name=collection)

    # Get sources from both systems
    qdrant_sources = await qdrant_manager.get_embedded_sources()
    sqlite_sources = await doc_tracker.get_all_sources()

    # Find differences
    missing_in_sqlite = qdrant_sources - sqlite_sources
    extra_in_sqlite = sqlite_sources - qdrant_sources

    in_sync = len(missing_in_sqlite) == 0 and len(extra_in_sqlite) == 0

    # Log results
    logger.info("=" * 60)
    logger.info("Sync Verification Results:")
    logger.info(f"  Qdrant sources: {len(qdrant_sources)}")
    logger.info(f"  SQLite sources: {len(sqlite_sources)}")
    logger.info(f"  Missing in SQLite: {len(missing_in_sqlite)}")
    logger.info(f"  Extra in SQLite: {len(extra_in_sqlite)}")
    logger.info(f"  In sync: {in_sync}")
    logger.info("=" * 60)

    if not in_sync:
        if missing_in_sqlite:
            logger.warning(f"Sources in Qdrant but not SQLite: {missing_in_sqlite}")
        if extra_in_sqlite:
            logger.warning(f"Sources in SQLite but not Qdrant: {extra_in_sqlite}")

    return {
        "collection_name": collection,
        "in_sync": in_sync,
        "qdrant_count": len(qdrant_sources),
        "sqlite_count": len(sqlite_sources),
        "missing_in_sqlite": list(missing_in_sqlite),
        "extra_in_sqlite": list(extra_in_sqlite),
    }


async def main():
    """Run the main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate Qdrant data to SQLite tracker"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name (defaults to config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without making changes (preview mode)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify sync status instead of migrating",
    )

    args = parser.parse_args()

    if args.verify:
        result = await verify_sync(collection_name=args.collection)
        if result["in_sync"]:
            logger.info("✓ Systems are in sync!")
            exit(0)
        else:
            logger.error("✗ Systems are NOT in sync")
            exit(1)
    else:
        result = await migrate(
            collection_name=args.collection,
            dry_run=args.dry_run,
        )
        if result["sources_migrated"] > 0 or result["sources_skipped"] > 0:
            exit(0)
        else:
            exit(1)


if __name__ == "__main__":
    asyncio.run(main())
