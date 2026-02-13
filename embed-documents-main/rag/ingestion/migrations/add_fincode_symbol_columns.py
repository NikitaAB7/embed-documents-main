"""Migration script to add fincode and symbol columns to embedded_documents table.

This migration adds two new columns to track financial identifiers:
- fincode: Financial code (integer)
- symbol: Stock symbol (text)

These are automatically populated from metadata for new documents.

Usage:
    python -m rag.ingestion.migrations.add_fincode_symbol_columns

    Or from code:
    from rag.ingestion.migrations.add_fincode_symbol_columns import migrate
    await migrate()
"""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any

import aiosqlite

from rag.ingestion.document_tracker import DEFAULT_DB_PATH
from utils.data_helpers import (
    fincode_to_symbol,
    get_metadata_item_by_attachment_name,
    initialize_metadata_data,
    initialize_stock_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def migrate(
    db_path: Path | None = None,
    backfill: bool = True,
) -> dict[str, Any]:
    """Add fincode and symbol columns to embedded_documents table.

    Args:
        db_path: Path to SQLite database (defaults to DEFAULT_DB_PATH)
        backfill: If True, attempt to populate columns for existing records

    Returns:
        Dictionary with migration statistics
    """
    db_path = db_path or DEFAULT_DB_PATH

    logger.info(f"Starting migration for database at {db_path}")
    logger.info(f"Backfill existing records: {backfill}")

    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        return {
            "success": False,
            "error": "Database not found",
        }

    # Initialize data helpers for backfill
    if backfill:
        logger.info("Initializing stock and metadata data...")
        await initialize_stock_data()
        await initialize_metadata_data()
        logger.info("Data initialized successfully")

    # Add columns to schema
    async with aiosqlite.connect(db_path) as db:
        # Check if columns already exist
        cursor = await db.execute("PRAGMA table_info(embedded_documents)")
        columns = await cursor.fetchall()
        column_names = [col[1] for col in columns]

        columns_added = []

        if "fincode" not in column_names:
            logger.info("Adding 'fincode' column...")
            await db.execute(
                "ALTER TABLE embedded_documents ADD COLUMN fincode INTEGER"
            )
            columns_added.append("fincode")
            logger.info("✓ Added 'fincode' column")
        else:
            logger.info("Column 'fincode' already exists, skipping")

        if "symbol" not in column_names:
            logger.info("Adding 'symbol' column...")
            await db.execute(
                "ALTER TABLE embedded_documents ADD COLUMN symbol TEXT"
            )
            columns_added.append("symbol")
            logger.info("✓ Added 'symbol' column")
        else:
            logger.info("Column 'symbol' already exists, skipping")

        await db.commit()

    logger.info(f"Schema migration completed. Added columns: {columns_added}")

    # Backfill existing records
    records_updated = 0
    records_skipped = 0
    records_failed = 0

    if backfill and columns_added:
        logger.info("Starting backfill for existing records...")

        async with aiosqlite.connect(db_path) as db:
            # Get all records that don't have fincode/symbol set
            cursor = await db.execute(
                """
                SELECT source_filename, collection_name
                FROM embedded_documents
                WHERE fincode IS NULL OR symbol IS NULL
                """
            )
            records = await cursor.fetchall()

            logger.info(f"Found {len(records)} records to backfill")

            for source_filename, collection_name in records:
                try:
                    # Try to get metadata from attachment name
                    metadata_item = get_metadata_item_by_attachment_name(source_filename)

                    if metadata_item and "fincode" in metadata_item:
                        fincode = metadata_item["fincode"]
                        symbol = fincode_to_symbol(fincode)

                        # Update record
                        await db.execute(
                            """
                            UPDATE embedded_documents
                            SET fincode = ?, symbol = ?
                            WHERE source_filename = ? AND collection_name = ?
                            """,
                            (fincode, symbol, source_filename, collection_name),
                        )

                        records_updated += 1
                        logger.debug(
                            f"Updated '{source_filename}': fincode={fincode}, symbol={symbol}"
                        )
                    else:
                        records_skipped += 1
                        logger.debug(
                            f"Skipped '{source_filename}': metadata not found"
                        )

                except Exception as e:
                    records_failed += 1
                    logger.warning(
                        f"Failed to update '{source_filename}': {e}"
                    )

            await db.commit()

        logger.info(f"Backfill completed: {records_updated} updated, {records_skipped} skipped, {records_failed} failed")

    # Create indexes for the new columns
    logger.info("Creating indexes for new columns...")
    async with aiosqlite.connect(db_path) as db:
        try:
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fincode
                ON embedded_documents(fincode)
                """
            )
            logger.info("✓ Created index on fincode")
        except Exception as e:
            logger.warning(f"Failed to create fincode index: {e}")

        try:
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_symbol
                ON embedded_documents(symbol)
                """
            )
            logger.info("✓ Created index on symbol")
        except Exception as e:
            logger.warning(f"Failed to create symbol index: {e}")

        await db.commit()

    # Log summary
    logger.info("=" * 60)
    logger.info("Migration Summary:")
    logger.info(f"  Database: {db_path}")
    logger.info(f"  Columns added: {', '.join(columns_added) if columns_added else 'None (already exist)'}")
    logger.info(f"  Records updated: {records_updated}")
    logger.info(f"  Records skipped: {records_skipped}")
    logger.info(f"  Records failed: {records_failed}")
    logger.info("=" * 60)
    logger.info("Migration completed successfully!")

    return {
        "success": True,
        "db_path": str(db_path),
        "columns_added": columns_added,
        "records_updated": records_updated,
        "records_skipped": records_skipped,
        "records_failed": records_failed,
    }


async def verify_migration(db_path: Path | None = None) -> dict[str, Any]:
    """Verify that the migration was successful.

    Args:
        db_path: Path to SQLite database (defaults to DEFAULT_DB_PATH)

    Returns:
        Dictionary with verification results
    """
    db_path = db_path or DEFAULT_DB_PATH

    logger.info(f"Verifying migration for database at {db_path}")

    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        return {
            "success": False,
            "error": "Database not found",
        }

    async with aiosqlite.connect(db_path) as db:
        # Check if columns exist
        cursor = await db.execute("PRAGMA table_info(embedded_documents)")
        columns = await cursor.fetchall()
        column_names = [col[1] for col in columns]

        has_fincode = "fincode" in column_names
        has_symbol = "symbol" in column_names

        # Check indexes
        cursor = await db.execute("PRAGMA index_list(embedded_documents)")
        indexes = await cursor.fetchall()
        index_names = [idx[1] for idx in indexes]

        has_fincode_index = "idx_fincode" in index_names
        has_symbol_index = "idx_symbol" in index_names

        # Count records with fincode/symbol
        cursor = await db.execute(
            """
            SELECT
                COUNT(*) as total,
                COUNT(fincode) as with_fincode,
                COUNT(symbol) as with_symbol
            FROM embedded_documents
            """
        )
        stats = await cursor.fetchone()
        total, with_fincode, with_symbol = stats

    # Log results
    logger.info("=" * 60)
    logger.info("Verification Results:")
    logger.info(f"  Fincode column exists: {'✓' if has_fincode else '✗'}")
    logger.info(f"  Symbol column exists: {'✓' if has_symbol else '✗'}")
    logger.info(f"  Fincode index exists: {'✓' if has_fincode_index else '✗'}")
    logger.info(f"  Symbol index exists: {'✓' if has_symbol_index else '✗'}")
    logger.info(f"  Total records: {total}")
    logger.info(f"  Records with fincode: {with_fincode} ({with_fincode/total*100 if total else 0:.1f}%)")
    logger.info(f"  Records with symbol: {with_symbol} ({with_symbol/total*100 if total else 0:.1f}%)")
    logger.info("=" * 60)

    success = has_fincode and has_symbol

    if success:
        logger.info("✓ Migration verification passed!")
    else:
        logger.error("✗ Migration verification failed!")

    return {
        "success": success,
        "has_fincode_column": has_fincode,
        "has_symbol_column": has_symbol,
        "has_fincode_index": has_fincode_index,
        "has_symbol_index": has_symbol_index,
        "total_records": total,
        "records_with_fincode": with_fincode,
        "records_with_symbol": with_symbol,
    }


async def main():
    """Run the main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Add fincode and symbol columns to embedded_documents table"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to SQLite database (defaults to DEFAULT_DB_PATH)",
    )
    parser.add_argument(
        "--no-backfill",
        action="store_true",
        help="Skip backfilling existing records",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify migration instead of running it",
    )

    args = parser.parse_args()

    if args.verify:
        result = await verify_migration(db_path=args.db_path)
        exit(0 if result["success"] else 1)
    else:
        result = await migrate(
            db_path=args.db_path,
            backfill=not args.no_backfill,
        )
        exit(0 if result["success"] else 1)


if __name__ == "__main__":
    asyncio.run(main())
