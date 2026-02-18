"""SQLite-based Document Tracking System for RAG Pipeline.

Provides fast, local tracking of embedded documents to avoid expensive
Qdrant queries for existence checks. Uses SQLite for O(log n) lookups
instead of O(n) collection scans.

Key features:
- Fast existence checks using indexed lookups
- File hash tracking for change detection
- Cost and statistics tracking
- Multi-collection support
- Async-safe operations
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiosqlite
from langchain_core.documents import Document

from rag.config import rag_config
from utils.data_helpers import (
    fincode_to_symbol,
    get_metadata_item_by_attachment_name,
    initialize_metadata_data,
    initialize_stock_data,
    is_metadata_initialized,
    is_stock_data_initialized,
)

logger = logging.getLogger(__name__)

# Default database location (can be overridden)
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "embedded_docs.db"


class DocumentTracker:
    """Manages SQLite database for tracking embedded documents."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        collection_name: Optional[str] = None,
    ):
        """Initialize document tracker.

        Args:
            db_path: Path to SQLite database file (defaults to DEFAULT_DB_PATH)
            collection_name: Qdrant collection name (defaults to config)
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.collection_name = collection_name or rag_config.qdrant_collection_name

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized DocumentTracker with database at {self.db_path}")

    async def initialize(self) -> None:
        """Initialize database schema.

        Creates tables and indexes if they don't exist.
        Safe to call multiple times (idempotent).
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Enable WAL mode for better concurrent read/write
            await db.execute("PRAGMA journal_mode=WAL")

            # Create main tracking table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS embedded_documents (
                    source_filename TEXT PRIMARY KEY,
                    file_hash TEXT,
                    embedded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    document_count INTEGER,
                    total_tokens INTEGER,
                    embedding_cost_usd REAL,
                    collection_name TEXT,
                    metadata_json TEXT
                )
                """
            )

            # Create indexes for fast lookups
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_collection
                ON embedded_documents(collection_name)
                """
            )

            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embedded_at
                ON embedded_documents(embedded_at)
                """
            )

            # Create parsed documents cache table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS parsed_documents_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_filename TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page_content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_filename, chunk_index)
                )
                """
            )

            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cache_source
                ON parsed_documents_cache(source_filename)
                """
            )

            await db.commit()

        logger.info("Database schema initialized successfully")

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hash of a file.

        Args:
            file_path: Path to file

        Returns:
            Hexadecimal hash string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def resolve_fincode_and_symbol(filename: str) -> tuple[Optional[int], Optional[str]]:
        """Resolve fincode and symbol from a filename using metadata.

        Args:
            filename: Source filename (attachment name)

        Returns:
            Tuple of (fincode, symbol). Both None if not found or data not initialized.

        Note:
            Requires initialize_stock_data() and initialize_metadata_data() to be called first.
        """
        try:
            # Check if data helpers are initialized
            if not is_metadata_initialized() or not is_stock_data_initialized():
                logger.debug(
                    "Stock/metadata data not initialized, cannot resolve fincode/symbol"
                )
                return None, None

            # Get metadata item by attachment name
            metadata_item = get_metadata_item_by_attachment_name(filename)

            if not metadata_item or "fincode" not in metadata_item:
                logger.debug(f"No metadata found for '{filename}'")
                return None, None

            fincode = metadata_item["fincode"]
            symbol = fincode_to_symbol(fincode)

            logger.debug(f"Resolved '{filename}': fincode={fincode}, symbol={symbol}")
            return fincode, symbol

        except Exception as e:
            logger.warning(f"Failed to resolve fincode/symbol for '{filename}': {e}")
            return None, None

    async def is_embedded(
        self,
        filename: str,
        collection_name: Optional[str] = None,
    ) -> bool:
        """Check if a document is already embedded.

        Args:
            filename: Source filename to check
            collection_name: Collection name (defaults to instance collection)

        Returns:
            True if document is embedded, False otherwise
        """
        collection = collection_name or self.collection_name

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT 1 FROM embedded_documents
                WHERE source_filename = ? AND collection_name = ?
                LIMIT 1
                """,
                (filename, collection),
            ) as cursor:
                result = await cursor.fetchone()
                return result is not None

    async def check_documents_exist(
        self,
        filenames: list[str],
        collection_name: Optional[str] = None,
    ) -> dict[str, bool]:
        """Check which documents are already embedded.

        Args:
            filenames: List of source filenames to check
            collection_name: Collection name (defaults to instance collection)

        Returns:
            Dictionary mapping filename to existence status
        """
        collection = collection_name or self.collection_name
        result = {}

        async with aiosqlite.connect(self.db_path) as db:
            for filename in filenames:
                async with db.execute(
                    """
                    SELECT 1 FROM embedded_documents
                    WHERE source_filename = ? AND collection_name = ?
                    LIMIT 1
                    """,
                    (filename, collection),
                ) as cursor:
                    exists = await cursor.fetchone() is not None
                    result[filename] = exists

        return result

    async def mark_embedded(
        self,
        filename: str,
        file_hash: Optional[str] = None,
        document_count: Optional[int] = None,
        total_tokens: Optional[int] = None,
        embedding_cost_usd: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        fincode: Optional[int] = None,
        symbol: Optional[str] = None,
        auto_resolve: bool = True,
    ) -> None:
        """Mark a document as embedded.

        Args:
            filename: Source filename
            file_hash: SHA256 hash of file content
            document_count: Number of document chunks created
            total_tokens: Total tokens embedded
            embedding_cost_usd: Cost of embedding operation
            metadata: Additional metadata to store
            collection_name: Collection name (defaults to instance collection)
            fincode: Financial code (auto-resolved if not provided and auto_resolve=True)
            symbol: Stock symbol (auto-resolved if not provided and auto_resolve=True)
            auto_resolve: If True, automatically resolve fincode/symbol from filename (default: True)
        """
        collection = collection_name or self.collection_name
        metadata_json = json.dumps(metadata) if metadata else None

        # Auto-resolve fincode and symbol if not provided
        if auto_resolve and (fincode is None or symbol is None):
            resolved_fincode, resolved_symbol = self.resolve_fincode_and_symbol(filename)
            fincode = fincode or resolved_fincode
            symbol = symbol or resolved_symbol

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO embedded_documents
                (source_filename, file_hash, embedded_at, document_count,
                 total_tokens, embedding_cost_usd, collection_name, metadata_json,
                 fincode, symbol)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    filename,
                    file_hash,
                    datetime.now().isoformat(),
                    document_count,
                    total_tokens,
                    embedding_cost_usd,
                    collection,
                    metadata_json,
                    fincode,
                    symbol,
                ),
            )
            await db.commit()

        logger.debug(
            f"Marked '{filename}' as embedded in collection '{collection}' "
            f"(fincode={fincode}, symbol={symbol})"
        )

    async def remove_document(
        self,
        filename: str,
        collection_name: Optional[str] = None,
    ) -> bool:
        """Remove a document from tracking.

        Args:
            filename: Source filename to remove
            collection_name: Collection name (defaults to instance collection)

        Returns:
            True if document was removed, False if not found
        """
        collection = collection_name or self.collection_name

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                DELETE FROM embedded_documents
                WHERE source_filename = ? AND collection_name = ?
                """,
                (filename, collection),
            )
            await db.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Removed '{filename}' from tracking")
        else:
            logger.warning(f"'{filename}' not found in tracking")

        return deleted

    async def get_all_sources(
        self,
        collection_name: Optional[str] = None,
    ) -> set[str]:
        """Get all embedded source filenames.

        Args:
            collection_name: Collection name (defaults to instance collection)

        Returns:
            Set of source filenames
        """
        collection = collection_name or self.collection_name
        sources = set()

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT source_filename FROM embedded_documents
                WHERE collection_name = ?
                """,
                (collection,),
            ) as cursor:
                async for row in cursor:
                    sources.add(row[0])

        return sources

    async def get_document_info(
        self,
        filename: str,
        collection_name: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Get detailed information about an embedded document.

        Args:
            filename: Source filename
            collection_name: Collection name (defaults to instance collection)

        Returns:
            Dictionary with document info, or None if not found
        """
        collection = collection_name or self.collection_name

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT * FROM embedded_documents
                WHERE source_filename = ? AND collection_name = ?
                """,
                (filename, collection),
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None

                # Convert row to dictionary
                info = dict(row)

                # Parse metadata JSON if present
                if info.get("metadata_json"):
                    try:
                        info["metadata"] = json.loads(info["metadata_json"])
                    except json.JSONDecodeError:
                        info["metadata"] = None
                    del info["metadata_json"]

                return info

    async def needs_reembedding(
        self,
        filename: str,
        current_hash: str,
        collection_name: Optional[str] = None,
    ) -> bool:
        """Check if a document needs re-embedding based on hash.

        Args:
            filename: Source filename
            current_hash: Current SHA256 hash of file
            collection_name: Collection name (defaults to instance collection)

        Returns:
            True if document needs re-embedding (hash mismatch or not embedded)
        """
        collection = collection_name or self.collection_name

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT file_hash FROM embedded_documents
                WHERE source_filename = ? AND collection_name = ?
                """,
                (filename, collection),
            ) as cursor:
                row = await cursor.fetchone()

                # Not embedded yet
                if row is None:
                    return True

                stored_hash = row[0]

                # No hash stored (old entries) - consider needs re-embedding
                if stored_hash is None:
                    return True

                # Hash mismatch - file changed
                return stored_hash != current_hash

    async def get_stats(
        self,
        collection_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get statistics about embedded documents.

        Args:
            collection_name: Collection name (defaults to instance collection)

        Returns:
            Dictionary with statistics
        """
        collection = collection_name or self.collection_name

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT
                    COUNT(*) as total_documents,
                    SUM(document_count) as total_chunks,
                    SUM(total_tokens) as total_tokens,
                    SUM(embedding_cost_usd) as total_cost,
                    MIN(embedded_at) as first_embedded,
                    MAX(embedded_at) as last_embedded
                FROM embedded_documents
                WHERE collection_name = ?
                """,
                (collection,),
            ) as cursor:
                row = await cursor.fetchone()

                return {
                    "collection_name": collection,
                    "total_documents": row[0] or 0,
                    "total_chunks": row[1] or 0,
                    "total_tokens": row[2] or 0,
                    "total_cost_usd": row[3] or 0.0,
                    "first_embedded": row[4],
                    "last_embedded": row[5],
                }

    async def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all collections.

        Returns:
            Dictionary mapping collection name to stats
        """
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT DISTINCT collection_name FROM embedded_documents
                """
            ) as cursor:
                collections = [row[0] async for row in cursor]

        stats = {}
        for collection in collections:
            stats[collection] = await self.get_stats(collection)

        return stats

    async def clear_collection(
        self,
        collection_name: Optional[str] = None,
    ) -> int:
        """Clear all tracking data for a collection.

        Args:
            collection_name: Collection name (defaults to instance collection)

        Returns:
            Number of records deleted
        """
        collection = collection_name or self.collection_name

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                DELETE FROM embedded_documents
                WHERE collection_name = ?
                """,
                (collection,),
            )
            await db.commit()
            deleted = cursor.rowcount

        logger.info(f"Cleared {deleted} records for collection '{collection}'")
        return deleted

    # ── Parsed Document Cache Methods ──────────────────────────────────

    async def cache_parsed_documents(
        self,
        source_filename: str,
        documents: list[Document],
    ) -> None:
        """Cache parsed Document chunks for a source file.

        Replaces any existing cache for this source in a single transaction.

        Args:
            source_filename: The PDF filename (cache key)
            documents: List of parsed Document objects to cache
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Clear existing cache for this source
            await db.execute(
                "DELETE FROM parsed_documents_cache WHERE source_filename = ?",
                (source_filename,),
            )

            # Insert all chunks
            await db.executemany(
                """
                INSERT INTO parsed_documents_cache
                (source_filename, chunk_index, page_content, metadata_json)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        source_filename,
                        idx,
                        doc.page_content,
                        json.dumps(doc.metadata),
                    )
                    for idx, doc in enumerate(documents)
                ],
            )
            await db.commit()

        logger.info(
            f"Cached {len(documents)} chunks for '{source_filename}'"
        )

    async def get_cached_documents(
        self,
        source_filename: str,
    ) -> Optional[list[Document]]:
        """Retrieve cached parsed Documents for a source file.

        Args:
            source_filename: The PDF filename to look up

        Returns:
            List of Document objects if cached, None on cache miss
        """
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT page_content, metadata_json
                FROM parsed_documents_cache
                WHERE source_filename = ?
                ORDER BY chunk_index
                """,
                (source_filename,),
            ) as cursor:
                rows = await cursor.fetchall()

        if not rows:
            return None

        documents = []
        for page_content, metadata_json in rows:
            metadata = json.loads(metadata_json)
            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents

    async def is_cached(self, source_filename: str) -> bool:
        """Check if parsed documents are cached for a source file.

        Args:
            source_filename: The PDF filename to check

        Returns:
            True if cache exists, False otherwise
        """
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT 1 FROM parsed_documents_cache
                WHERE source_filename = ?
                LIMIT 1
                """,
                (source_filename,),
            ) as cursor:
                return await cursor.fetchone() is not None

    async def clear_parse_cache(
        self,
        source_filename: Optional[str] = None,
    ) -> int:
        """Clear the parsed documents cache.

        Args:
            source_filename: If provided, clear only this source's cache.
                             If None, clear all cached documents.

        Returns:
            Number of rows deleted
        """
        async with aiosqlite.connect(self.db_path) as db:
            if source_filename:
                cursor = await db.execute(
                    "DELETE FROM parsed_documents_cache WHERE source_filename = ?",
                    (source_filename,),
                )
            else:
                cursor = await db.execute("DELETE FROM parsed_documents_cache")
            await db.commit()
            deleted = cursor.rowcount

        logger.info(
            f"Cleared {deleted} rows from parse cache"
            + (f" for '{source_filename}'" if source_filename else " (all)")
        )
        return deleted

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the parsed documents cache.

        Returns:
            Dictionary with cache statistics
        """
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT
                    COUNT(DISTINCT source_filename) as cached_sources,
                    COUNT(*) as total_chunks,
                    MIN(parsed_at) as first_cached,
                    MAX(parsed_at) as last_cached
                FROM parsed_documents_cache
                """
            ) as cursor:
                row = await cursor.fetchone()

                return {
                    "cached_sources": row[0] or 0,
                    "total_chunks": row[1] or 0,
                    "first_cached": row[2],
                    "last_cached": row[3],
                }
