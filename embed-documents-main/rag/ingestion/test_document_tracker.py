"""Tests for DocumentTracker SQLite-based tracking system.

These tests demonstrate the key features of the document tracking system
and verify that it works correctly.

Run with: pytest rag/ingestion/test_document_tracker.py -v
"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from rag.ingestion.document_tracker import DocumentTracker


@pytest.fixture
async def temp_tracker():
    """Create a temporary DocumentTracker for testing."""
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        tracker = DocumentTracker(db_path=db_path, collection_name="test_collection")
        await tracker.initialize()
        yield tracker


@pytest.mark.asyncio
async def test_initialize_creates_database(temp_tracker):
    """Test that initialization creates the database and tables."""
    # Database should exist after initialization
    assert temp_tracker.db_path.exists()

    # Should be able to query without errors
    stats = await temp_tracker.get_stats()
    assert stats["total_documents"] == 0


@pytest.mark.asyncio
async def test_mark_and_check_embedded(temp_tracker):
    """Test marking documents as embedded and checking existence."""
    # Initially not embedded
    assert not await temp_tracker.is_embedded("doc1.pdf")

    # Mark as embedded
    await temp_tracker.mark_embedded(
        filename="doc1.pdf",
        file_hash="abc123",
        document_count=10,
        total_tokens=5000,
        embedding_cost_usd=0.10,
    )

    # Now should be embedded
    assert await temp_tracker.is_embedded("doc1.pdf")

    # Check multiple documents at once
    results = await temp_tracker.check_documents_exist(
        ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    )
    assert results == {
        "doc1.pdf": True,
        "doc2.pdf": False,
        "doc3.pdf": False,
    }


@pytest.mark.asyncio
async def test_get_all_sources(temp_tracker):
    """Test getting all embedded sources."""
    # Start with empty set
    sources = await temp_tracker.get_all_sources()
    assert len(sources) == 0

    # Add some documents
    for i in range(5):
        await temp_tracker.mark_embedded(
            filename=f"doc{i}.pdf",
            document_count=10,
        )

    # Should have all 5 sources
    sources = await temp_tracker.get_all_sources()
    assert len(sources) == 5
    assert "doc0.pdf" in sources
    assert "doc4.pdf" in sources


@pytest.mark.asyncio
async def test_remove_document(temp_tracker):
    """Test removing documents from tracking."""
    # Add a document
    await temp_tracker.mark_embedded("doc1.pdf", document_count=10)
    assert await temp_tracker.is_embedded("doc1.pdf")

    # Remove it
    removed = await temp_tracker.remove_document("doc1.pdf")
    assert removed is True

    # Should no longer be embedded
    assert not await temp_tracker.is_embedded("doc1.pdf")

    # Removing again should return False
    removed = await temp_tracker.remove_document("doc1.pdf")
    assert removed is False


@pytest.mark.asyncio
async def test_get_document_info(temp_tracker):
    """Test getting detailed document information."""
    # Mark a document with metadata
    await temp_tracker.mark_embedded(
        filename="doc1.pdf",
        file_hash="abc123",
        document_count=15,
        total_tokens=7500,
        embedding_cost_usd=0.15,
        metadata={"custom": "value", "tags": ["important"]},
    )

    # Get info
    info = await temp_tracker.get_document_info("doc1.pdf")

    assert info is not None
    assert info["source_filename"] == "doc1.pdf"
    assert info["file_hash"] == "abc123"
    assert info["document_count"] == 15
    assert info["total_tokens"] == 7500
    assert info["embedding_cost_usd"] == 0.15
    assert info["metadata"]["custom"] == "value"
    assert "important" in info["metadata"]["tags"]

    # Non-existent document should return None
    info = await temp_tracker.get_document_info("nonexistent.pdf")
    assert info is None


@pytest.mark.asyncio
async def test_needs_reembedding(temp_tracker):
    """Test checking if document needs re-embedding based on hash."""
    # Document not embedded - needs embedding
    assert await temp_tracker.needs_reembedding("doc1.pdf", "hash123") is True

    # Mark as embedded with hash
    await temp_tracker.mark_embedded(
        filename="doc1.pdf",
        file_hash="hash123",
        document_count=10,
    )

    # Same hash - doesn't need re-embedding
    assert await temp_tracker.needs_reembedding("doc1.pdf", "hash123") is False

    # Different hash - needs re-embedding (file changed)
    assert await temp_tracker.needs_reembedding("doc1.pdf", "hash456") is True


@pytest.mark.asyncio
async def test_get_stats(temp_tracker):
    """Test getting statistics about embedded documents."""
    # Add some documents with various costs
    await temp_tracker.mark_embedded(
        "doc1.pdf", document_count=10, total_tokens=5000, embedding_cost_usd=0.10
    )
    await temp_tracker.mark_embedded(
        "doc2.pdf", document_count=20, total_tokens=10000, embedding_cost_usd=0.20
    )
    await temp_tracker.mark_embedded(
        "doc3.pdf", document_count=15, total_tokens=7500, embedding_cost_usd=0.15
    )

    # Get stats
    stats = await temp_tracker.get_stats()

    assert stats["collection_name"] == "test_collection"
    assert stats["total_documents"] == 3
    assert stats["total_chunks"] == 45  # 10 + 20 + 15
    assert stats["total_tokens"] == 22500  # 5000 + 10000 + 7500
    assert stats["total_cost_usd"] == pytest.approx(0.45)  # 0.10 + 0.20 + 0.15


@pytest.mark.asyncio
async def test_multiple_collections(temp_tracker):
    """Test that different collections are isolated."""
    # Add to default collection
    await temp_tracker.mark_embedded(
        "doc1.pdf", document_count=10, collection_name="collection_a"
    )

    # Add to different collection
    await temp_tracker.mark_embedded(
        "doc1.pdf", document_count=20, collection_name="collection_b"
    )

    # Each collection should have its own data
    assert await temp_tracker.is_embedded("doc1.pdf", collection_name="collection_a")
    assert await temp_tracker.is_embedded("doc1.pdf", collection_name="collection_b")

    # Stats should be per-collection
    stats_a = await temp_tracker.get_stats(collection_name="collection_a")
    stats_b = await temp_tracker.get_stats(collection_name="collection_b")

    assert stats_a["total_documents"] == 1
    assert stats_b["total_documents"] == 1


@pytest.mark.asyncio
async def test_clear_collection(temp_tracker):
    """Test clearing all data for a collection."""
    # Add some documents
    for i in range(5):
        await temp_tracker.mark_embedded(f"doc{i}.pdf", document_count=10)

    assert len(await temp_tracker.get_all_sources()) == 5

    # Clear collection
    deleted = await temp_tracker.clear_collection()
    assert deleted == 5

    # Should be empty now
    assert len(await temp_tracker.get_all_sources()) == 0


@pytest.mark.asyncio
async def test_update_existing_document(temp_tracker):
    """Test that marking an already-embedded document updates it."""
    # Initial mark
    await temp_tracker.mark_embedded(
        "doc1.pdf",
        file_hash="hash1",
        document_count=10,
        total_tokens=5000,
    )

    info = await temp_tracker.get_document_info("doc1.pdf")
    assert info["file_hash"] == "hash1"
    assert info["document_count"] == 10

    # Update with new hash and count
    await temp_tracker.mark_embedded(
        "doc1.pdf",
        file_hash="hash2",
        document_count=15,
        total_tokens=7500,
    )

    # Should be updated
    info = await temp_tracker.get_document_info("doc1.pdf")
    assert info["file_hash"] == "hash2"
    assert info["document_count"] == 15
    assert info["total_tokens"] == 7500

    # Should still only be 1 document
    assert len(await temp_tracker.get_all_sources()) == 1


@pytest.mark.asyncio
async def test_performance_bulk_checks():
    """Test that bulk existence checks are fast (demonstrate O(log n) lookup)."""
    import time

    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "perf_test.db"
        tracker = DocumentTracker(db_path=db_path)
        await tracker.initialize()

        # Add 1000 documents
        for i in range(1000):
            await tracker.mark_embedded(f"doc{i}.pdf", document_count=1)

        # Check 100 documents (mix of existing and non-existing)
        filenames = [f"doc{i}.pdf" for i in range(50)] + [
            f"new{i}.pdf" for i in range(50)
        ]

        start = time.time()
        results = await tracker.check_documents_exist(filenames)
        elapsed = time.time() - start

        # Should be very fast (< 100ms for 100 checks)
        assert elapsed < 0.1

        # Verify correctness
        assert sum(results.values()) == 50  # 50 existing, 50 new


if __name__ == "__main__":
    # Run tests with asyncio
    pytest.main([__file__, "-v"])
