"""FastAPI web application for viewing embedded documents."""

import json
import logging
from pathlib import Path
from typing import Optional

import aiosqlite
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from rag.ingestion.document_fetcher import DocumentFetcher
from rag.ingestion.document_tracker import DocumentTracker
from rag.retrieval.pipeline import RetrievalPipeline
from rag.retrieval.answer_synthesizer import AnswerSynthesizer
from rag.schemas.rag_schemas import QueryRequest, QueryResponse, RetrievedChunk
from utils.data_helpers import (
    get_metadata_item_by_attachment_name,
    initialize_metadata_data,
    initialize_stock_data,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base paths for static assets
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

# Initialize FastAPI app
app = FastAPI(
    title="Embedded Documents Viewer",
    description="View and download embedded company documents",
    version="1.0.0",
)

# Initialize document tracker and fetcher
tracker = DocumentTracker()
fetcher = DocumentFetcher()
retrieval_pipeline = RetrievalPipeline()
answer_synthesizer = AnswerSynthesizer()


@app.on_event("startup")
async def startup_event():
    """Initialize database and load data on startup."""
    logger.info("Starting up application...")

    # Initialize database schema
    await tracker.initialize()

    # Initialize stock and metadata mappings
    await initialize_stock_data()
    await initialize_metadata_data()

    logger.info("Application startup complete")


@app.get("/")
async def root():
    """Redirect to the static index page."""
    return FileResponse(INDEX_FILE)


@app.get("/api/stats")
async def get_stats():
    """Get overall database statistics.

    Returns:
        Dictionary with overall statistics across all collections.
    """
    try:
        all_stats = await tracker.get_all_stats()

        # Calculate totals across all collections
        total_documents = sum(stats["total_documents"] for stats in all_stats.values())
        total_chunks = sum(stats["total_chunks"] for stats in all_stats.values())

        return {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "collections": all_stats,
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/symbols/summary")
async def get_symbols_summary():
    """Get summary of all symbols with file counts and categories.

    Returns:
        List of dictionaries with symbol, fincode, file count, and categories.
    """
    try:
        

        results = []

        async with aiosqlite.connect(tracker.db_path) as db:
            # Query to get symbol-level aggregates
            query = """
                SELECT
                    symbol,
                    fincode,
                    COUNT(*) as file_count,
                    GROUP_CONCAT(DISTINCT collection_name) as collections,
                    SUM(document_count) as total_chunks
                FROM embedded_documents
                WHERE symbol IS NOT NULL
                GROUP BY symbol, fincode
                ORDER BY symbol
            """

            async with db.execute(query) as cursor:
                async for row in cursor:
                    symbol, fincode, file_count, collections, total_chunks = row

                    results.append({
                        "symbol": symbol,
                        "fincode": fincode,
                        "file_count": file_count,
                        "collections": collections.split(",") if collections else [],
                        "total_chunks": total_chunks or 0,
                    })

        logger.info(f"Returning summary for {len(results)} symbols")
        return results

    except Exception as e:
        logger.error(f"Error getting symbols summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/symbols/{symbol}/files")
async def get_files_by_symbol(symbol: str):
    """Get all embedded files for a specific symbol.

    Args:
        symbol: Stock ticker symbol (e.g., "SBIN", "TCS")

    Returns:
        List of file details including filename, collection, chunks, etc.
    """
    try:
        results = []

        async with aiosqlite.connect(tracker.db_path) as db:
            db.row_factory = aiosqlite.Row

            query = """
                SELECT
                    source_filename,
                    file_hash,
                    embedded_at,
                    document_count,
                    total_tokens,
                    embedding_cost_usd,
                    collection_name,
                    metadata_json,
                    fincode
                FROM embedded_documents
                WHERE symbol = ?
                ORDER BY embedded_at DESC
            """

            async with db.execute(query, (symbol.upper(),)) as cursor:
                async for row in cursor:
                    result = dict(row)

                    # Enrich with metadata (subcategory and date)
                    source_filename = result.get("source_filename")
                    if source_filename:
                        metadata_item = get_metadata_item_by_attachment_name(source_filename)
                        if metadata_item:
                            result["subcatname"] = metadata_item.get("subcatname")
                            result["newsDt"] = metadata_item.get("newsDt")
                        else:
                            result["subcatname"] = None
                            result["newsDt"] = None
                    else:
                        result["subcatname"] = None
                        result["newsDt"] = None

                    results.append(result)

        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"No files found for symbol: {symbol}"
            )

        logger.info(f"Returning {len(results)} files for symbol {symbol}")
        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting files for symbol {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/{filename}")
async def download_file(filename: str, category: Optional[str] = None):
    """Download a document file.

    Args:
        filename: Source filename to download
        category: Document category (optional, will be looked up from metadata)

    Returns:
        Streaming response with PDF content
    """
    try:
        # Get document info from tracker
        doc_info = await tracker.get_document_info(filename)

        if not doc_info:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {filename}"
            )

        # Try to determine category from metadata
        if not category:

            metadata_json = doc_info.get("metadata_json")
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                    category = metadata.get("category") or metadata.get("subcatname")
                except json.JSONDecodeError:
                    pass

        # If still no category, try to fetch from metadata helpers
        if not category:
            

            meta_item = get_metadata_item_by_attachment_name(filename)
            if meta_item:
                category = meta_item.get("subcatname")

        if not category:
            raise HTTPException(
                status_code=400,
                detail="Category not found. Please provide category parameter."
            )

        logger.info(f"Fetching document: {filename} (category: {category})")

        # Fetch document from API
        doc_stream = await fetcher.get_pdf_doc_stream(filename, category)

        # Return as streaming response
        return StreamingResponse(
            doc_stream.stream,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/categories")
async def get_categories():
    """Get list of all unique categories in the database.

    Returns:
        List of category names
    """
    try:
        categories = []

        async with aiosqlite.connect(tracker.db_path) as db:
            query = "SELECT DISTINCT collection_name FROM embedded_documents ORDER BY collection_name"

            async with db.execute(query) as cursor:
                async for row in cursor:
                    if row[0]:
                        categories.append(row[0])

        return categories

    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Run retrieval pipeline (router -> hybrid -> rerank -> RRF -> HYDE)."""
    try:
        filters = request.filters.model_dump() if request.filters else None
        retrieval_pipeline.use_hyde = request.use_hyde
        result = await retrieval_pipeline.retrieve(
            request.query, k=request.top_k, filters=filters
        )

        chunks: list[RetrievedChunk] = []
        for doc in result.documents:
            meta = doc.metadata or {}
            chunks.append(
                RetrievedChunk(
                    content=doc.page_content,
                    source=meta.get("source"),
                    chunk_id=meta.get("chunk_id"),
                    reference_chunk_id=meta.get("reference_chunk_id"),
                    metadata=meta,
                )
            )

        answer = None
        citations = None
        validation = None
        structured_answer = None
        faithfulness_score = None
        if request.synthesize:
            synthesis = await answer_synthesizer.synthesize(
                request.query,
                result.documents,
                strict_citations=request.strict_citations,
                structured_output=request.structured_output,
                evaluate_faithfulness=request.evaluate_faithfulness,
            )
            answer = synthesis.answer
            citations = synthesis.citations
            validation = synthesis.validation
            structured_answer = synthesis.structured_answer
            faithfulness_score = synthesis.faithfulness_score

        return QueryResponse(
            use_rag=result.route.use_rag,
            confidence=result.route.confidence,
            reason=result.route.reason,
            results=chunks,
            answer=answer,
            citations=citations,
            validation=validation,
            structured_answer=structured_answer,
            faithfulness_score=faithfulness_score,
        )

    except Exception as e:
        logger.error(f"Error in query pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files directory (must be last)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
