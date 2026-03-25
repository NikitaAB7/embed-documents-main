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
from rag.observability.local_evaluators import LocalEvaluator
from rag.schemas.rag_schemas import (
    ExtractedFiltersResponse,
    EvaluationScores,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)
from utils.data_helpers import (
    get_metadata_item_by_attachment_name,
    get_symbol_map,
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

# Initialize local evaluator
local_evaluator = LocalEvaluator()
retrieval_pipeline = RetrievalPipeline(use_dynamic_filters=True)
answer_synthesizer = AnswerSynthesizer()


class EvaluationCache:
    """Load eval_results.json and match questions to evaluations."""

    def __init__(self):
        """Initialize evaluation cache."""
        self.evals = {}
        self.load_evaluations()

    def load_evaluations(self):
        """Load evaluation results from eval_results.json."""
        eval_file = Path(__file__).resolve().parent.parent / "eval_results.json"
        if eval_file.exists():
            try:
                with open(eval_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.evals = data.get("results", {})
                    logger.info(f"Loaded evaluations for {len(self.evals)} metrics")
            except Exception as e:
                logger.warning(f"Failed to load evaluation results: {e}")
        else:
            logger.info("No eval_results.json found, evaluations will not be available")

    def get_eval_for_question(self, question_idx: int) -> dict:
        """Get evaluation for a specific question by index.

        Args:
            question_idx: Index of the question (0-based)

        Returns:
            Dictionary with evaluation scores and reasoning
        """
        result = {}
        for metric, scores in self.evals.items():
            if isinstance(scores, list) and question_idx < len(scores):
                result[metric] = scores[question_idx]
        return result

    def get_summary(self) -> dict:
        """Get overall evaluation summary.

        Returns:
            Summary statistics from eval_results.json
        """
        eval_file = Path(__file__).resolve().parent.parent / "eval_results.json"
        if eval_file.exists():
            try:
                with open(eval_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data
            except Exception as e:
                logger.warning(f"Failed to read evaluation summary: {e}")
        return {}


eval_cache = EvaluationCache()


@app.on_event("startup")
async def startup_event():
    """Initialize database and load data on startup."""
    logger.info("Starting up application...")

    # Initialize database schema
    await tracker.initialize()

    # Initialize stock and metadata mappings
    await initialize_stock_data()
    await initialize_metadata_data()

    # Update retrieval pipeline with symbol lookup for dynamic filters
    try:
        symbol_map = get_symbol_map()
        if symbol_map:
            retrieval_pipeline.symbol_lookup = symbol_map
            if retrieval_pipeline.dynamic_router:
                retrieval_pipeline.dynamic_router.symbol_lookup = symbol_map
            logger.info(f"Loaded {len(symbol_map)} symbols for dynamic filter routing")
    except RuntimeError:
        logger.warning("Symbol map not available, dynamic filter routing will work without fincode resolution")

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
        Streaming response with PDF content, or FileResponse for local files
    """
    try:
        # Get document info from tracker
        doc_info = await tracker.get_document_info(filename)

        # Try to find source_path (local file location)
        source_path = None
        
        # First check tracker metadata
        if doc_info:
            metadata = doc_info.get("metadata")
            if metadata and metadata.get("source_path"):
                source_path = metadata["source_path"]
        
        # If not in tracker, query Qdrant directly for the source_path
        if not source_path:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                import os
                qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
                collection_name = os.getenv("QDRANT_COLLECTION", "compliance_docs")
                client = QdrantClient(url=qdrant_url)
                # Search for one document matching this source filename (nested in metadata)
                results = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="metadata.source", match=MatchValue(value=filename))]
                    ),
                    limit=1,
                    with_payload=True,
                )
                logger.info(f"Qdrant scroll results for '{filename}': {len(results[0]) if results and results[0] else 0} points")
                if results and results[0]:
                    point = results[0][0]
                    # source_path is nested in metadata
                    meta = point.payload.get("metadata", {})
                    source_path = meta.get("source_path")
                    logger.info(f"Found source_path from Qdrant: {source_path}")
            except Exception as e:
                logger.warning(f"Could not query Qdrant for source_path: {e}")

        # Serve local file if source_path exists
        if source_path:
            local_path = Path(source_path)
            # Try as absolute path first, then relative to project directory
            if not local_path.is_absolute():
                # Try relative to project directory (parent of web folder)
                project_dir = Path(__file__).resolve().parent.parent
                local_path = project_dir / source_path
            if local_path.exists():
                logger.info(f"Serving local file: {local_path}")
                # Determine media type based on extension
                ext = local_path.suffix.lower()
                media_types = {
                    ".pdf": "application/pdf",
                    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ".doc": "application/msword",
                    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    ".xls": "application/vnd.ms-excel",
                    ".txt": "text/plain",
                    ".md": "text/markdown",
                }
                media_type = media_types.get(ext, "application/octet-stream")
                return FileResponse(
                    path=local_path,
                    media_type=media_type,
                    filename=filename,
                )
            else:
                logger.warning(f"Local file not found: {local_path}")

        if not doc_info:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {filename}"
            )

        # Try to determine category from metadata for remote fetch
        if not category:
            metadata_json = doc_info.get("metadata_json")
            if metadata_json:
                try:
                    parsed_meta = json.loads(metadata_json)
                    category = parsed_meta.get("category") or parsed_meta.get("subcatname")
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


@app.get("/api/evaluations")
async def get_evaluations():
    """Get full evaluation results from eval_results.json.

    Returns:
        Complete evaluation data with all metrics and results
    """
    try:
        summary = eval_cache.get_summary()
        if not summary:
            raise HTTPException(
                status_code=404,
                detail="No evaluation results available. Run evaluations first.",
            )
        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting evaluations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/evaluations/summary")
async def get_evaluations_summary():
    """Get summary statistics of all evaluations.

    Returns:
        Dictionary with mean, min, max, and pass_rate for each metric
    """
    try:
        full_data = eval_cache.get_summary()
        if not full_data:
            raise HTTPException(
                status_code=404,
                detail="No evaluation results available.",
            )
        return {
            "dataset_name": full_data.get("dataset_name"),
            "num_examples": full_data.get("num_examples"),
            "summary": full_data.get("summary", {}),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting evaluations summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/evaluations/question/{question_idx}")
async def get_question_evaluation(question_idx: int):
    """Get evaluation scores for a specific question.

    Args:
        question_idx: Zero-based index of the question

    Returns:
        Evaluation scores and reasoning for that question
    """
    try:
        evals = eval_cache.get_eval_for_question(question_idx)
        if not evals:
            raise HTTPException(
                status_code=404,
                detail=f"No evaluation found for question index {question_idx}",
            )
        return evals

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting question evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Run retrieval pipeline with optional dynamic filter extraction.

    When use_dynamic_filters=True, the LLM analyzes the query to extract
    metadata filters (symbol, category, date range) automatically.

    Optionally includes evaluation scores if question_idx is provided.
    """
    try:
        filters = request.filters.model_dump() if request.filters else None
        retrieval_pipeline.use_hyde = request.use_hyde

        # Enable dynamic filter extraction if requested
        result = await retrieval_pipeline.retrieve(
            request.query,
            k=request.top_k,
            filters=filters,
            use_dynamic_filters=request.use_dynamic_filters,
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

        # Build extracted filters response if available
        extracted_filters_response = None
        if result.extracted_filters:
            ef = result.extracted_filters
            extracted_filters_response = ExtractedFiltersResponse(
                fincode=ef.fincode,
                symbol=ef.symbol,
                company_name=ef.company_name,
                category=ef.category,
                date_from=ef.date_from,
                date_to=ef.date_to,
                confidence=ef.confidence,
                reasoning=ef.reasoning,
            )

        # Include evaluation scores if answer is available
        evaluation_scores = None
        if answer and answer.strip():
            try:
                contexts = [doc.page_content for doc in result.documents[:5]]
                eval_results = await local_evaluator.evaluate_all(
                    question=request.query,
                    answer=answer,
                    contexts=contexts,
                )
                evaluation_scores = EvaluationScores(
                    correctness=eval_results.get("correctness"),
                    relevancy=eval_results.get("relevancy"),
                    logical_coherence=eval_results.get("logical_coherence"),
                    groundedness=eval_results.get("groundedness"),
                    reasoning=eval_results.get("reasoning"),
                )
                logger.info(f"Generated evaluation scores: correctness={evaluation_scores.correctness:.2f}, relevancy={evaluation_scores.relevancy:.2f}, numerical_accuracy={evaluation_scores.numerical_accuracy:.2f}")
            except Exception as e:
                logger.warning(f"Failed to evaluate answer: {e}")

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
            extracted_filters=extracted_filters_response,
            evaluation_scores=evaluation_scores,
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
