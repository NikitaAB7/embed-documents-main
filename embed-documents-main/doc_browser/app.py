"""FastAPI backend for Document Browser.

Provides two main endpoints:
1. GET /api/documents - List documents with descriptions and timestamps
2. GET /api/documents/{filename} - Get/download a specific document
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import aiosqlite
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import from parent modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.ingestion.document_fetcher import DocumentFetcher
from rag.ingestion.document_tracker import DocumentTracker
from utils.data_helpers import (
    get_metadata_item_by_attachment_name,
    get_symbol_map,
    initialize_metadata_data,
    initialize_stock_data,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

# Initialize FastAPI app
app = FastAPI(
    title="Document Browser",
    description="Browse and download embedded company documents",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize tracker and fetcher
tracker = DocumentTracker()
fetcher = DocumentFetcher()


# Response models
class DocumentListItem(BaseModel):
    """Document list item with description and timestamp."""
    filename: str
    symbol: Optional[str] = None
    fincode: Optional[int] = None
    category: Optional[str] = None
    description: Optional[str] = None
    document_date: Optional[str] = None
    embedded_at: Optional[str] = None
    chunk_count: Optional[int] = None


class DocumentListResponse(BaseModel):
    """Response for document list endpoint."""
    total: int
    documents: List[DocumentListItem]


class CompanyInfo(BaseModel):
    """Company information."""
    symbol: str
    fincode: Optional[int] = None
    file_count: int
    categories: List[str]


# Document categories available
DOCUMENT_CATEGORIES = [
    {"value": "annual-report", "label": "Annual Reports"},
    {"value": "concall", "label": "Concall Transcripts"},
    {"value": "investor-presentation", "label": "Investor Presentations"},
    {"value": "quarterly-results", "label": "Quarterly Results"},
    {"value": "corporate-announcement", "label": "Corporate Announcements"},
]


@app.on_event("startup")
async def startup_event():
    """Initialize database and load data on startup."""
    logger.info("Starting Document Browser application...")
    
    # Initialize database schema
    await tracker.initialize()
    
    # Initialize stock and metadata mappings
    await initialize_stock_data()
    await initialize_metadata_data()
    
    logger.info("Document Browser startup complete")


@app.get("/")
async def root():
    """Serve the main index page."""
    return FileResponse(INDEX_FILE)


@app.get("/api/companies", response_model=List[CompanyInfo])
async def get_companies():
    """Get list of all companies with embedded documents.
    
    Returns a list of companies with their symbols, fincodes, and document counts.
    """
    try:
        results = []
        
        async with aiosqlite.connect(tracker.db_path) as db:
            query = """
                SELECT 
                    symbol,
                    fincode,
                    COUNT(*) as file_count,
                    GROUP_CONCAT(DISTINCT collection_name) as collections
                FROM embedded_documents
                WHERE symbol IS NOT NULL
                GROUP BY symbol, fincode
                ORDER BY symbol
            """
            
            async with db.execute(query) as cursor:
                async for row in cursor:
                    symbol, fincode, file_count, collections = row
                    
                    results.append(CompanyInfo(
                        symbol=symbol,
                        fincode=fincode,
                        file_count=file_count,
                        categories=collections.split(",") if collections else [],
                    ))
        
        logger.info(f"Returning {len(results)} companies")
        return results
        
    except Exception as e:
        logger.error(f"Error getting companies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/categories")
async def get_categories():
    """Get available document categories."""
    return DOCUMENT_CATEGORIES


@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents(
    symbol: Optional[str] = Query(None, description="Filter by stock symbol (e.g., TCS, INFY)"),
    category: Optional[str] = Query(None, description="Filter by document category"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of documents to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """Get list of documents with descriptions and timestamps.
    
    This is the first API call - returns a list of documents matching the filters.
    Use the filename from this response to fetch the actual document.
    
    Args:
        symbol: Stock ticker symbol (e.g., "TCS", "INFY")
        category: Document category (e.g., "annual-report", "concall")
        limit: Maximum results to return
        offset: Pagination offset
        
    Returns:
        List of documents with descriptions and timestamps
    """
    try:
        documents = []
        
        async with aiosqlite.connect(tracker.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Build dynamic query based on filters
            where_clauses = []
            params = []
            
            if symbol:
                where_clauses.append("symbol = ?")
                params.append(symbol.upper())
            
            if category:
                where_clauses.append("collection_name LIKE ?")
                params.append(f"%{category}%")
            
            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            query = f"""
                SELECT 
                    source_filename,
                    symbol,
                    fincode,
                    collection_name,
                    embedded_at,
                    document_count,
                    metadata_json
                FROM embedded_documents
                WHERE {where_sql}
                ORDER BY embedded_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
            
            async with db.execute(query, params) as cursor:
                async for row in cursor:
                    row_dict = dict(row)
                    
                    # Get additional metadata (subcategory and date)
                    source_filename = row_dict.get("source_filename")
                    metadata_item = get_metadata_item_by_attachment_name(source_filename) if source_filename else None
                    
                    # Build description from metadata
                    subcatname = metadata_item.get("subcatname") if metadata_item else None
                    news_date = metadata_item.get("newsDt") if metadata_item else None
                    
                    description = f"{subcatname or 'Document'}"
                    if news_date:
                        description += f" - {news_date[:10]}"
                    
                    documents.append(DocumentListItem(
                        filename=source_filename,
                        symbol=row_dict.get("symbol"),
                        fincode=row_dict.get("fincode"),
                        category=subcatname or row_dict.get("collection_name"),
                        description=description,
                        document_date=news_date[:10] if news_date else None,
                        embedded_at=row_dict.get("embedded_at"),
                        chunk_count=row_dict.get("document_count"),
                    ))
            
            # Get total count
            count_query = f"""
                SELECT COUNT(*) FROM embedded_documents WHERE {where_sql}
            """
            async with db.execute(count_query, params[:-2]) as cursor:  # Remove limit/offset params
                total = (await cursor.fetchone())[0]
        
        logger.info(f"Returning {len(documents)} documents (total: {total})")
        return DocumentListResponse(total=total, documents=documents)
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{filename:path}")
async def get_document(filename: str, download: bool = Query(False, description="Force download")):
    """Get/download a specific document.
    
    This is the second API call - use the filename from the list endpoint.
    
    Args:
        filename: Document filename to retrieve
        download: If True, force download instead of inline display
        
    Returns:
        The document file (PDF, etc.)
    """
    try:
        # Query database directly (across all collections)
        doc_info = None
        async with aiosqlite.connect(tracker.db_path) as db:
            db.row_factory = aiosqlite.Row
            query = """
                SELECT * FROM embedded_documents
                WHERE source_filename = ?
                LIMIT 1
            """
            async with db.execute(query, (filename,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    doc_info = dict(row)
        
        if not doc_info:
            raise HTTPException(status_code=404, detail=f"Document not found: {filename}")
        
        # Try to find local source_path from metadata_json
        source_path = None
        category = None
        
        metadata_json = doc_info.get("metadata_json")
        if metadata_json:
            try:
                parsed = json.loads(metadata_json)
                source_path = parsed.get("source_path")
                category = parsed.get("category")  # Get category from stored metadata
            except json.JSONDecodeError:
                pass
        
        # If we have a local source_path, serve the file
        if source_path:
            local_path = Path(source_path)
            if not local_path.is_absolute():
                project_dir = Path(__file__).resolve().parent.parent
                local_path = project_dir / source_path
            
            if local_path.exists():
                logger.info(f"Serving local file: {local_path}")
                
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
                
                disposition = "attachment" if download else "inline"
                
                return FileResponse(
                    path=local_path,
                    media_type=media_type,
                    filename=filename,
                    headers={"Content-Disposition": f'{disposition}; filename="{filename}"'}
                )
        
        # Try to get category from metadata helpers if not in stored metadata
        if not category:
            metadata_item = get_metadata_item_by_attachment_name(filename)
            if metadata_item:
                category = metadata_item.get("subcatname")
        
        if not category:
            raise HTTPException(
                status_code=400,
                detail="Cannot determine document category for remote fetch"
            )
        
        logger.info(f"Fetching document from remote: {filename} (category: {category})")
        
        # Fetch from remote API
        doc_stream = await fetcher.get_pdf_doc_stream(filename, category)
        
        if doc_stream is None:
            raise HTTPException(status_code=404, detail=f"Could not fetch document: {filename}")
        
        # Return streaming response
        disposition = "attachment" if download else "inline"
        
        async def stream_content():
            yield doc_stream.stream.read()
        
        return StreamingResponse(
            stream_content(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'{disposition}; filename="{filename}"',
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files last to avoid catching API routes
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
