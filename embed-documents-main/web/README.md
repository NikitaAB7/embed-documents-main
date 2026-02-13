# Embedded Documents Viewer

A web-based interface for viewing and downloading embedded company documents from the RAG system.

## Features

- **Dashboard Statistics**: View total documents, companies, and document chunks
- **Symbol Browser**: Browse all companies with embedded documents
- **Search & Filter**: Search by symbol or company name
- **File Viewer**: View all embedded files for each company
- **Download**: Download PDF documents directly from the API
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start

### 1. Install Dependencies

```bash
# Install the package with web dependencies
pip install -e .
```

### 2. Launch the Web Server

```bash
# From the project root directory
python run_web.py
```

### 3. Open in Browser

Navigate to: **http://localhost:8000**

## API Endpoints

The FastAPI backend provides the following REST API endpoints:

### Statistics
- `GET /api/stats` - Get overall database statistics

### Symbols
- `GET /api/symbols/summary` - Get all symbols with file counts and categories
- `GET /api/symbols/{symbol}/files` - Get all files for a specific symbol

### Files
- `GET /api/download/{filename}?category={category}` - Download a specific file

### Categories
- `GET /api/categories` - Get list of all unique collections/categories

## Project Structure

```
web/
├── app.py              # FastAPI backend application
├── static/
│   ├── index.html      # Main HTML page
│   ├── app.js          # Frontend JavaScript logic
│   └── style.css       # Styling
└── README.md           # This file
```

## How It Works

### Backend (FastAPI)
1. **Startup**: Initializes the database schema and loads stock/metadata mappings
2. **DocumentTracker**: Queries the SQLite database for embedded documents
3. **DocumentFetcher**: Fetches PDF files from the DefineEdge API
4. **API Endpoints**: Serves data to the frontend via REST API

### Frontend (HTML/JS/CSS)
1. **Dashboard**: Displays overall statistics fetched from `/api/stats`
2. **Table View**: Shows all symbols with file counts from `/api/symbols/summary`
3. **Search**: Client-side filtering of symbols
4. **Modal**: Opens when "View Files" is clicked, fetches files for that symbol
5. **Download**: Streams PDF file from backend to browser

### Data Flow

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Browser   │ ───► │ FastAPI      │ ───► │  SQLite DB  │
│  (Frontend) │ ◄─── │  (Backend)   │ ◄─── │ (Tracker)   │
└─────────────┘      └──────────────┘      └─────────────┘
                           │
                           ▼
                     ┌──────────────┐
                     │ DefineEdge   │
                     │     API      │
                     └──────────────┘
```

## Configuration

The application uses settings from:
- `rag/config.py` - RAG configuration (Qdrant collection name, etc.)
- `.env` - Environment variables for API access

## Development

### Run with Auto-Reload

The launch script (`run_web.py`) runs with auto-reload enabled, so changes to Python files will automatically restart the server.

### Modify Frontend

To modify the frontend:
1. Edit files in `web/static/`
2. Refresh your browser (no server restart needed)

### API Documentation

FastAPI provides automatic interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Troubleshooting

### Port Already in Use

If port 8000 is already in use, modify `run_web.py`:

```python
uvicorn.run(
    "web.app:app",
    host="0.0.0.0",
    port=8080,  # Change to any available port
    reload=True,
    log_level="info"
)
```

### Database Not Found

Ensure the database exists at: `rag/data/embedded_docs.db`

If not, run your embedding pipeline first to populate the database.

### API Connection Issues

Check that your `.env` file contains valid API credentials for DefineEdge API.

## Dependencies

- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI
- **aiosqlite**: Async SQLite database driver
- **DocumentTracker**: Custom module for database queries
- **DocumentFetcher**: Custom module for fetching PDFs

All dependencies are specified in `pyproject.toml`.
