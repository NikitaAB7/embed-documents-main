# Document Browser

A simple web UI for browsing and downloading embedded company documents.

## Features

- **Company Selection**: Choose from available companies with embedded documents
- **Document Type Filtering**: Filter by document categories (Annual Reports, Concall Transcripts, etc.)
- **Document List**: View documents with descriptions, timestamps, and metadata
- **Document Viewer**: Preview documents directly in the browser
- **Download**: Download documents locally

## API Endpoints

### 1. Get List of Companies
```
GET /api/companies
```
Returns list of companies with embedded documents.

**Response:**
```json
[
  {
    "symbol": "TCS",
    "fincode": 100034,
    "file_count": 15,
    "categories": ["annual-report", "concall"]
  }
]
```

### 2. Get Document Categories
```
GET /api/categories
```
Returns available document categories.

### 3. List Documents
```
GET /api/documents?symbol=TCS&category=annual-report&limit=50&offset=0
```
Returns list of documents matching the filters.

**Parameters:**
- `symbol` (optional): Stock ticker symbol (e.g., "TCS", "INFY")
- `category` (optional): Document category
- `limit` (default: 50): Maximum results
- `offset` (default: 0): Pagination offset

**Response:**
```json
{
  "total": 100,
  "documents": [
    {
      "filename": "TCS_Annual_Report_2024.pdf",
      "symbol": "TCS",
      "fincode": 100034,
      "category": "Annual Reports",
      "description": "Annual Reports - 2024-03-15",
      "document_date": "2024-03-15",
      "embedded_at": "2024-03-20T10:30:00",
      "chunk_count": 45
    }
  ]
}
```

### 4. Get/Download Document
```
GET /api/documents/{filename}?download=false
```
Returns the actual document file.

**Parameters:**
- `filename`: Document filename (URL encoded)
- `download` (default: false): Force download vs inline display

## Usage Flow

1. **First API Call** - Get document list:
   ```javascript
   const response = await fetch('/api/documents?symbol=TCS');
   const data = await response.json();
   // data.documents contains list of documents
   ```

2. **Second API Call** - Get specific document:
   ```javascript
   const filename = data.documents[0].filename;
   window.open(`/api/documents/${encodeURIComponent(filename)}`);
   ```

## Running the Application

```bash
# From the project root
python run_doc_browser.py

# Or directly with uvicorn
uvicorn doc_browser.app:app --host 0.0.0.0 --port 8001 --reload
```

The application will be available at: http://localhost:8001

## Project Structure

```
doc_browser/
├── __init__.py      # Package init
├── app.py           # FastAPI backend
├── static/
│   ├── index.html   # Main HTML page
│   ├── style.css    # Styles
│   └── app.js       # Frontend JavaScript
└── README.md        # This file
```

## Dependencies

The Document Browser uses the same dependencies as the main project:
- FastAPI
- aiosqlite
- uvicorn
- pydantic

No additional dependencies required.
