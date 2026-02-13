# Embedded Documents Database Documentation

## Overview
SQLite database file: `embedded_docs.db`
- **Total Records**: 46
- **Purpose**: Tracks embedded documents and their metadata for RAG (Retrieval-Augmented Generation) operations

---

## Database Schema

### Table: `embedded_documents`

This table stores metadata about documents that have been processed and embedded for vector search.

#### Columns

| Column Name | Data Type | Constraints | Description |
|------------|-----------|-------------|-------------|
| `source_filename` | TEXT | PRIMARY KEY | Unique identifier/filename of the source document (e.g., UUID-based PDF filename) |
| `file_hash` | TEXT | - | Hash value of the file content for integrity verification (currently NULL in sample data) |
| `embedded_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Timestamp when the document was embedded |
| `document_count` | INTEGER | - | Number of document chunks/splits created from the source file |
| `total_tokens` | INTEGER | - | Total number of tokens processed (currently NULL in sample data) |
| `embedding_cost_usd` | REAL | - | Cost in USD for embedding this document (currently NULL in sample data) |
| `collection_name` | TEXT | - | Name of the collection this document belongs to (e.g., "company_files") |
| `metadata_json` | TEXT | - | JSON string containing additional metadata about the document |
| `fincode` | INTEGER | - | Financial company code identifier (e.g., 100112, 100570) |
| `symbol` | TEXT | - | Stock ticker symbol for the company (e.g., SBIN, INFY, TMPV) |

---

## Indexes

The database includes four indexes for query optimization:

1. **`idx_collection`**
   - Column: `collection_name`
   - Purpose: Optimize queries filtering by collection

2. **`idx_embedded_at`**
   - Column: `embedded_at`
   - Purpose: Optimize queries sorting or filtering by embedding timestamp

3. **`idx_fincode`**
   - Column: `fincode`
   - Purpose: Optimize queries filtering by financial company code

4. **`idx_symbol`**
   - Column: `symbol`
   - Purpose: Optimize queries filtering by stock ticker symbol

---

## Sample Data

```
source_filename: a44221e1-a209-4710-ae79-59320332429a.pdf
file_hash: NULL
embedded_at: 2025-11-05T19:02:24.197712
document_count: 166
total_tokens: NULL
embedding_cost_usd: NULL
collection_name: company_files
metadata_json: {"migrated": true, "migration_date": 148422.265}
fincode: 100112
symbol: SBIN
```

---

## Metadata JSON Structure

The `metadata_json` field contains JSON objects with the following observed fields:

- `migrated` (boolean): Indicates if the record was migrated from a previous system
- `migration_date` (number): Timestamp or identifier for when migration occurred

---

## Usage Notes

1. **Primary Key**: The `source_filename` serves as the unique identifier for each document
2. **Nullable Fields**: Several fields (`file_hash`, `total_tokens`, `embedding_cost_usd`) are currently NULL, suggesting they may be optional or populated by specific operations
3. **Collection Organization**: Documents are organized into collections via the `collection_name` field
4. **Migration Status**: The metadata indicates this database may have been populated through a migration process
5. **Company Identification**: Documents are linked to financial companies via two fields:
   - `fincode`: Numeric company identifier
   - `symbol`: Stock ticker symbol
   - Both fields are indexed for efficient querying

---

## Common Queries

### Get all documents in a collection
```sql
SELECT * FROM embedded_documents
WHERE collection_name = 'company_files';
```

### Get documents embedded after a specific date
```sql
SELECT * FROM embedded_documents
WHERE embedded_at > '2025-11-05';
```

### Get total document count by collection
```sql
SELECT collection_name,
       COUNT(*) as total_docs,
       SUM(document_count) as total_chunks
FROM embedded_documents
GROUP BY collection_name;
```

### Find documents with the most chunks
```sql
SELECT source_filename, document_count
FROM embedded_documents
ORDER BY document_count DESC
LIMIT 10;
```

### Get all documents for a specific company by symbol
```sql
SELECT source_filename, document_count, embedded_at
FROM embedded_documents
WHERE symbol = 'SBIN';
```

### Get all documents for a specific company by fincode
```sql
SELECT source_filename, symbol, document_count
FROM embedded_documents
WHERE fincode = 100112;
```

### Get list of unique companies in the database
```sql
SELECT DISTINCT fincode, symbol
FROM embedded_documents
ORDER BY symbol;
```

### Get document count and chunk statistics by company
```sql
SELECT symbol, fincode,
       COUNT(*) as total_documents,
       SUM(document_count) as total_chunks,
       AVG(document_count) as avg_chunks_per_doc
FROM embedded_documents
GROUP BY symbol, fincode
ORDER BY total_documents DESC;
```

---

**Generated**: 2025-11-12
**Last Updated**: 2025-11-12
**Database Version**: SQLite
