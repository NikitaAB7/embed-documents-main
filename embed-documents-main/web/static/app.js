// Global state
let allSymbols = [];
let currentSymbol = null;

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    // Search input
    const searchInput = document.getElementById('search-input');
    searchInput.addEventListener('input', handleSearch);

    // Query input - submit on Ctrl+Enter
    const queryInput = document.getElementById('query-input');
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            submitQuery();
        }
    });

    // Refresh button
    const refreshBtn = document.getElementById('refresh-btn');
    refreshBtn.addEventListener('click', () => {
        loadSymbolsData();
    });

    // Close modal on outside click
    const modal = document.getElementById('file-modal');
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeModal();
        }
    });

    // Close modal on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeModal();
        }
    });
}

// Submit query to RAG pipeline
async function submitQuery() {
    const queryInput = document.getElementById('query-input');
    const query = queryInput.value.trim();

    if (!query) {
        alert('Please enter a question');
        return;
    }

    const useHyde = document.getElementById('use-hyde').checked;
    const synthesize = document.getElementById('synthesize').checked;
    const topK = parseInt(document.getElementById('top-k').value);

    // Show loading state
    const resultsDiv = document.getElementById('query-results');
    const loadingDiv = document.getElementById('query-loading');
    const answerDiv = document.getElementById('query-answer');
    const chunksDiv = document.getElementById('query-chunks');
    const queryBtn = document.getElementById('query-btn');

    resultsDiv.style.display = 'block';
    loadingDiv.style.display = 'flex';
    answerDiv.style.display = 'none';
    chunksDiv.style.display = 'none';
    queryBtn.disabled = true;
    queryBtn.textContent = 'Searching...';

    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                top_k: topK,
                use_hyde: useHyde,
                synthesize: synthesize,
                strict_citations: true,
                structured_output: false,
                evaluate_faithfulness: false,
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Query failed');
        }

        const data = await response.json();
        displayQueryResults(data);

    } catch (error) {
        console.error('Query error:', error);
        loadingDiv.innerHTML = `
            <div class="error-message">
                <p>Query failed: ${error.message}</p>
            </div>
        `;
    } finally {
        queryBtn.disabled = false;
        queryBtn.textContent = 'Search Documents';
    }
}

// Display query results
function displayQueryResults(data) {
    const loadingDiv = document.getElementById('query-loading');
    const answerDiv = document.getElementById('query-answer');
    const chunksDiv = document.getElementById('query-chunks');

    loadingDiv.style.display = 'none';

    // Check if RAG was used
    if (!data.use_rag) {
        chunksDiv.innerHTML = `
            <div class="no-results">
                <p>This query doesn't require document search.</p>
                <p>Confidence: ${(data.confidence * 100).toFixed(1)}% | Reason: ${data.reason}</p>
            </div>
        `;
        chunksDiv.style.display = 'block';
        return;
    }

    // Display answer if available
    if (data.answer) {
        const answerContent = document.getElementById('answer-content');
        const citationsDiv = document.getElementById('citations');

        answerContent.textContent = data.answer;

        // Display citations
        if (data.citations && data.citations.length > 0) {
            citationsDiv.innerHTML = `
                <strong>Sources:</strong>
                ${data.citations.map(c => `
                    <span class="citation-item" title="${c.source || 'Unknown source'}">
                        ${c.chunk_id || c.source || 'Source'}
                    </span>
                `).join('')}
            `;
        } else {
            citationsDiv.innerHTML = '';
        }

        answerDiv.style.display = 'block';
    }

    // Display retrieved chunks
    if (data.results && data.results.length > 0) {
        const chunksList = document.getElementById('chunks-list');

        chunksList.innerHTML = data.results.map((chunk, index) => {
            const source = chunk.source || 'Unknown';
            const category = chunk.metadata?.category || '';
            const ticker = chunk.metadata?.ticker || '';

            return `
                <div class="chunk-item">
                    <div class="chunk-header">
                        <span class="chunk-source">${truncateFilename(source, 60)}</span>
                        <div class="chunk-meta">
                            ${ticker ? `<span class="badge">${ticker}</span>` : ''}
                            ${category ? `<span class="badge">${category}</span>` : ''}
                            <span class="badge">#${index + 1}</span>
                        </div>
                    </div>
                    <div class="chunk-content">${escapeHtml(chunk.content)}</div>
                </div>
            `;
        }).join('');

        chunksDiv.querySelector('h3').textContent = `Retrieved Chunks (${data.results.length})`;
        chunksDiv.style.display = 'block';
    } else {
        const chunksList = document.getElementById('chunks-list');
        chunksList.innerHTML = `
            <div class="no-results">
                <p>No relevant chunks found for your query.</p>
            </div>
        `;
        chunksDiv.style.display = 'block';
    }
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize the application
async function initializeApp() {
    try {
        await Promise.all([
            loadStats(),
            loadSymbolsData()
        ]);
    } catch (error) {
        showError('Failed to initialize application: ' + error.message);
    }
}

// Load statistics
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        if (!response.ok) throw new Error('Failed to load statistics');

        const data = await response.json();

        document.getElementById('total-documents').textContent =
            data.total_documents.toLocaleString();
        document.getElementById('total-chunks').textContent =
            data.total_chunks.toLocaleString();

    } catch (error) {
        console.error('Error loading stats:', error);
        // Don't show error for stats, it's not critical
    }
}

// Load symbols data
async function loadSymbolsData() {
    const loading = document.getElementById('loading');
    const tableBody = document.getElementById('table-body');

    try {
        loading.style.display = 'flex';
        hideError();

        const response = await fetch('/api/symbols/summary');
        if (!response.ok) throw new Error('Failed to load symbols data');

        allSymbols = await response.json();

        // Update total symbols stat
        document.getElementById('total-symbols').textContent =
            allSymbols.length.toLocaleString();

        // Render table
        renderSymbolsTable(allSymbols);

    } catch (error) {
        showError('Failed to load data: ' + error.message);
        tableBody.innerHTML = '<tr><td colspan="6" class="no-data">Failed to load data</td></tr>';
    } finally {
        loading.style.display = 'none';
    }
}

// Render symbols table
function renderSymbolsTable(symbols) {
    const tableBody = document.getElementById('table-body');

    if (symbols.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="6" class="no-data">No data available</td></tr>';
        return;
    }

    tableBody.innerHTML = symbols.map(symbol => `
        <tr data-symbol="${symbol.symbol}">
            <td class="symbol-cell">
                <strong>${symbol.symbol}</strong>
            </td>
            <td>${symbol.fincode}</td>
            <td>${symbol.file_count}</td>
            <td>${symbol.total_chunks.toLocaleString()}</td>
            <td>
                <div class="collections">
                    ${symbol.collections.map(c => `<span class="badge">${c}</span>`).join('')}
                </div>
            </td>
            <td>
                <button
                    class="btn btn-primary btn-sm"
                    onclick="viewFiles('${symbol.symbol}')">
                    View Files
                </button>
            </td>
        </tr>
    `).join('');
}

// Handle search
function handleSearch(event) {
    const searchTerm = event.target.value.toLowerCase().trim();

    if (!searchTerm) {
        renderSymbolsTable(allSymbols);
        return;
    }

    const filtered = allSymbols.filter(symbol =>
        symbol.symbol.toLowerCase().includes(searchTerm) ||
        symbol.fincode.toString().includes(searchTerm)
    );

    renderSymbolsTable(filtered);
}

// View files for a symbol
async function viewFiles(symbol) {
    currentSymbol = symbol;
    const modal = document.getElementById('file-modal');
    const modalSymbol = document.getElementById('modal-symbol');
    const modalLoading = document.getElementById('modal-loading');
    const filesTable = document.getElementById('files-table');
    const filesTableBody = document.getElementById('files-table-body');

    // Show modal
    modal.style.display = 'flex';
    modalSymbol.textContent = symbol;
    modalLoading.style.display = 'flex';
    filesTable.style.display = 'none';
    filesTableBody.innerHTML = '';

    try {
        const response = await fetch(`/api/symbols/${symbol}/files`);
        if (!response.ok) throw new Error('Failed to load files');

        const files = await response.json();

        // Render files table
        filesTableBody.innerHTML = files.map(file => {
            const embeddedDate = new Date(file.embedded_at).toLocaleDateString();
            const filename = file.source_filename;

            // Format document date from metadata
            const documentDate = file.newsDt
                ? new Date(file.newsDt).toLocaleDateString()
                : '-';

            // Get subcategory from metadata
            const subcategory = file.subcatname || '-';

            return `
                <tr>
                    <td title="${filename}" class="filename-cell">
                        ${truncateFilename(filename, 40)}
                    </td>
                    <td>${subcategory}</td>
                    <td>${documentDate}</td>
                    <td>
                        <span class="badge">${file.collection_name}</span>
                    </td>
                    <td>${file.document_count || 0}</td>
                    <td>${embeddedDate}</td>
                    <td>
                        <button
                            class="btn btn-success btn-sm"
                            onclick="downloadFile('${filename}', '${file.collection_name}')">
                            Download
                        </button>
                    </td>
                </tr>
            `;
        }).join('');

        modalLoading.style.display = 'none';
        filesTable.style.display = 'table';

    } catch (error) {
        modalLoading.innerHTML = `
            <div class="error-message">
                <p>Failed to load files: ${error.message}</p>
            </div>
        `;
    }
}

// Download a file
async function downloadFile(filename, category) {
    try {
        // Show downloading indicator
        const btn = event.target;
        const originalText = btn.textContent;
        btn.textContent = 'Downloading...';
        btn.disabled = true;

        const url = `/api/download/${encodeURIComponent(filename)}?category=${encodeURIComponent(category)}`;

        const response = await fetch(url);
        if (!response.ok) throw new Error('Download failed');

        // Create blob and trigger download
        const blob = await response.blob();
        const downloadUrl = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(downloadUrl);
        document.body.removeChild(a);

        // Reset button
        btn.textContent = originalText;
        btn.disabled = false;

    } catch (error) {
        alert('Failed to download file: ' + error.message);
        // Reset button
        event.target.textContent = 'Download';
        event.target.disabled = false;
    }
}

// Close modal
function closeModal() {
    const modal = document.getElementById('file-modal');
    modal.style.display = 'none';
    currentSymbol = null;
}

// Utility: Truncate filename
function truncateFilename(filename, maxLength) {
    if (filename.length <= maxLength) return filename;

    const extension = filename.split('.').pop();
    const nameWithoutExt = filename.substring(0, filename.lastIndexOf('.'));
    const truncatedName = nameWithoutExt.substring(0, maxLength - extension.length - 4) + '...';

    return truncatedName + '.' + extension;
}

// Show error message
function showError(message) {
    const errorDiv = document.getElementById('error');
    const errorText = document.getElementById('error-text');

    errorText.textContent = message;
    errorDiv.style.display = 'block';
}

// Hide error message
function hideError() {
    const errorDiv = document.getElementById('error');
    errorDiv.style.display = 'none';
}
