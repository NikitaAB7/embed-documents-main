// Global state
let allSymbols = [];
let currentSymbol = null;
let lastQueryFilters = null;
let currentChunks = [];
const pdfCache = new Map();
let currentRenderTask = null; // Track current PDF render task
const pdfViewerState = {
    pdfDoc: null,
    currentPage: 1,
    totalPages: 1,
    chunkText: '',
    filename: '',
    category: '',
    symbol: '',
    pageHint: 1,
    cacheKey: null,
    scale: 1.25,
    // Citation bbox info
    bbox: null,
    bboxPage: null,
    bboxCoordOrigin: null,
};

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

    const filterControls = ['filter-symbol', 'filter-category', 'filter-date-from', 'filter-date-to'];
    filterControls.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            const eventType = el.tagName === 'SELECT' ? 'change' : 'input';
            el.addEventListener(eventType, updateActiveFiltersDisplay);
        }
    });

    const clearFiltersBtn = document.getElementById('clear-filters');
    if (clearFiltersBtn) {
        clearFiltersBtn.addEventListener('click', () => {
            clearFilters();
            updateActiveFiltersDisplay();
        });
    }

    // Close modal on outside click
    const modal = document.getElementById('file-modal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeModal();
            }
        });
    }

    const pdfModal = document.getElementById('pdf-modal');
    if (pdfModal) {
        pdfModal.addEventListener('click', (e) => {
            if (e.target === pdfModal) {
                closePdfModal();
            }
        });
    }

    // Close modal on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeModal();
            closePdfModal();
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
    const filters = getQueryFilters();
    lastQueryFilters = filters ? { ...filters } : null;

    // Show loading state
    const resultsDiv = document.getElementById('query-results');
    const loadingDiv = document.getElementById('query-loading');
    const answerDiv = document.getElementById('query-answer');
    const chunksDiv = document.getElementById('query-chunks');
    const queryContext = document.getElementById('query-context');
    const queryBtn = document.getElementById('query-btn');

    resultsDiv.style.display = 'block';
    loadingDiv.style.display = 'flex';
    answerDiv.style.display = 'none';
    chunksDiv.style.display = 'none';
    if (queryContext) {
        queryContext.style.display = 'none';
    }
    queryBtn.disabled = true;
    queryBtn.textContent = 'Searching...';

    try {
        const payload = {
            query: query,
            top_k: topK,
            use_hyde: useHyde,
            synthesize: synthesize,
            strict_citations: true,
            structured_output: false,
            evaluate_faithfulness: false,
        };

        if (filters) {
            payload.filters = filters;
        }

        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Query failed');
        }

        const data = await response.json();
        displayQueryResults(data, lastQueryFilters);

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
function displayQueryResults(data, appliedFilters = null) {
    const loadingDiv = document.getElementById('query-loading');
    const answerDiv = document.getElementById('query-answer');
    const chunksDiv = document.getElementById('query-chunks');
    const queryContext = document.getElementById('query-context');
    const routeInfo = document.getElementById('route-info');
    const contextFilters = document.getElementById('context-filters');

    loadingDiv.style.display = 'none';
    currentChunks = Array.isArray(data.results) ? data.results : [];

    if (queryContext && routeInfo && contextFilters) {
        routeInfo.innerHTML = buildRouteInfo(data);
        contextFilters.innerHTML = buildFilterChips(appliedFilters);
        queryContext.style.display = 'flex';
    }

    // Check if RAG was used
    if (!data.use_rag) {
        const confidence = typeof data.confidence === 'number'
            ? `${(data.confidence * 100).toFixed(1)}%`
            : 'N/A';
        const reasonText = escapeHtml(data.reason || 'LLM responded directly.');
        chunksDiv.innerHTML = `
            <div class="no-results">
                <p>This query doesn't require document search.</p>
                <p>Confidence: ${confidence} | Reason: ${reasonText}</p>
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
            const citationCards = data.citations.map(renderCitation).join('');
            citationsDiv.innerHTML = `
                <strong>Sources:</strong>
                <div class="citation-list">${citationCards}</div>
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
            const source = chunk.source || chunk.metadata?.source || 'Unknown';
            const category = chunk.metadata?.category || '';
            const ticker = chunk.metadata?.symbol || chunk.metadata?.ticker || '';
            const page = chunk.metadata?.page || chunk.metadata?.page_no || '';
            const docDate = chunk.metadata?.document_date || chunk.metadata?.newsDt || '';
            const fincode = chunk.metadata?.fincode || '';
            const chunkId = normalizeChunkId(chunk.chunk_id || chunk.metadata?.chunk_id || chunk.metadata?.reference_chunk_id || `chunk-${index}`);
            const citationAnchor = `C${index + 1}`;

            const detailItems = [];
            if (docDate) detailItems.push(`Document date: ${docDate}`);
            if (fincode) detailItems.push(`Fincode: ${fincode}`);

            return `
                <div class="chunk-item" data-chunk-id="${chunkId}" data-citation-id="${citationAnchor}">
                    <div class="chunk-header">
                        <span class="chunk-source" title="${escapeHtml(source)}">${truncateFilename(source, 60)}</span>
                        <div class="chunk-meta">
                            ${ticker ? `<span class="badge">${ticker}</span>` : ''}
                            ${category ? `<span class="badge">${category}</span>` : ''}
                            ${page ? `<span class="badge">Pg ${page}</span>` : ''}
                            <span class="badge">#${index + 1}</span>
                        </div>
                    </div>
                    ${detailItems.length ? `<div class="chunk-details">${detailItems.map(item => `<span>${escapeHtml(item)}</span>`).join('')}</div>` : ''}
                    <div class="chunk-content">${escapeHtml(chunk.content)}</div>
                    <div class="chunk-actions">
                        <button class="btn btn-link" type="button" onclick="openPdfViewerByIndex(${index})">
                            View in PDF
                        </button>
                    </div>
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

        populateSymbolFilter();
        updateActiveFiltersDisplay();

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
            <td class="action-cell">
                <button
                    class="btn btn-primary btn-sm"
                    onclick="viewFiles('${symbol.symbol}')">
                    View Files
                </button>
                <button
                    class="btn btn-secondary btn-sm"
                    onclick="applySymbolFilter('${symbol.symbol}')">
                    Ask About Symbol
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
    if (!filename) return 'Unknown source';
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

function populateSymbolFilter() {
    const select = document.getElementById('filter-symbol');
    if (!select) return;

    const previous = select.value;
    const options = ['<option value="">All symbols</option>'];
    allSymbols.forEach(symbol => {
        const value = symbol.symbol;
        options.push(`<option value="${value}">${escapeHtml(value)}</option>`);
    });
    select.innerHTML = options.join('');

    if (previous && allSymbols.some(item => item.symbol === previous)) {
        select.value = previous;
    }
}

function getQueryFilters() {
    const filters = {};
    const symbolValue = document.getElementById('filter-symbol')?.value;
    const categoryValue = document.getElementById('filter-category')?.value;
    const dateFromValue = document.getElementById('filter-date-from')?.value;
    const dateToValue = document.getElementById('filter-date-to')?.value;

    if (symbolValue) filters.symbol = symbolValue;
    if (categoryValue) filters.category = categoryValue;
    if (dateFromValue) filters.date_from = dateFromValue;
    if (dateToValue) filters.date_to = dateToValue;

    return Object.keys(filters).length ? filters : null;
}

function updateActiveFiltersDisplay() {
    const chipsContainer = document.getElementById('active-filter-chips');
    if (!chipsContainer) return;
    const filters = getQueryFilters();
    chipsContainer.innerHTML = buildFilterChips(filters);
}

function buildFilterChips(filters) {
    if (!filters || Object.keys(filters).length === 0) {
        return '<span class="filter-chip muted">No filters applied</span>';
    }

    return Object.entries(filters)
        .map(([key, value]) => `<span class="filter-chip">${escapeHtml(formatFilterLabel(key, value))}</span>`)
        .join('');
}

function formatFilterLabel(key, value) {
    switch (key) {
        case 'symbol':
            return `Symbol: ${value}`;
        case 'category':
            return `Category: ${value}`;
        case 'date_from':
            return `From: ${value}`;
        case 'date_to':
            return `To: ${value}`;
        default:
            return `${key}: ${value}`;
    }
}

function clearFilters() {
    const ids = ['filter-symbol', 'filter-category', 'filter-date-from', 'filter-date-to'];
    ids.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.value = '';
        }
    });
}

function buildRouteInfo(data) {
    const confidence = typeof data.confidence === 'number'
        ? `${(data.confidence * 100).toFixed(1)}%`
        : 'N/A';
    const reason = data.reason || 'No routing reason provided.';
    const faithfulness = typeof data.faithfulness_score === 'number'
        ? `<span class="route-extra">Faithfulness ${(data.faithfulness_score * 100).toFixed(0)}%</span>`
        : '';
    const ragStatus = data.use_rag
        ? '<span class="route-pill success">RAG search</span>'
        : '<span class="route-pill warning">LLM only</span>';

    return `
        ${ragStatus}
        <span class="route-extra">Confidence ${confidence}</span>
        ${faithfulness}
        <span class="route-reason">${escapeHtml(reason)}</span>
    `;
}

function renderCitation(citation) {
    const label = citation.id || 'Source';
    const sourceLabel = truncateFilename(citation.source || 'Unknown source', 50);
    const pageLabel = citation.page ? `Page ${citation.page}` : null;
    const chunkLabel = citation.chunk_id ? `Chunk ${citation.chunk_id}` : null;
    const metaParts = [sourceLabel];
    if (pageLabel) metaParts.push(pageLabel);
    if (chunkLabel) metaParts.push(chunkLabel);

    const normalizedChunkRef = citation.chunk_id ? normalizeChunkId(citation.chunk_id) : null;
    const normalizedReference = citation.reference_chunk_id ? normalizeChunkId(citation.reference_chunk_id) : null;
    const chunkTarget = citation.id || normalizedChunkRef || normalizedReference;

    // Build action buttons
    let actionButtons = '';
    
    if (chunkTarget) {
        actionButtons += `<button class="btn btn-link" type="button" onclick="scrollToChunk('${chunkTarget}')">Jump to chunk</button>`;
    }
    
    // Add "View in PDF" button if we have source and page info
    if (citation.source && citation.page) {
        const bboxAttr = citation.bbox ? `data-bbox='${JSON.stringify(citation.bbox)}'` : '';
        const coordOrigin = citation.bbox?.coord_origin || citation.coord_origin || 'TOPLEFT';
        actionButtons += ` <button class="btn btn-link" type="button" 
            data-filename="${escapeHtml(citation.source)}" 
            data-page="${citation.page}"
            data-coord-origin="${coordOrigin}"
            ${bboxAttr}
            onclick="openPdfFromCitation(this)">View in PDF</button>`;
    }

    return `
        <div class="citation-card">
            <div>
                <div class="citation-label">${escapeHtml(label)}</div>
                <div class="citation-meta">
                    ${metaParts.map(part => `<span>${escapeHtml(part)}</span>`).join('')}
                </div>
            </div>
            <div class="citation-actions">${actionButtons}</div>
        </div>
    `;
}

function openPdfFromCitation(button) {
    const filename = button.getAttribute('data-filename');
    const page = parseInt(button.getAttribute('data-page'), 10) || 1;
    const coordOrigin = button.getAttribute('data-coord-origin') || 'TOPLEFT';
    let bbox = null;
    
    try {
        const bboxAttr = button.getAttribute('data-bbox');
        if (bboxAttr) {
            bbox = JSON.parse(bboxAttr);
        }
    } catch (e) {
        console.warn('Failed to parse bbox:', e);
    }
    
    // Try to find the chunk in currentChunks that matches this citation
    const matchingChunk = currentChunks.find(chunk => {
        const meta = chunk.metadata || {};
        const chunkSource = chunk.source || meta.source || meta.filename || '';
        const chunkPage = parseInt(meta.page || meta.page_no || meta.pageNumber, 10);
        return chunkSource === filename && chunkPage === page;
    });
    
    if (matchingChunk) {
        // If bbox was provided in citation, inject it into the chunk metadata
        if (bbox && !matchingChunk.metadata?.bbox) {
            matchingChunk.metadata = matchingChunk.metadata || {};
            matchingChunk.metadata.bbox = bbox;
            matchingChunk.metadata.coord_origin = coordOrigin;
        }
        openPdfViewer(matchingChunk).catch(error => {
            console.error('PDF viewer error:', error);
            showPdfError('Failed to open PDF: ' + error.message);
        });
    } else {
        // Create a synthetic chunk for the PDF viewer
        const syntheticChunk = {
            source: filename,
            content: `Citation from ${filename}, page ${page}`,
            metadata: {
                source: filename,
                page: page,
                bbox: bbox,
                coord_origin: coordOrigin
            }
        };
        openPdfViewer(syntheticChunk).catch(error => {
            console.error('PDF viewer error:', error);
            showPdfError('Failed to open PDF: ' + error.message);
        });
    }
}

function scrollToChunk(anchorId) {
    if (!anchorId) return;

    let target = document.querySelector(`[data-citation-id="${anchorId}"]`);
    if (!target) {
        target = document.querySelector(`[data-chunk-id="${anchorId}"]`);
    }

    if (target) {
        target.classList.add('chunk-highlight');
        target.scrollIntoView({ behavior: 'smooth', block: 'center' });
        setTimeout(() => target.classList.remove('chunk-highlight'), 2000);
    }
}

function normalizeChunkId(value) {
    return value ? encodeURIComponent(value) : '';
}

function applySymbolFilter(symbol) {
    const select = document.getElementById('filter-symbol');
    if (!select) return;

    select.value = symbol;
    updateActiveFiltersDisplay();

    const querySection = document.querySelector('.query-section');
    if (querySection) {
        querySection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    const queryInput = document.getElementById('query-input');
    if (queryInput) {
        queryInput.focus();
    }
}

function openPdfViewerByIndex(index) {
    const chunk = currentChunks[index];
    if (!chunk) {
        alert('Unable to locate this chunk. Please refresh and try again.');
        return;
    }
    openPdfViewer(chunk).catch(error => {
        console.error('PDF viewer error:', error);
        showPdfError('Failed to open PDF: ' + error.message);
    });
}

async function openPdfViewer(chunk) {
    const modal = document.getElementById('pdf-modal');
    if (!modal) return;

    const metadata = chunk.metadata || {};
    const filename = chunk.source || metadata.source || metadata.filename;
    const category = metadata.collection_name || metadata.category || metadata.subcatname || '';
    if (!filename) {
        showPdfError('This chunk is missing its source filename.');
        return;
    }

    resetPdfModal();
    modal.style.display = 'flex';
    setPdfLoadingState(true);

    const symbol = metadata.symbol || metadata.ticker || '';
    const pdfSymbol = document.getElementById('pdf-symbol');
    if (pdfSymbol) {
        if (symbol) {
            pdfSymbol.style.display = 'inline-block';
            pdfSymbol.textContent = symbol;
        } else {
            pdfSymbol.style.display = 'none';
        }
    }

    const pdfFilename = document.getElementById('pdf-filename');
    if (pdfFilename) {
        pdfFilename.textContent = filename;
    }

    const chunkTextEl = document.getElementById('pdf-chunk-text');
    if (chunkTextEl) {
        chunkTextEl.textContent = chunk.content || 'No chunk text available.';
    }

    if (!window.pdfjsLib) {
        showPdfError('PDF viewer library failed to load. Please refresh the page.');
        return;
    }

    // DEBUG: Log metadata to see what's available
    console.log('=== PDF Viewer Debug ===');
    console.log('Chunk metadata:', JSON.stringify(metadata, null, 2));
    console.log('Has bbox?', !!metadata.bbox);
    console.log('Has bounding_box?', !!metadata.bounding_box);
    console.log('Has page?', metadata.page);
    console.log('Has dl_meta?', !!metadata.dl_meta);

    // Extract bbox info for citation highlighting
    let bbox = null;
    let bboxPage = null;
    let bboxCoordOrigin = null;
    
    // Check for bbox in various possible locations
    if (metadata.bbox) {
        bbox = metadata.bbox;
        bboxCoordOrigin = bbox.coord_origin || metadata.coord_origin || 'TOPLEFT';
        console.log('Using metadata.bbox:', bbox);
    } else if (metadata.bounding_box) {
        bbox = metadata.bounding_box;
        bboxCoordOrigin = bbox.coord_origin || metadata.coord_origin || 'TOPLEFT';
        console.log('Using metadata.bounding_box:', bbox);
    } else if (metadata.dl_meta) {
        // Try to extract from Docling metadata
        try {
            const dlMeta = typeof metadata.dl_meta === 'string' 
                ? JSON.parse(metadata.dl_meta) 
                : metadata.dl_meta;
            const prov = dlMeta?.doc_items?.[0]?.prov?.[0];
            if (prov?.bbox) {
                bbox = prov.bbox;
                bboxCoordOrigin = prov.coord_origin || 'BOTTOMLEFT';
                console.log('Extracted bbox from dl_meta:', bbox, 'origin:', bboxCoordOrigin);
            }
        } catch (e) {
            console.warn('Failed to parse dl_meta for bbox:', e);
        }
    }
    
    // Get bbox page - could be stored separately or within bbox object
    const pageHint = parseInt(metadata.page || metadata.page_no || metadata.pageNumber, 10);
    if (bbox && pageHint) {
        bboxPage = pageHint;
    }
    
    // Get page dimensions for normalization (if stored)
    const pageWidth = metadata.page_width || null;
    const pageHeight = metadata.page_height || null;
    
    console.log('Final bbox:', bbox, 'page:', bboxPage, 'origin:', bboxCoordOrigin);
    console.log('Page dimensions:', pageWidth, 'x', pageHeight);

    try {
        const pdfBytes = await fetchPdfBytes(filename, category);
        const pdfDoc = await pdfjsLib.getDocument({ data: pdfBytes }).promise;

        pdfViewerState.pdfDoc = pdfDoc;
        pdfViewerState.filename = filename;
        pdfViewerState.category = category;
        pdfViewerState.symbol = symbol;
        pdfViewerState.chunkText = chunk.content || '';
        pdfViewerState.totalPages = pdfDoc.numPages;
        pdfViewerState.currentPage = pageHint && pageHint > 0 ? Math.min(pageHint, pdfDoc.numPages) : 1;
        
        // Store bbox info for highlighting
        pdfViewerState.bbox = bbox;
        pdfViewerState.bboxPage = bboxPage;
        pdfViewerState.bboxCoordOrigin = bboxCoordOrigin;
        pdfViewerState.bboxPageWidth = pageWidth;
        pdfViewerState.bboxPageHeight = pageHeight;
       
        await renderPdfPage(pdfViewerState.currentPage);
        setPdfLoadingState(false);
        updatePdfNavButtons();
    } catch (error) {
        showPdfError(error.message);
        throw error;
    }
}

async function fetchPdfBytes(filename, category) {
    const cacheKey = `${filename}|${category || 'NA'}`;
    if (pdfCache.has(cacheKey)) {
        const cachedBuffer = pdfCache.get(cacheKey);
        return cachedBuffer.slice(0);
    }

    let url = `/api/download/${encodeURIComponent(filename)}`;
    if (category) {
        url += `?category=${encodeURIComponent(category)}`;
    }

    const response = await fetch(url);
    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Unable to download PDF file.');
    }

    const buffer = await response.arrayBuffer();
    pdfCache.set(cacheKey, buffer.slice(0));
    return buffer.slice(0);
}

async function renderPdfPage(pageNumber) {
    if (!pdfViewerState.pdfDoc) return;

    // Cancel any in-progress render task
    if (currentRenderTask) {
        try {
            currentRenderTask.cancel();
        } catch (e) {
            // Ignore cancel errors
        }
        currentRenderTask = null;
    }

    const page = await pdfViewerState.pdfDoc.getPage(pageNumber);
    const viewport = page.getViewport({ scale: pdfViewerState.scale });

    const canvas = document.getElementById('pdf-canvas');
    const pageContainer = document.getElementById('pdf-page-container');
    const context = canvas.getContext('2d');

    // CRITICAL: Set canvas dimensions to match viewport exactly
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    canvas.style.width = `${viewport.width}px`;
    canvas.style.height = `${viewport.height}px`;

    // Set container dimensions to match
    if (pageContainer) {
        pageContainer.style.width = `${viewport.width}px`;
        pageContainer.style.height = `${viewport.height}px`;
    }

    // CRITICAL: Clear canvas before rendering
    context.clearRect(0, 0, canvas.width, canvas.height);

    // Remove any existing highlight overlays
    clearHighlightOverlays();

    // Render the page
    const renderContext = {
        canvasContext: context,
        viewport: viewport
    };
    
    try {
        currentRenderTask = page.render(renderContext);
        await currentRenderTask.promise;
        currentRenderTask = null;
    } catch (error) {
        if (error.name === 'RenderingCancelledException') {
            // Render was cancelled, ignore
            return;
        }
        throw error;
    }

    // Draw bbox highlight if on the correct page
    if (pdfViewerState.bbox && pdfViewerState.bboxPage === pageNumber) {
        drawBboxHighlight(
            canvas, 
            pdfViewerState.bbox, 
            pdfViewerState.bboxCoordOrigin,
            pdfViewerState.bboxPageWidth,
            pdfViewerState.bboxPageHeight
        );
    }

    // Update page indicator
    const indicator = document.getElementById('pdf-page-indicator');
    if (indicator) {
        indicator.textContent = `Page ${pageNumber} / ${pdfViewerState.totalPages}`;
    }

    // Update zoom indicator
    const zoomIndicator = document.getElementById('pdf-zoom-indicator');
    if (zoomIndicator) {
        zoomIndicator.textContent = `${Math.round(pdfViewerState.scale * 100)}%`;
    }
}

function clearHighlightOverlays() {
    const pageContainer = document.getElementById('pdf-page-container');
    if (!pageContainer) return;
    const overlays = pageContainer.querySelectorAll('.pdf-highlight-overlay');
    overlays.forEach(el => el.remove());
}

function drawBboxHighlight(canvas, bbox, coordOrigin, pageWidth, pageHeight) {
    if (!bbox || !canvas) return;

    const pageContainer = document.getElementById('pdf-page-container');
    if (!pageContainer) return;

    console.log('Drawing bbox:', bbox, 'origin:', coordOrigin, 'pageSize:', pageWidth, 'x', pageHeight);

    // Determine bbox format and normalize to pixel coordinates
    let left, top, width, height;
    
    // Check if bbox is in {x, y, w, h} format (absolute PDF points)
    if (bbox.x !== undefined && bbox.y !== undefined && bbox.w !== undefined && bbox.h !== undefined) {
        // Absolute PDF points format - need to scale to canvas size
        const pdfWidth = pageWidth || 595;  // Default A4 width in points
        const pdfHeight = pageHeight || 842; // Default A4 height in points
        
        // Skip full-page bboxes (covers ≥90% of page area and starts near origin)
        const nearOrigin = bbox.x <= 5 && bbox.y <= 5;
        const bboxArea = bbox.w * bbox.h;
        const pageArea = pdfWidth * pdfHeight;
        const coverage = pageArea > 0 ? bboxArea / pageArea : 0;
        if (nearOrigin && coverage >= 0.90) {
            console.log('Skipping full-page bbox (coverage:', (coverage * 100).toFixed(1) + '%)');
            return;
        }
        
        const scaleX = canvas.width / pdfWidth;
        const scaleY = canvas.height / pdfHeight;
        
        left = bbox.x * scaleX;
        width = bbox.w * scaleX;
        
        if (coordOrigin === 'BOTTOMLEFT') {
            // y is from bottom in PDF coordinates
            top = canvas.height - (bbox.y + bbox.h) * scaleY;
        } else {
            // y is from top
            top = bbox.y * scaleY;
        }
        height = bbox.h * scaleY;
    } else if (bbox.l !== undefined && bbox.t !== undefined && bbox.r !== undefined && bbox.b !== undefined) {
        // Normalized {l, t, r, b} format (0-1 range)
        if (coordOrigin === 'BOTTOMLEFT') {
            top = (1 - bbox.t) * canvas.height;
            const bottom = (1 - bbox.b) * canvas.height;
            height = bottom - top;
        } else {
            top = bbox.t * canvas.height;
            height = (bbox.b - bbox.t) * canvas.height;
        }
        left = bbox.l * canvas.width;
        width = (bbox.r - bbox.l) * canvas.width;
    } else {
        console.warn('Unknown bbox format:', bbox);
        return;
    }
    
    console.log('Computed highlight rect:', left, top, width, height);

    // Create highlight overlay div
    const highlight = document.createElement('div');
    highlight.className = 'pdf-highlight-overlay';
    highlight.style.position = 'absolute';
    highlight.style.left = `${left}px`;
    highlight.style.top = `${top}px`;
    highlight.style.width = `${width}px`;
    highlight.style.height = `${height}px`;

    pageContainer.appendChild(highlight);

    // Scroll the highlight into view
    setTimeout(() => {
        highlight.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);
}

function highlightTextLayer(container, chunkText) {
    // Deprecated - using canvas-only rendering now
}

function changePdfPage(delta) {
    if (!pdfViewerState.pdfDoc) return;
    const target = pdfViewerState.currentPage + delta;
    if (target < 1 || target > pdfViewerState.totalPages) {
        return;
    }
    pdfViewerState.currentPage = target;
    renderPdfPage(pdfViewerState.currentPage);
    updatePdfNavButtons();
}

function changePdfZoom(delta) {
    if (!pdfViewerState.pdfDoc) return;
    const newScale = Math.max(0.5, Math.min(3.0, pdfViewerState.scale + delta));
    if (newScale === pdfViewerState.scale) return;
    pdfViewerState.scale = newScale;
    renderPdfPage(pdfViewerState.currentPage);
}

function updatePdfNavButtons() {
    const prevBtn = document.getElementById('pdf-prev-btn');
    const nextBtn = document.getElementById('pdf-next-btn');
    const hasDoc = Boolean(pdfViewerState.pdfDoc);
    if (prevBtn) {
        prevBtn.disabled = !hasDoc || pdfViewerState.currentPage <= 1;
    }
    if (nextBtn) {
        nextBtn.disabled = !hasDoc || pdfViewerState.currentPage >= pdfViewerState.totalPages;
    }
}

function closePdfModal() {
    const modal = document.getElementById('pdf-modal');
    if (modal) {
        modal.style.display = 'none';
    }
    // Cancel any in-progress render
    if (currentRenderTask) {
        try {
            currentRenderTask.cancel();
        } catch (e) {
            // Ignore
        }
        currentRenderTask = null;
    }
    // Clear state
    pdfViewerState.pdfDoc = null;
    pdfViewerState.chunkText = '';
    pdfViewerState.bbox = null;
    pdfViewerState.bboxPage = null;
    pdfViewerState.bboxCoordOrigin = null;
    // Clear highlight overlays
    clearHighlightOverlays();
    updatePdfNavButtons();
}

function resetPdfModal() {
    const viewer = document.getElementById('pdf-viewer');
    const errorDiv = document.getElementById('pdf-error');
    const loading = document.getElementById('pdf-loading');
    if (viewer) viewer.style.display = 'none';
    if (errorDiv) errorDiv.style.display = 'none';
    if (loading) loading.style.display = 'none';
}

function setPdfLoadingState(isLoading) {
    const loading = document.getElementById('pdf-loading');
    const viewer = document.getElementById('pdf-viewer');
    const errorDiv = document.getElementById('pdf-error');
    if (loading) loading.style.display = isLoading ? 'flex' : 'none';
    if (viewer && !isLoading) viewer.style.display = 'flex';
    if (errorDiv && isLoading) errorDiv.style.display = 'none';
}

function showPdfError(message) {
    const errorDiv = document.getElementById('pdf-error');
    const errorText = document.getElementById('pdf-error-text');
    const loading = document.getElementById('pdf-loading');
    const viewer = document.getElementById('pdf-viewer');
    if (loading) loading.style.display = 'none';
    if (viewer) viewer.style.display = 'none';
    if (errorDiv && errorText) {
        errorText.textContent = message;
        errorDiv.style.display = 'block';
    }
    updatePdfNavButtons();
}
