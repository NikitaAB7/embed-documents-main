/**
 * Document Browser - Frontend JavaScript
 * Handles document listing, viewing, and downloading
 */

// State
let currentPage = 0;
let totalDocuments = 0;
let currentFilename = null;
const PAGE_SIZE = 50;

// DOM Elements
const companySelect = document.getElementById('company-select');
const categorySelect = document.getElementById('category-select');
const searchBtn = document.getElementById('search-btn');
const loadingEl = document.getElementById('loading');
const emptyStateEl = document.getElementById('empty-state');
const documentsListEl = document.getElementById('documents-list');
const resultsCountEl = document.getElementById('results-count');
const paginationEl = document.getElementById('pagination');
const pageInfoEl = document.getElementById('page-info');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');
const modalEl = document.getElementById('document-modal');
const modalTitleEl = document.getElementById('modal-title');
const modalLoadingEl = document.getElementById('modal-loading');
const documentFrameEl = document.getElementById('document-frame');

/**
 * Initialize the application
 */
async function init() {
    await Promise.all([
        loadCompanies(),
        loadCategories()
    ]);
}

/**
 * Load companies for the dropdown
 */
async function loadCompanies() {
    try {
        const response = await fetch('/api/companies');
        if (!response.ok) throw new Error('Failed to load companies');
        
        const companies = await response.json();
        
        // Sort by symbol
        companies.sort((a, b) => a.symbol.localeCompare(b.symbol));
        
        // Populate dropdown
        companySelect.innerHTML = '<option value="">All companies</option>';
        companies.forEach(company => {
            const option = document.createElement('option');
            option.value = company.symbol;
            option.textContent = `${company.symbol} (${company.file_count} docs)`;
            companySelect.appendChild(option);
        });
        
    } catch (error) {
        console.error('Error loading companies:', error);
    }
}

/**
 * Load document categories for the dropdown
 */
async function loadCategories() {
    try {
        const response = await fetch('/api/categories');
        if (!response.ok) throw new Error('Failed to load categories');
        
        const categories = await response.json();
        
        // Populate dropdown
        categorySelect.innerHTML = '<option value="">All document types</option>';
        categories.forEach(cat => {
            const option = document.createElement('option');
            option.value = cat.value;
            option.textContent = cat.label;
            categorySelect.appendChild(option);
        });
        
    } catch (error) {
        console.error('Error loading categories:', error);
    }
}

/**
 * Search for documents based on filters
 */
async function searchDocuments() {
    currentPage = 0;
    await loadDocuments();
}

/**
 * Load documents from the API
 */
async function loadDocuments() {
    // Show loading state
    loadingEl.style.display = 'flex';
    emptyStateEl.style.display = 'none';
    documentsListEl.style.display = 'none';
    paginationEl.style.display = 'none';
    searchBtn.disabled = true;
    
    try {
        // Build query params
        const params = new URLSearchParams();
        
        const symbol = companySelect.value;
        const category = categorySelect.value;
        
        if (symbol) params.append('symbol', symbol);
        if (category) params.append('category', category);
        params.append('limit', PAGE_SIZE);
        params.append('offset', currentPage * PAGE_SIZE);
        
        // Fetch documents
        const response = await fetch(`/api/documents?${params.toString()}`);
        if (!response.ok) throw new Error('Failed to load documents');
        
        const data = await response.json();
        totalDocuments = data.total;
        
        // Update results count
        resultsCountEl.textContent = `${totalDocuments} document${totalDocuments !== 1 ? 's' : ''} found`;
        
        // Render documents
        if (data.documents.length === 0) {
            emptyStateEl.style.display = 'flex';
            emptyStateEl.querySelector('h3').textContent = 'No documents found';
            emptyStateEl.querySelector('p').textContent = 'Try adjusting your search filters';
        } else {
            renderDocuments(data.documents);
            documentsListEl.style.display = 'flex';
            updatePagination();
        }
        
    } catch (error) {
        console.error('Error loading documents:', error);
        emptyStateEl.style.display = 'flex';
        emptyStateEl.querySelector('h3').textContent = 'Error loading documents';
        emptyStateEl.querySelector('p').textContent = error.message;
    } finally {
        loadingEl.style.display = 'none';
        searchBtn.disabled = false;
    }
}

/**
 * Render documents to the list
 */
function renderDocuments(documents) {
    documentsListEl.innerHTML = '';
    
    documents.forEach(doc => {
        const card = document.createElement('div');
        card.className = 'document-card';
        
        // Format date
        const dateStr = doc.document_date 
            ? new Date(doc.document_date).toLocaleDateString('en-IN', {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            })
            : 'Unknown date';
        
        // Truncate filename for display
        const displayName = doc.filename.length > 50 
            ? doc.filename.substring(0, 47) + '...'
            : doc.filename;
        
        card.innerHTML = `
            <div class="document-info">
                <div class="document-title" title="${escapeHtml(doc.filename)}">${escapeHtml(displayName)}</div>
                <div class="document-meta">
                    ${doc.symbol ? `<span>🏢 ${escapeHtml(doc.symbol)}</span>` : ''}
                    ${doc.category ? `<span class="document-category">${escapeHtml(doc.category)}</span>` : ''}
                    <span>📅 ${dateStr}</span>
                    ${doc.chunk_count ? `<span>📄 ${doc.chunk_count} chunks</span>` : ''}
                </div>
            </div>
            <div class="document-actions">
                <button class="btn btn-primary btn-sm" onclick="viewDocument('${escapeJsString(doc.filename)}')">
                    View
                </button>
                <button class="btn btn-secondary btn-sm" onclick="downloadDocument('${escapeJsString(doc.filename)}')">
                    Download
                </button>
            </div>
        `;
        
        documentsListEl.appendChild(card);
    });
}

/**
 * Update pagination controls
 */
function updatePagination() {
    const totalPages = Math.ceil(totalDocuments / PAGE_SIZE);
    
    if (totalPages <= 1) {
        paginationEl.style.display = 'none';
        return;
    }
    
    paginationEl.style.display = 'flex';
    pageInfoEl.textContent = `Page ${currentPage + 1} of ${totalPages}`;
    
    prevBtn.disabled = currentPage === 0;
    nextBtn.disabled = currentPage >= totalPages - 1;
}

/**
 * Go to previous page
 */
function prevPage() {
    if (currentPage > 0) {
        currentPage--;
        loadDocuments();
    }
}

/**
 * Go to next page
 */
function nextPage() {
    const totalPages = Math.ceil(totalDocuments / PAGE_SIZE);
    if (currentPage < totalPages - 1) {
        currentPage++;
        loadDocuments();
    }
}

/**
 * View a document in the modal
 */
function viewDocument(filename) {
    currentFilename = filename;
    
    // Show modal
    modalEl.style.display = 'flex';
    modalTitleEl.textContent = filename;
    modalLoadingEl.style.display = 'flex';
    documentFrameEl.style.display = 'none';
    
    // Load document in iframe
    const encodedFilename = encodeURIComponent(filename);
    documentFrameEl.src = `/api/documents/${encodedFilename}`;
    
    documentFrameEl.onload = () => {
        modalLoadingEl.style.display = 'none';
        documentFrameEl.style.display = 'block';
    };
    
    documentFrameEl.onerror = () => {
        modalLoadingEl.innerHTML = `
            <div class="empty-icon">❌</div>
            <h3>Failed to load document</h3>
            <p>The document could not be loaded. Try downloading it instead.</p>
        `;
    };
}

/**
 * Download a document
 */
function downloadDocument(filename) {
    const downloadFilename = filename || currentFilename;
    if (!downloadFilename) return;
    
    const encodedFilename = encodeURIComponent(downloadFilename);
    const downloadUrl = `/api/documents/${encodedFilename}?download=true`;
    
    // Create temporary link and click it
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = downloadFilename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

/**
 * Close the document modal
 */
function closeModal() {
    modalEl.style.display = 'none';
    documentFrameEl.src = '';
    currentFilename = null;
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Escape string for use in JavaScript
 */
function escapeJsString(str) {
    if (!str) return '';
    return str.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '\\"');
}

// Close modal on escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modalEl.style.display === 'flex') {
        closeModal();
    }
});

// Close modal on backdrop click
modalEl.addEventListener('click', (e) => {
    if (e.target === modalEl) {
        closeModal();
    }
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);
