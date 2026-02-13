## Getting Started

Create a virtual environment using Python 3.12:

    py -3.12 -m venv .venv

Activate the virtual environment:

    .venv\Scripts\activate

Install dependencies:

    pip install --upgrade pip
    pip install -e .

LlamaParse setup (for PDF/XML parsing):

    setx LLAMA_CLOUD_API_KEY "<your_llama_cloud_api_key>"

LlamaParse processor module:
    rag/ingestion/llama_parse_processor.py (use process_document)


## Clean cache

    pip uninstall -y agent && pip cache purge && pip install -e . 

## Open with Cluade haiku

    claude --model claude-haiku-4-5



1. Launch the Server

python run_web.py

2. Open in Browser

Navigate to: http://localhost:8000