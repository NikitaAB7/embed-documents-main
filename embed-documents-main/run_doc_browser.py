"""Run the Document Browser web application."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("doc_browser.app:app", host="0.0.0.0", port=8001, reload=True)
