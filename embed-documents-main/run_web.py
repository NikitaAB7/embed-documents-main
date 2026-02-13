#!/usr/bin/env python3
"""Launch script for the Embedded Documents Viewer web application."""

import uvicorn

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Embedded Documents Viewer")
    print("=" * 60)
    print("\n  Starting web server...")
    print("  Open your browser and navigate to:")
    print("\n  \033[1m\033[94mhttp://localhost:8000\033[0m\n")
    print("  Press CTRL+C to stop the server")
    print("=" * 60 + "\n")

    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
