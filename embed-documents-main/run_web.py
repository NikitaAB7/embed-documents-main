#!/usr/bin/env python3
"""Launch script for the Embedded Documents Viewer web application."""

import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

# Load existing .env file if present
env_file = Path(__file__).parent / ".env"
load_dotenv(env_file)


def prompt_for_env_var(var_name: str, description: str, required: bool = True, default: str = "") -> str:
    """Prompt user for an environment variable if not set."""
    current = os.getenv(var_name, "")
    if current:
        return current
    
    if not required:
        return default
    
    print(f"\n  {description}")
    value = input(f"  Enter {var_name}: ").strip()
    
    if not value and required:
        print(f"  ERROR: {var_name} is required!")
        sys.exit(1)
    
    os.environ[var_name] = value
    return value


def setup_environment():
    """Prompt for required API keys and save to .env file."""
    print("\n" + "=" * 60)
    print("  ENVIRONMENT SETUP")
    print("=" * 60)
    
    env_vars = {}
    
    # Required: OpenAI API Key
    openai_key = prompt_for_env_var(
        "OPENAI_API_KEY",
        "OpenAI API Key (required for embeddings and LLM calls)",
        required=True
    )
    env_vars["OPENAI_API_KEY"] = openai_key
    
    # Required: Qdrant URL
    qdrant_url = prompt_for_env_var(
        "QDRANT_URL",
        "Qdrant URL (e.g., http://localhost:6333 or cloud URL)",
        required=True
    )
    env_vars["QDRANT_URL"] = qdrant_url
    
    # Optional: Qdrant API Key (required for cloud)
    if "cloud.qdrant.io" in qdrant_url:
        qdrant_key = prompt_for_env_var(
            "QDRANT_API_KEY",
            "Qdrant API Key (required for Qdrant Cloud)",
            required=True
        )
        env_vars["QDRANT_API_KEY"] = qdrant_key
    else:
        qdrant_key = os.getenv("QDRANT_API_KEY", "")
        if qdrant_key:
            env_vars["QDRANT_API_KEY"] = qdrant_key
    
    # Optional: LangSmith
    print("\n  LangSmith (optional - for tracing and evaluation)")
    enable_langsmith = input("  Enable LangSmith tracing? (y/N): ").strip().lower() == "y"
    
    if enable_langsmith:
        langsmith_key = prompt_for_env_var(
            "LANGCHAIN_API_KEY",
            "LangSmith API Key (get it at https://smith.langchain.com)",
            required=True
        )
        env_vars["LANGCHAIN_API_KEY"] = langsmith_key
        env_vars["LANGCHAIN_TRACING_V2"] = "true"
        env_vars["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rag-pipeline")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        env_vars["LANGCHAIN_TRACING_V2"] = "false"
    
    # Optional: LlamaParse
    llama_key = os.getenv("LLAMA_CLOUD_API_KEY", "")
    if not llama_key:
        print("\n  LlamaParse (optional - for document parsing)")
        add_llama = input("  Add LlamaParse API Key? (y/N): ").strip().lower() == "y"
        if add_llama:
            llama_key = prompt_for_env_var(
                "LLAMA_CLOUD_API_KEY",
                "LlamaParse API Key (get it at https://cloud.llamaindex.ai)",
                required=True
            )
            env_vars["LLAMA_CLOUD_API_KEY"] = llama_key
    else:
        env_vars["LLAMA_CLOUD_API_KEY"] = llama_key
    
    # Ask to save to .env
    print("\n" + "-" * 60)
    save_env = input("  Save configuration to .env file? (Y/n): ").strip().lower() != "n"
    
    if save_env:
        with open(env_file, "w") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        print(f"  Configuration saved to {env_file}")
    
    print("=" * 60)


def check_required_vars() -> bool:
    """Check if all required environment variables are set."""
    required = ["OPENAI_API_KEY", "QDRANT_URL"]
    missing = [var for var in required if not os.getenv(var)]
    return len(missing) == 0


if __name__ == "__main__":
    # Check if we need to prompt for environment variables
    if not check_required_vars():
        setup_environment()
    
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
