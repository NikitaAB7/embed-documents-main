"""Simple document query script for Qdrant."""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI

load_dotenv()

# Initialize clients
qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
openai_client = OpenAI()

# Use the same model as the RAG pipeline
EMBEDDING_MODEL = "text-embedding-3-large"


def get_embedding(text: str) -> list[float]:
    """Get embedding from OpenAI."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def query_documents(query: str, limit: int = 5):
    """Search for relevant documents."""
    # Get query embedding
    query_vector = get_embedding(query)
    
    # Search Qdrant
    results = qdrant.search(
        collection_name="company_files",
        query_vector=query_vector,
        limit=limit
    )
    
    return results


def main():
    # Check collection info first
    info = qdrant.get_collection("company_files")
    print(f"📊 Collection has {info.points_count} vectors")
    print(f"📐 Vector dimensions: {info.config.params.vectors.size}")
    print("-" * 50)
    
    # Interactive query loop
    while True:
        query = input("\n🔍 Enter your query (or 'quit' to exit): ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        print("\nSearching...")
        results = query_documents(query)
        
        print(f"\n📄 Found {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"--- Result {i} (Score: {result.score:.4f}) ---")
            
            # Print metadata if available
            if result.payload:
                for key, value in result.payload.items():
                    if key not in ("text", "content", "page_content"):
                        print(f"  {key}: {value}")
            
            # Print text content (truncated)
            text = result.payload.get("text") or result.payload.get("content") or result.payload.get("page_content", "")
            if text:
                preview = text[:500] + "..." if len(text) > 500 else text
                print(f"\n  Content: {preview}\n")


if __name__ == "__main__":
    main()
