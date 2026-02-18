"""Test script to check LlamaParse JSON output format and bbox availability."""

import asyncio
import json
from pathlib import Path
from llama_parse import LlamaParse

async def test_llamaparse_json():
    """Test what LlamaParse returns in JSON mode."""
    
    # You'll need to provide a test PDF path
    test_pdf = input("Enter path to a test PDF file: ").strip()
    if not Path(test_pdf).exists():
        print(f"File not found: {test_pdf}")
        return
    
    print("\n=== Testing LlamaParse JSON mode ===\n")
    
    parser = LlamaParse(result_type="json")
    docs = await parser.aload_data(test_pdf)
    
    if not docs:
        print("No documents returned!")
        return
    
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print(f"Type: {type(doc)}")
        
        # Get content
        if hasattr(doc, "text"):
            content = doc.text
        elif hasattr(doc, "get_content"):
            content = doc.get_content()
        else:
            content = str(doc)
        
        print(f"Content type: {type(content)}")
        
        # Try to parse as JSON
        try:
            if isinstance(content, str):
                data = json.loads(content)
            else:
                data = content
            
            print(f"JSON keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            
            # Check for pages
            if "pages" in data:
                print(f"Number of pages: {len(data['pages'])}")
                
                for page_idx, page in enumerate(data["pages"][:2]):  # First 2 pages
                    print(f"\n  Page {page_idx + 1} keys: {list(page.keys())}")
                    
                    # Check for items/elements
                    items = page.get("items", page.get("elements", []))
                    print(f"  Number of items: {len(items)}")
                    
                    for item_idx, item in enumerate(items[:3]):  # First 3 items
                        print(f"\n    Item {item_idx + 1} keys: {list(item.keys())}")
                        
                        # Check for bbox
                        bbox = item.get("bbox", item.get("bounding_box", item.get("boundingBox")))
                        if bbox:
                            print(f"    BBOX FOUND: {bbox}")
                        else:
                            print("    No bbox found in this item")
            else:
                print(f"\nFull JSON structure (truncated):")
                print(json.dumps(data, indent=2)[:2000])
                
        except json.JSONDecodeError as e:
            print(f"Not valid JSON: {e}")
            print(f"Raw content (first 500 chars):\n{content[:500]}")

if __name__ == "__main__":
    asyncio.run(test_llamaparse_json())
