#!/usr/bin/env python3
"""Quick helper script to inspect DocumentTracker progress."""

import argparse
import asyncio
from datetime import datetime
from typing import Any

from rag.ingestion.document_tracker import DocumentTracker


def _format_ts(ts: str | None) -> str:
    if not ts:
        return "-"
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return ts


def _print_stats(label: str, stats: dict[str, Any]) -> None:
    cost = stats.get("total_cost_usd") or 0.0
    documents = stats.get("total_documents") or 0
    chunks = stats.get("total_chunks") or 0
    tokens = stats.get("total_tokens") or 0
    print(f"\n[{label}]")
    print(f"  Collection     : {stats.get('collection_name', label)}")
    print(f"  Documents      : {documents}")
    print(f"  Chunks         : {chunks}")
    print(f"  Tokens         : {tokens}")
    print(f"  Cost (USD)     : {cost:.4f}")
    print(f"  First Embedded : {_format_ts(stats.get('first_embedded'))}")
    print(f"  Last Embedded  : {_format_ts(stats.get('last_embedded'))}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show SQLite tracker progress for one or more Qdrant collections"
    )
    parser.add_argument(
        "--collection",
        default="compliance_docs",
        help="Specific collection to inspect (default: compliance_docs)",
    )
    parser.add_argument(
        "--all-collections",
        action="store_true",
        help="Print stats for every collection stored in the tracker",
    )

    args = parser.parse_args()

    tracker = DocumentTracker(collection_name=args.collection)
    await tracker.initialize()

    if args.all_collections:
        all_stats = await tracker.get_all_stats()
        if not all_stats:
            print("No embedded collections found yet.")
            return
        for name, stats in all_stats.items():
            _print_stats(name, stats)
        return

    stats = await tracker.get_stats()
    _print_stats(args.collection, stats)


if __name__ == "__main__":
    asyncio.run(main())
