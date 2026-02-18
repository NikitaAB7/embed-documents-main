import asyncio

from rag.retrieval.pipeline import RetrievalPipeline


async def main() -> None:
    pipeline = RetrievalPipeline()
    result = await pipeline.retrieve(
        "According to the annual report for TCS, what was the segment revenue in FY2024?",
        k=5,
        filters={"symbol": "TCS"},
    )
    print("use_rag", result.route.use_rag, "confidence", result.route.confidence)
    for doc in result.documents:
        meta = doc.metadata or {}
        print(meta.get("ticker"), meta.get("source"), meta.get("chunk_id"))


if __name__ == "__main__":
    asyncio.run(main())
