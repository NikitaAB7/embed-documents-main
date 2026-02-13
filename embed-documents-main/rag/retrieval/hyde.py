"""HYDE: Hypothetical Document Embeddings for retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from rag.config import rag_config


@dataclass
class HydeResult:
    """HYDE generation result."""

    hypothetical_answer: str


class HydeGenerator:
    """Generate a hypothetical answer to improve retrieval."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or rag_config.synthesis_model
        self._client = ChatOpenAI(model=self.model, temperature=0.2)

    async def generate(self, query: str) -> HydeResult:
        prompt = (
            "You are a domain expert. Write a concise hypothetical answer that "
            "would appear in internal documents. Focus on factual phrasing and "
            "include key terms and entities that might appear in the source."
        )
        response = await self._client.ainvoke(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ]
        )
        return HydeResult(hypothetical_answer=response.content.strip())
