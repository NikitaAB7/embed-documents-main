"""Answer synthesis with citation formatting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from rag.config import rag_config
from rag.retrieval.citation_validator import build_structured_answer, validate_citations


@dataclass
class AnswerResult:
    """Answer synthesis result."""

    answer: str
    citations: list[dict]
    validation: Optional[dict] = None
    structured_answer: Optional[dict] = None
    faithfulness_score: Optional[float] = None


class AnswerSynthesizer:
    """Synthesize grounded answers with citations."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or rag_config.synthesis_model
        self._client = ChatOpenAI(model=self.model, temperature=rag_config.synthesis_temperature)

    async def synthesize(
        self,
        query: str,
        documents: Iterable[Document],
        max_docs: int = 8,
        strict_citations: bool = False,
        structured_output: bool = False,
        evaluate_faithfulness: bool = False,
    ) -> AnswerResult:
        docs = list(documents)[:max_docs]
        if not docs:
            return AnswerResult(answer="No relevant sources found.", citations=[])

        citation_map = []
        context_lines = []
        for idx, doc in enumerate(docs, start=1):
            meta = doc.metadata or {}
            citation_id = f"C{idx}"
            source = meta.get("source")
            chunk_id = meta.get("chunk_id")
            ref_chunk_id = meta.get("reference_chunk_id")
            page = _extract_page(meta, doc.page_content)
            citation_map.append(
                {
                    "id": citation_id,
                    "source": source,
                    "page": page,
                    "chunk_id": chunk_id,
                    "reference_chunk_id": ref_chunk_id,
                }
            )
            context_lines.append(
                f"[{citation_id}] source={source} page={page} chunk_id={chunk_id}\n{doc.page_content}"
            )

        system_prompt = (
            "You are a grounded assistant. Use ONLY the provided contexts. "
            "Cite every factual claim using the citation ids in square brackets. "
            "If the answer is not in the contexts, say you don't know."
        )
        user_prompt = (
            f"Question: {query}\n\nContexts:\n" + "\n\n".join(context_lines)
        )

        response = await self._client.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        answer = response.content.strip()
        answer = _ensure_citations(answer, [c["id"] for c in citation_map])

        validation = None
        structured = None
        if strict_citations:
            report = validate_citations(answer)
            validation = {
                "total_sentences": report.total_sentences,
                "cited_sentences": report.cited_sentences,
                "missing_citations": report.missing_citations,
                "coverage": report.coverage,
            }

        if structured_output:
            structured = build_structured_answer(answer)

        faithfulness_score = None
        if evaluate_faithfulness:
            faithfulness_score = await self._evaluate_faithfulness(query, context_lines, answer)

        return AnswerResult(
            answer=answer,
            citations=citation_map,
            validation=validation,
            structured_answer=structured,
            faithfulness_score=faithfulness_score,
        )

    async def _evaluate_faithfulness(
        self, query: str, context_lines: list[str], answer: str
    ) -> Optional[float]:
        """LLM-based faithfulness check (0-1)."""
        prompt = (
            "Score the faithfulness of the answer to the provided contexts from 0 to 1. "
            "Return only a number between 0 and 1."
        )
        user_prompt = (
            f"Question: {query}\n\nContexts:\n" + "\n\n".join(context_lines)
            + f"\n\nAnswer:\n{answer}"
        )
        try:
            response = await self._client.ainvoke(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            value = float(response.content.strip())
            if value < 0:
                return 0.0
            if value > 1:
                return 1.0
            return value
        except Exception:
            return None


def _extract_page(metadata: dict, page_content: str) -> str | None:
    page = metadata.get("page") or metadata.get("page_no")
    if page:
        return str(page)

    dl_meta = metadata.get("dl_meta")
    try:
        if dl_meta:
            items = dl_meta.get("doc_items", [])
            if items:
                prov = items[0].get("prov", [])
                if prov:
                    return str(prov[0].get("page_no"))
    except Exception:
        pass

    match = re.search(r"page no:\s*(\d+)", page_content, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def _ensure_citations(answer: str, citation_ids: list[str]) -> str:
    if not citation_ids:
        return answer
    if re.search(r"\[C\d+\]", answer):
        return answer
    return answer + f" [{citation_ids[0]}]"
