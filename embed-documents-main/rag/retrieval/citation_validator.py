"""Citation validation and structured answer utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass
class ValidationReport:
    """Validation report for citations."""

    total_sentences: int
    cited_sentences: int
    missing_citations: list[str]
    coverage: float


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (simple heuristic)."""
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def validate_citations(answer: str) -> ValidationReport:
    """Validate that each sentence contains at least one citation."""
    sentences = split_sentences(answer)
    missing = []
    cited = 0

    for sentence in sentences:
        if re.search(r"\[C\d+\]", sentence):
            cited += 1
        else:
            missing.append(sentence)

    coverage = (cited / len(sentences)) if sentences else 0.0
    return ValidationReport(
        total_sentences=len(sentences),
        cited_sentences=cited,
        missing_citations=missing,
        coverage=coverage,
    )


def build_structured_answer(answer: str) -> dict:
    """Build a structured answer with claim-level citations."""
    claims = []
    for sentence in split_sentences(answer):
        citations = re.findall(r"\[(C\d+)\]", sentence)
        clean = re.sub(r"\s*\[(C\d+)\]", "", sentence).strip()
        if clean:
            claims.append({"claim": clean, "citations": citations})

    return {"claims": claims, "raw": answer}
