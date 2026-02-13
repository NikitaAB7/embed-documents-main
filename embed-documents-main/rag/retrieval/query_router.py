"""Query router to decide whether RAG is required."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RouteDecision:
    """Routing decision for a query."""

    use_rag: bool
    confidence: float
    reason: str


class QueryRouter:
    """Rule-based query router with confidence threshold."""

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def should_use_rag(self, query: str) -> RouteDecision:
        """Decide whether RAG is required for a query."""
        q = query.strip().lower()

        if not q:
            return RouteDecision(use_rag=False, confidence=0.0, reason="empty query")

        rag_signals = [
            "according to",
            "in the document",
            "in the report",
            "pdf",
            "filing",
            "prospectus",
            "annual report",
            "concall",
            "investor presentation",
            "presentation",
            "what does the document say",
        ]

        generic_signals = [
            "define",
            "what is",
            "who is",
            "explain",
            "summarize",
            "difference between",
        ]

        rag_score = 0.0
        for phrase in rag_signals:
            if phrase in q:
                rag_score += 0.2

        if re.search(r"\b(fincode|ticker|symbol)\b", q):
            rag_score += 0.3

        if len(q.split()) >= 10:
            rag_score += 0.1

        generic_score = 0.0
        for phrase in generic_signals:
            if q.startswith(phrase):
                generic_score += 0.2

        confidence = min(max(rag_score - generic_score + 0.5, 0.0), 1.0)
        use_rag = confidence >= self.threshold

        reason = "rag signals" if use_rag else "generic query"
        return RouteDecision(use_rag=use_rag, confidence=confidence, reason=reason)
