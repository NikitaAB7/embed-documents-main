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

    def __init__(self, threshold: float = 0.4) -> None:  # Lowered from 0.6
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
            # Financial/business signals
            "revenue",
            "profit",
            "earnings",
            "quarter",
            "fy",
            "q1",
            "q2",
            "q3",
            "q4",
            "margins",
            "guidance",
            "outlook",
            "growth",
            "capex",
            "dividend",
            "debt",
            "ebitda",
            "eps",
            "net income",
            "sales",
            "operating",
            "financial",
            "performance",
            "results",
            "highlights",
        ]

        generic_signals = [
            "define",
            "what is a",  # More specific - "what is a stock" vs "what is TCS revenue"
            "who is",
            "explain the concept",
            "difference between",
        ]

        rag_score = 0.0
        for phrase in rag_signals:
            if phrase in q:
                rag_score += 0.2

        if re.search(r"\b(fincode|ticker|symbol)\b", q):
            rag_score += 0.3
        
        # Company name patterns (3+ letter uppercase or mixed case words)
        if re.search(r"\b[A-Z][A-Za-z]{2,}\b", query):  # Use original case
            rag_score += 0.15

        if len(q.split()) >= 10:
            rag_score += 0.1
        
        # Short financial questions should still use RAG
        if len(q.split()) >= 4:
            rag_score += 0.1

        generic_score = 0.0
        for phrase in generic_signals:
            if q.startswith(phrase):
                generic_score += 0.2

        confidence = min(max(rag_score - generic_score + 0.5, 0.0), 1.0)
        use_rag = confidence >= self.threshold

        reason = "rag signals" if use_rag else "generic query"
        return RouteDecision(use_rag=use_rag, confidence=confidence, reason=reason)
