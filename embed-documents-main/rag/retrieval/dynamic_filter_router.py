"""Dynamic LLM-based filter router for query understanding.

This module uses an LLM to analyze user queries and extract metadata filters
dynamically. The extracted filters are then used for Qdrant metadata filtering.

Flow:
    Input query -> LLM call -> Extract filters -> Apply to Qdrant search
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import AsyncOpenAI

from rag.config import rag_config

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFilters:
    """Filters extracted from user query by LLM."""

    fincode: Optional[int] = None
    symbol: Optional[str] = None
    company_name: Optional[str] = None
    category: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""
    raw_llm_response: Optional[dict] = field(default=None, repr=False)

    def to_filter_dict(self) -> dict[str, Any]:
        """Convert to filter dict for hybrid retriever."""
        filters: dict[str, Any] = {}
        if self.fincode is not None:
            filters["fincode"] = self.fincode
        if self.symbol:
            filters["symbol"] = self.symbol
        if self.category:
            filters["category"] = self.category
        if self.date_from:
            filters["date_from"] = self.date_from
        if self.date_to:
            filters["date_to"] = self.date_to
        return filters

    def has_filters(self) -> bool:
        """Check if any filters were extracted."""
        return bool(
            self.fincode is not None
            or self.symbol
            or self.category
            or self.date_from
            or self.date_to
        )


SYSTEM_PROMPT = """You are a financial document query analyzer. Your job is to extract metadata filters from user queries to search a financial document database.

The database contains documents about Indian publicly listed companies with the following metadata:
- **fincode**: Unique numeric company identifier (e.g., 100034)
- **symbol** / **ticker**: Stock exchange symbol (e.g., "INFY", "TCS", "RELIANCE", "HDFCBANK")
- **company_name**: Full company name (e.g., "Infosys Limited", "Tata Consultancy Services")
- **category**: Document type - one of:
  - "annual-report": Annual reports including financial statements
  - "concall": Quarterly earnings call transcripts
  - "investor-presentation": Investor presentations and slides
- **document_date**: Document date in YYYY-MM-DD format

Your task: Analyze the user query and extract any implicit or explicit filters.

Examples:
1. "What was TCS revenue in FY23?" → symbol: "TCS", category: "annual-report" (since asking about revenue implies annual report)
2. "What did the Reliance management say about Jio?" → symbol: "RELIANCE", category: "concall" (management commentary implies concall)
3. "Show me HDFC Bank's investor presentation from 2024" → symbol: "HDFCBANK", category: "investor-presentation", date_from: "2024-01-01", date_to: "2024-12-31"
4. "Annual report analysis of Infosys" → symbol: "INFY", category: "annual-report"
5. "What are the key risks mentioned by ICICI Bank?" → symbol: "ICICIBANK" (no specific category - could be in any document)
6. "Compare Q3 FY24 concall of Bajaj Finance and HDFC Bank" → This implies multiple companies - extract one or leave empty for cross-company search
7. "How is Reliance Retail segment performing?" → symbol: "RELIANCE" (Reliance Retail is a segment of Reliance Industries)

Output only valid JSON with this exact structure:
{
    "symbol": "<STOCK_SYMBOL or null>",
    "company_name": "<FULL_COMPANY_NAME or null>",
    "category": "<annual-report|concall|investor-presentation or null>",
    "date_from": "<YYYY-MM-DD or null>",
    "date_to": "<YYYY-MM-DD or null>",
    "confidence": <0.0 to 1.0>,
    "reasoning": "<brief explanation of extracted filters>"
}

Rules:
1. Use UPPERCASE for symbols (TCS, INFY, RELIANCE, HDFCBANK, etc.)
2. Only set category if there's strong indication in the query
3. For fiscal years (FY23, FY2023), convert to calendar date ranges: FY23 = April 2022 to March 2023
4. For quarters (Q1 FY24), convert appropriately: Q1 FY24 = April-June 2023
5. Set confidence based on how certain you are about the extracted filters
6. If query asks about multiple companies, set symbol to null (cross-company search)
7. Return null for any field you cannot confidently extract"""


class DynamicFilterRouter:
    """LLM-based router that extracts metadata filters from queries."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        symbol_lookup: Optional[dict[str, int]] = None,
    ) -> None:
        """Initialize the dynamic filter router.

        Args:
            model: OpenAI model to use for filter extraction
            temperature: LLM temperature (0 for deterministic)
            symbol_lookup: Optional dict mapping symbols to fincodes
        """
        self.model = model
        self.temperature = temperature
        self.client = AsyncOpenAI()
        self.symbol_lookup = symbol_lookup or {}

    async def extract_filters(self, query: str) -> ExtractedFilters:
        """Extract metadata filters from user query using LLM.

        Args:
            query: User's natural language query

        Returns:
            ExtractedFilters with extracted metadata and confidence
        """
        if not query.strip():
            return ExtractedFilters(confidence=0.0, reasoning="Empty query")

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
            )

            content = response.choices[0].message.content
            if not content:
                return ExtractedFilters(
                    confidence=0.0, reasoning="Empty LLM response"
                )

            parsed = json.loads(content)
            logger.info(f"LLM extracted filters: {parsed}")

            # Build extracted filters
            extracted = ExtractedFilters(
                symbol=parsed.get("symbol"),
                company_name=parsed.get("company_name"),
                category=parsed.get("category"),
                date_from=parsed.get("date_from"),
                date_to=parsed.get("date_to"),
                confidence=float(parsed.get("confidence", 0.5)),
                reasoning=parsed.get("reasoning", ""),
                raw_llm_response=parsed,
            )

            # Try to resolve symbol to fincode if we have lookup data
            if extracted.symbol and self.symbol_lookup:
                fincode = self.symbol_lookup.get(extracted.symbol.upper())
                if fincode:
                    extracted.fincode = fincode
                    logger.info(
                        f"Resolved symbol {extracted.symbol} to fincode {fincode}"
                    )

            return extracted

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return ExtractedFilters(
                confidence=0.0, reasoning=f"JSON parse error: {e}"
            )
        except Exception as e:
            logger.error(f"Error extracting filters: {e}")
            return ExtractedFilters(confidence=0.0, reasoning=f"Error: {e}")

    async def route_query(
        self,
        query: str,
        user_filters: Optional[dict[str, Any]] = None,
    ) -> ExtractedFilters:
        """Route query by extracting filters and merging with user-provided filters.

        User-provided filters take precedence over LLM-extracted filters.

        Args:
            query: User's natural language query
            user_filters: Optional explicit filters from user

        Returns:
            ExtractedFilters with merged filters
        """
        # First extract filters from query using LLM
        extracted = await self.extract_filters(query)

        # Merge with user-provided filters (user filters take precedence)
        if user_filters:
            if user_filters.get("fincode") is not None:
                extracted.fincode = user_filters["fincode"]
            if user_filters.get("symbol"):
                extracted.symbol = user_filters["symbol"]
            if user_filters.get("category"):
                extracted.category = user_filters["category"]
            if user_filters.get("date_from"):
                extracted.date_from = user_filters["date_from"]
            if user_filters.get("date_to"):
                extracted.date_to = user_filters["date_to"]

        return extracted


# Convenience function for quick filter extraction
async def extract_query_filters(
    query: str,
    user_filters: Optional[dict[str, Any]] = None,
    symbol_lookup: Optional[dict[str, int]] = None,
) -> ExtractedFilters:
    """Extract filters from query using LLM.

    Args:
        query: User's natural language query
        user_filters: Optional explicit filters to merge
        symbol_lookup: Optional symbol->fincode mapping

    Returns:
        ExtractedFilters with extracted metadata
    """
    router = DynamicFilterRouter(symbol_lookup=symbol_lookup)
    return await router.route_query(query, user_filters)
