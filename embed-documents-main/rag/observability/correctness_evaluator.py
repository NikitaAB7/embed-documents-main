"""Correctness evaluator for RAG outputs.

Evaluates model outputs based on factual accuracy, completeness, and consistency.
"""

import json
import logging
from typing import Optional

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

CORRECTNESS_RUBRIC = """
You are an expert data labeler evaluating model outputs for correctness. Your task is to assign a score based on the following rubric:

**Correct Answer Criteria:**
- Provides accurate and complete information
- Contains no factual errors
- Addresses all parts of the question
- Is logically consistent
- Uses precise and accurate terminology

**Penalize for:**
- Factual errors or inaccuracies
- Incomplete or partial answers
- Misleading or ambiguous statements
- Incorrect terminology
- Logical inconsistencies
- Missing key information

**Scoring:**
- 1.0: Excellent - No errors, complete, accurate, precise
- 0.8: Good - Minor issues or slight incompleteness
- 0.6: Fair - Some errors or missing information
- 0.4: Poor - Significant errors or major incompleteness
- 0.2: Very Poor - Many errors or severely incomplete
- 0.0: Incorrect - Completely wrong or unusable
"""


async def evaluate_correctness(
    query: str,
    answer: str,
    contexts: Optional[list[str]] = None,
    reference_answer: Optional[str] = None,
) -> dict:
    """Evaluate the correctness of an answer.

    Args:
        query: The original question
        answer: The generated answer to evaluate
        contexts: Optional retrieved context chunks
        reference_answer: Optional reference/ground truth answer

    Returns:
        Dictionary with score, reasoning, and details
    """
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Build evaluation prompt
        contexts_text = ""
        if contexts:
            contexts_text = "\n\n".join(
                f"[Context {i+1}]: {c[:300]}" for i, c in enumerate(contexts[:5])
            )

        reference_text = ""
        if reference_answer:
            reference_text = f"\n\nReference Answer (for comparison):\n{reference_answer}"

        prompt = f"""{CORRECTNESS_RUBRIC}

<input>
Question: {query}

Retrieved Contexts:
{contexts_text if contexts_text else "No contexts provided"}
</input>

<output>
{answer}
</output>
{reference_text}

Instructions:
1. Carefully read the question and answer
2. Check for factual accuracy and completeness
3. Focus on correctness of information rather than style or verbosity
4. Consider how well the answer addresses the question using the available contexts

Provide your evaluation as JSON with the following format:
{{
    "score": <float between 0 and 1>,
    "reasoning": "<brief explanation of the score>",
    "strengths": ["<strength1>", "<strength2>", ...],
    "weaknesses": ["<weakness1>", "<weakness2>", ...]
}}

Only output valid JSON, no other text."""

        response = await llm.ainvoke(prompt)
        content = response.content.strip()

        # Parse JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content)
        return {
            "key": "correctness",
            "score": float(result.get("score", 0.5)),
            "reasoning": str(result.get("reasoning", "")),
            "strengths": result.get("strengths", []),
            "weaknesses": result.get("weaknesses", []),
        }

    except Exception as e:
        logger.error(f"Error evaluating correctness: {e}")
        return {
            "key": "correctness",
            "score": 0.5,
            "reasoning": f"Error during evaluation: {str(e)}",
            "strengths": [],
            "weaknesses": [],
        }


async def evaluate_answer_quality(
    query: str,
    answer: str,
    contexts: Optional[list[str]] = None,
) -> dict:
    """Quick evaluation of answer quality (correctness + relevance).

    Args:
        query: The original question
        answer: The generated answer
        contexts: Retrieved context chunks

    Returns:
        Dictionary with quality metrics
    """
    # Get correctness score
    correctness = await evaluate_correctness(query, answer, contexts)

    # Calculate additional metrics
    answer_length = len(answer.split())
    context_coverage = 0.0

    if contexts and answer:
        # Rough heuristic: check if answer references or uses context info
        context_text = " ".join(contexts[:3])
        common_terms = len(
            set(answer.lower().split()) & set(context_text.lower().split())
        )
        context_coverage = min(1.0, common_terms / max(1, len(answer.split()) / 2))

    return {
        "correctness_score": correctness["score"],
        "correctness_reasoning": correctness["reasoning"],
        "strengths": correctness.get("strengths", []),
        "weaknesses": correctness.get("weaknesses", []),
        "answer_length": answer_length,
        "context_coverage": context_coverage,
    }
