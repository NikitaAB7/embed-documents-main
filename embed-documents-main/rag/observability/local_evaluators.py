"""Local evaluators without LangSmith integration.

Evaluates model outputs using hardcoded prompts for:
- Correctness: Factual accuracy and completeness
- Relevancy: How well answer addresses the question
- Numerical Accuracy: Accuracy of numerical values
"""

import json
import logging
from typing import Optional

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Hardcoded evaluation prompts
CORRECTNESS_PROMPT = """You are an expert evaluator assessing the correctness of a model's response.

<Rubric>
  A correct answer:
  - Provides accurate and complete information
  - Contains no factual errors
  - Addresses all parts of the question
  - Is logically consistent
  - Uses precise and accurate terminology

  Penalize for:
  - Factual errors or inaccuracies
  - Incomplete or partial answers
  - Misleading or ambiguous statements
  - Incorrect terminology
  - Logical inconsistencies
  - Missing key information
</Rubric>

Question: {question}

Reference Answer: {reference}

Model Answer: {answer}

Evaluate the correctness on a scale of 0-10:
- 10: Perfect - all information accurate, complete, and well-explained
- 7-9: Good - mostly accurate with minor issues
- 4-6: Fair - some accuracy but notable errors or incompleteness
- 1-3: Poor - mostly inaccurate or incomplete
- 0: Completely incorrect

Output Format (STRICT JSON):
{{
    "score": <integer 0-10>,
    "reasoning": "<brief explanation>"
}}

Output only valid JSON, no other text."""

RELEVANCY_PROMPT = """You are an expert evaluator assessing the relevancy of a model's response to the question.

Evaluation Guidelines:
1. Does the answer directly address the question asked?
2. Does it cover the main points that were requested?
3. Is there irrelevant information included?
4. Does it answer all parts of a multi-part question?
5. Is the response appropriately focused?

Question: {question}

Model Answer: {answer}

Retrieved Contexts: {contexts}

Evaluate relevancy on a scale of 0-10:
- 10: Perfectly addresses the question, covers all aspects
- 7-9: Mostly relevant with minor tangential content
- 4-6: Partially relevant, some off-topic content
- 1-3: Mostly irrelevant or missing key aspects
- 0: Completely irrelevant

Output Format (STRICT JSON):
{{
    "score": <integer 0-10>,
    "reasoning": "<brief explanation>"
}}

Output only valid JSON, no other text."""

LOGICAL_COHERENCE_PROMPT = """You are an expert evaluator assessing the logical coherence of a model's response.

Evaluation Guidelines:
1. Does the answer follow a clear logical structure?
2. Are the arguments well-reasoned and supported?
3. Are there any logical contradictions or inconsistencies?
4. Does each point flow naturally to the next?
5. Is the conclusion justified by the presented facts?

Question: {question}

Model Answer: {answer}

Evaluate logical coherence on a scale of 0-10:
- 10: Perfectly logical, well-structured, clear flow, no contradictions
- 7-9: Mostly logical with minor gaps in reasoning
- 4-6: Some logical issues or unclear connections between points
- 1-3: Mostly incoherent or confusing structure
- 0: Completely illogical or unintelligible

Output Format (STRICT JSON):
{{
    "score": <integer 0-10>,
    "reasoning": "<brief explanation of logical structure>"
}}

Output only valid JSON, no other text."""

GROUNDEDNESS_PROMPT = """You are an expert evaluator assessing the groundedness of a model's response.

Groundedness measures whether the answer is supported by the provided context rather than making unsupported claims.

Evaluation Guidelines:
1. Are all major claims supported by the retrieved contexts?
2. Are there any claims without supporting evidence?
3. Is the answer based on context or does it rely on generic knowledge?
4. Are citations or references to source material clear?
5. How many unsupported assertions are made?

Question: {question}

Model Answer: {answer}

Retrieved Contexts: {contexts}

Evaluate groundedness on a scale of 0-10:
- 10: All claims fully supported by provided context
- 7-9: Most claims supported, minor unsupported details
- 4-6: Mix of supported and unsupported claims
- 1-3: Mostly unsupported claims with little grounding
- 0: No grounding in provided contexts

Output Format (STRICT JSON):
{{
    "score": <integer 0-10>,
    "reasoning": "<brief explanation of grounding in provided context>"
}}

Output only valid JSON, no other text."""


class LocalEvaluator:
    """Local evaluator using hardcoded prompts without LangSmith."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize evaluator with LLM model."""
        self.llm = ChatOpenAI(model=model, temperature=0)

    async def evaluate_correctness(
        self,
        question: str,
        answer: str,
        reference: Optional[str] = None,
    ) -> dict:
        """Evaluate correctness of answer.

        Args:
            question: Original question
            answer: Model-generated answer
            reference: Reference/ground truth answer (optional)

        Returns:
            Dictionary with score and reasoning
        """
        try:
            prompt = CORRECTNESS_PROMPT.format(
                question=question,
                reference=reference or "Not provided",
                answer=answer,
            )

            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()

            # Parse JSON
            result = self._parse_json(content)
            return {
                "key": "correctness",
                "score": float(result.get("score", 5)) / 10.0,  # Normalize to 0-1
                "reasoning": str(result.get("reasoning", "")),
            }
        except Exception as e:
            logger.error(f"Error evaluating correctness: {e}")
            return {"key": "correctness", "score": 0.5, "reasoning": f"Error: {str(e)}"}

    async def evaluate_relevancy(
        self,
        question: str,
        answer: str,
        contexts: Optional[list[str]] = None,
    ) -> dict:
        """Evaluate relevancy of answer to question.

        Args:
            question: Original question
            answer: Model-generated answer
            contexts: Retrieved context chunks (optional)

        Returns:
            Dictionary with score and reasoning
        """
        try:
            contexts_text = ""
            if contexts:
                contexts_text = "\n".join(
                    f"[{i+1}] {c[:200]}" for i, c in enumerate(contexts[:3])
                )

            prompt = RELEVANCY_PROMPT.format(
                question=question,
                answer=answer,
                contexts=contexts_text or "No contexts provided",
            )

            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()

            result = self._parse_json(content)
            return {
                "key": "relevancy",
                "score": float(result.get("score", 5)) / 10.0,  # Normalize to 0-1
                "reasoning": str(result.get("reasoning", "")),
            }
        except Exception as e:
            logger.error(f"Error evaluating relevancy: {e}")
            return {"key": "relevancy", "score": 0.5, "reasoning": f"Error: {str(e)}"}

    async def evaluate_logical_coherence(
        self,
        question: str,
        answer: str,
    ) -> dict:
        """Evaluate logical coherence of answer.

        Args:
            question: Original question
            answer: Model-generated answer

        Returns:
            Dictionary with score and reasoning
        """
        try:
            prompt = LOGICAL_COHERENCE_PROMPT.format(
                question=question,
                answer=answer,
            )

            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()

            result = self._parse_json(content)
            return {
                "key": "logical_coherence",
                "score": float(result.get("score", 5)) / 10.0,  # Normalize to 0-1
                "reasoning": str(result.get("reasoning", "")),
            }
        except Exception as e:
            logger.error(f"Error evaluating logical coherence: {e}")
            return {"key": "logical_coherence", "score": 0.5, "reasoning": f"Error: {str(e)}"}

    async def evaluate_groundedness(
        self,
        question: str,
        answer: str,
        contexts: Optional[list[str]] = None,
    ) -> dict:
        """Evaluate groundedness of answer (supporting evidence from contexts).

        Args:
            question: Original question
            answer: Model-generated answer
            contexts: Retrieved context chunks (optional)

        Returns:
            Dictionary with score and reasoning
        """
        try:
            contexts_text = ""
            if contexts:
                contexts_text = "\n".join(
                    f"[{i+1}] {c[:200]}" for i, c in enumerate(contexts[:3])
                )

            prompt = GROUNDEDNESS_PROMPT.format(
                question=question,
                answer=answer,
                contexts=contexts_text or "No contexts provided",
            )

            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()

            result = self._parse_json(content)
            return {
                "key": "groundedness",
                "score": float(result.get("score", 5)) / 10.0,  # Normalize to 0-1
                "reasoning": str(result.get("reasoning", "")),
            }
        except Exception as e:
            logger.error(f"Error evaluating groundedness: {e}")
            return {"key": "groundedness", "score": 0.5, "reasoning": f"Error: {str(e)}"}

    async def evaluate_all(
        self,
        question: str,
        answer: str,
        reference: Optional[str] = None,
        contexts: Optional[list[str]] = None,
    ) -> dict:
        """Run all evaluations.

        Args:
            question: Original question
            answer: Model-generated answer
            reference: Reference answer (optional)
            contexts: Retrieved contexts (optional)

        Returns:
            Dictionary with all evaluation scores
        """
        import asyncio

        # Run evaluations in parallel
        correctness_task = self.evaluate_correctness(question, answer, reference)
        relevancy_task = self.evaluate_relevancy(question, answer, contexts)
        logical_task = self.evaluate_logical_coherence(question, answer)
        groundedness_task = self.evaluate_groundedness(question, answer, contexts)

        correctness, relevancy, logical, groundedness = await asyncio.gather(
            correctness_task, relevancy_task, logical_task, groundedness_task, return_exceptions=True
        )

        # Handle exceptions
        if isinstance(correctness, Exception):
            correctness = {"key": "correctness", "score": 0.5, "reasoning": "Error"}
        if isinstance(relevancy, Exception):
            relevancy = {"key": "relevancy", "score": 0.5, "reasoning": "Error"}
        if isinstance(logical, Exception):
            logical = {"key": "logical_coherence", "score": 0.5, "reasoning": "Error"}
        if isinstance(groundedness, Exception):
            groundedness = {
                "key": "groundedness",
                "score": 0.5,
                "reasoning": "Error",
            }

        return {
            "correctness": correctness["score"],
            "relevancy": relevancy["score"],
            "logical_coherence": logical["score"],
            "groundedness": groundedness["score"],
            "reasoning": {
                "correctness": correctness.get("reasoning", ""),
                "relevancy": relevancy.get("reasoning", ""),
                "logical_coherence": logical.get("reasoning", ""),
                "groundedness": groundedness.get("reasoning", ""),
            },
        }

    @staticmethod
    def _parse_json(content: str) -> dict:
        """Parse JSON from response, handling markdown code blocks."""
        try:
            # Try direct parsing
            return json.loads(content)
        except json.JSONDecodeError:
            # Try removing markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON: {content[:100]}")
                return {"score": 5, "reasoning": "Parsing error"}
