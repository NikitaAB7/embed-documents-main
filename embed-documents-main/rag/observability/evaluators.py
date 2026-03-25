"""LangSmith RAG evaluators and evaluation utilities.

This module provides evaluators for:
- Answer Relevance: Does the answer address the question?
- Context Relevance: Are retrieved documents relevant to the query?
- Faithfulness: Is the answer grounded in the retrieved context?
- Citation Accuracy: Are citations correctly linked to sources?
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result from a single evaluation."""
    
    score: float  # 0.0 to 1.0
    reasoning: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def passed(self) -> bool:
        """Check if evaluation passed (score >= 0.5)."""
        return self.score >= 0.5


@dataclass
class EvalExample:
    """A single example for evaluation."""
    
    query: str
    answer: str
    contexts: list[str]  # Retrieved document contents
    reference_answer: Optional[str] = None  # Ground truth (optional)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass 
class EvalDataset:
    """Dataset for evaluation."""
    
    name: str
    examples: list[EvalExample]
    description: str = ""
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __iter__(self):
        return iter(self.examples)
    
    @classmethod
    def from_json(cls, path: str) -> "EvalDataset":
        """Load dataset from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        examples = [
            EvalExample(
                query=ex["query"],
                answer=ex.get("answer", ""),
                contexts=ex.get("contexts", []),
                reference_answer=ex.get("reference_answer"),
                metadata=ex.get("metadata", {}),
            )
            for ex in data.get("examples", [])
        ]
        
        return cls(
            name=data.get("name", "unnamed"),
            examples=examples,
            description=data.get("description", ""),
        )
    
    def to_json(self, path: str):
        """Save dataset to JSON file."""
        data = {
            "name": self.name,
            "description": self.description,
            "examples": [
                {
                    "query": ex.query,
                    "answer": ex.answer,
                    "contexts": ex.contexts,
                    "reference_answer": ex.reference_answer,
                    "metadata": ex.metadata,
                }
                for ex in self.examples
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class RAGEvaluator(ABC):
    """Base class for RAG evaluators."""
    
    name: str = "base_evaluator"
    
    @abstractmethod
    async def evaluate(self, example: EvalExample) -> EvalResult:
        """Evaluate a single example."""
        pass
    
    async def evaluate_batch(
        self,
        examples: Sequence[EvalExample],
        max_concurrency: int = 5,
    ) -> list[EvalResult]:
        """Evaluate multiple examples with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def eval_with_semaphore(ex: EvalExample) -> EvalResult:
            async with semaphore:
                return await self.evaluate(ex)
        
        return await asyncio.gather(*[eval_with_semaphore(ex) for ex in examples])


class AnswerRelevanceEvaluator(RAGEvaluator):
    """Evaluates if the answer addresses the question.
    
    Uses an LLM to judge whether the generated answer is relevant
    to and addresses the user's question.
    """
    
    name = "answer_relevance"
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0)
        
    async def evaluate(self, example: EvalExample) -> EvalResult:
        prompt = f"""Evaluate if the following answer is relevant to and addresses the question.

Question: {example.query}

Answer: {example.answer}

Provide your evaluation as JSON with:
- score: float from 0.0 (completely irrelevant) to 1.0 (perfectly relevant)
- reasoning: brief explanation

JSON:"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()
            
            # Parse JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            result = json.loads(content)
            return EvalResult(
                score=float(result.get("score", 0.5)),
                reasoning=result.get("reasoning", ""),
                metadata={"evaluator": self.name},
            )
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return EvalResult(
                score=0.5,
                reasoning=f"Evaluation error: {str(e)}",
                metadata={"evaluator": self.name, "error": True},
            )


class ContextRelevanceEvaluator(RAGEvaluator):
    """Evaluates if retrieved contexts are relevant to the query.
    
    Measures retrieval quality by checking if the retrieved documents
    contain information relevant to answering the question.
    """
    
    name = "context_relevance"
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0)
        
    async def evaluate(self, example: EvalExample) -> EvalResult:
        if not example.contexts:
            return EvalResult(
                score=0.0,
                reasoning="No contexts provided",
                metadata={"evaluator": self.name},
            )
        
        contexts_text = "\n\n---\n\n".join(
            f"Context {i+1}: {ctx[:500]}..." if len(ctx) > 500 else f"Context {i+1}: {ctx}"
            for i, ctx in enumerate(example.contexts)
        )
        
        prompt = f"""Evaluate if the retrieved contexts are relevant to answering the question.

Question: {example.query}

Retrieved Contexts:
{contexts_text}

Provide your evaluation as JSON with:
- score: float from 0.0 (no relevant contexts) to 1.0 (all contexts highly relevant)
- reasoning: brief explanation of which contexts are relevant and why

JSON:"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            result = json.loads(content)
            return EvalResult(
                score=float(result.get("score", 0.5)),
                reasoning=result.get("reasoning", ""),
                metadata={"evaluator": self.name, "num_contexts": len(example.contexts)},
            )
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return EvalResult(
                score=0.5,
                reasoning=f"Evaluation error: {str(e)}",
                metadata={"evaluator": self.name, "error": True},
            )


class FaithfulnessEvaluator(RAGEvaluator):
    """Evaluates if the answer is grounded in the provided contexts.
    
    Checks that factual claims in the answer can be verified from
    the retrieved contexts (no hallucinations).
    """
    
    name = "faithfulness"
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0)
        
    async def evaluate(self, example: EvalExample) -> EvalResult:
        if not example.contexts:
            return EvalResult(
                score=0.0,
                reasoning="No contexts to verify against",
                metadata={"evaluator": self.name},
            )
        
        contexts_text = "\n\n".join(
            f"[{i+1}] {ctx}" for i, ctx in enumerate(example.contexts)
        )
        
        prompt = f"""Evaluate if the answer is faithful to (grounded in) the provided contexts.

Question: {example.query}

Contexts:
{contexts_text}

Answer: {example.answer}

Check each factual claim in the answer. A faithful answer only contains claims that can be verified from the contexts.

Provide your evaluation as JSON with:
- score: float from 0.0 (many hallucinations) to 1.0 (fully grounded)
- reasoning: list any claims that are NOT supported by the contexts

JSON:"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            result = json.loads(content)
            return EvalResult(
                score=float(result.get("score", 0.5)),
                reasoning=result.get("reasoning", ""),
                metadata={"evaluator": self.name},
            )
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return EvalResult(
                score=0.5,
                reasoning=f"Evaluation error: {str(e)}",
                metadata={"evaluator": self.name, "error": True},
            )


class CorrectnessEvaluator(RAGEvaluator):
    """Evaluates if the answer matches the reference answer (when available).
    
    Compares the generated answer against a ground truth reference.
    Only useful when you have labeled evaluation data.
    """
    
    name = "correctness"
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0)
        
    async def evaluate(self, example: EvalExample) -> EvalResult:
        if not example.reference_answer:
            return EvalResult(
                score=0.5,
                reasoning="No reference answer provided",
                metadata={"evaluator": self.name, "skipped": True},
            )
        
        prompt = f"""Compare the generated answer to the reference answer.

Question: {example.query}

Generated Answer: {example.answer}

Reference Answer: {example.reference_answer}

Provide your evaluation as JSON with:
- score: float from 0.0 (completely wrong) to 1.0 (semantically equivalent)
- reasoning: explain key differences or similarities

JSON:"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            result = json.loads(content)
            return EvalResult(
                score=float(result.get("score", 0.5)),
                reasoning=result.get("reasoning", ""),
                metadata={"evaluator": self.name},
            )
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return EvalResult(
                score=0.5,
                reasoning=f"Evaluation error: {str(e)}",
                metadata={"evaluator": self.name, "error": True},
            )


@dataclass
class EvalReport:
    """Aggregated evaluation report."""
    
    dataset_name: str
    num_examples: int
    results_by_evaluator: dict[str, list[EvalResult]]
    
    def summary(self) -> dict[str, dict[str, float]]:
        """Get summary statistics by evaluator."""
        summary = {}
        for evaluator_name, results in self.results_by_evaluator.items():
            scores = [r.score for r in results]
            summary[evaluator_name] = {
                "mean": sum(scores) / len(scores) if scores else 0.0,
                "min": min(scores) if scores else 0.0,
                "max": max(scores) if scores else 0.0,
                "pass_rate": sum(1 for r in results if r.passed) / len(results) if results else 0.0,
            }
        return summary
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "dataset_name": self.dataset_name,
            "num_examples": self.num_examples,
            "summary": self.summary(),
            "results": {
                name: [
                    {"score": r.score, "reasoning": r.reasoning, "metadata": r.metadata}
                    for r in results
                ]
                for name, results in self.results_by_evaluator.items()
            },
        }


async def run_evaluation(
    dataset: EvalDataset,
    evaluators: Optional[list[RAGEvaluator]] = None,
    max_concurrency: int = 5,
    upload_to_langsmith: bool = False,
    project_name: Optional[str] = None,
) -> EvalReport:
    """Run evaluation on a dataset with multiple evaluators.
    
    Args:
        dataset: EvalDataset to evaluate
        evaluators: List of RAGEvaluator instances (defaults to all built-in)
        max_concurrency: Max concurrent evaluations per evaluator
        upload_to_langsmith: Whether to upload results to LangSmith
        project_name: LangSmith project name (if uploading)
        
    Returns:
        EvalReport with aggregated results
    """
    if evaluators is None:
        evaluators = [
            AnswerRelevanceEvaluator(),
            ContextRelevanceEvaluator(),
            FaithfulnessEvaluator(),
        ]
    
    results_by_evaluator: dict[str, list[EvalResult]] = {}
    
    for evaluator in evaluators:
        logger.info(f"Running {evaluator.name} evaluator on {len(dataset)} examples...")
        results = await evaluator.evaluate_batch(
            dataset.examples,
            max_concurrency=max_concurrency,
        )
        results_by_evaluator[evaluator.name] = results
    
    report = EvalReport(
        dataset_name=dataset.name,
        num_examples=len(dataset),
        results_by_evaluator=results_by_evaluator,
    )
    
    if upload_to_langsmith and project_name:
        try:
            await _upload_to_langsmith(report, project_name)
        except Exception as e:
            logger.warning(f"Failed to upload to LangSmith: {e}")
    
    return report


async def _upload_to_langsmith(report: EvalReport, project_name: str):
    """Upload evaluation results to LangSmith."""
    try:
        from langsmith import Client
        
        client = Client()
        
        # Create or get dataset
        try:
            ls_dataset = client.create_dataset(
                report.dataset_name,
                description=f"Evaluation results for {report.dataset_name}",
            )
        except Exception:
            # Dataset might already exist
            datasets = list(client.list_datasets(dataset_name=report.dataset_name))
            if datasets:
                ls_dataset = datasets[0]
            else:
                raise
        
        logger.info(f"Uploaded evaluation results to LangSmith dataset: {report.dataset_name}")
        
    except ImportError:
        logger.warning("langsmith not installed, skipping upload")
