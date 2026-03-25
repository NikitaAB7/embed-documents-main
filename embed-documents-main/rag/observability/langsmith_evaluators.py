"""LangSmith native evaluators for RAG evaluation.

This module integrates with LangSmith's evaluation framework so results
appear directly in the LangSmith UI with full tracing and comparison.

Usage:
    from rag.observability.langsmith_evaluators import (
        run_langsmith_evaluation,
        create_langsmith_dataset,
    )
    
    # Create dataset in LangSmith
    dataset_name = await create_langsmith_dataset(questions, "my-eval-dataset")
    
    # Run evaluation (results appear in LangSmith UI)
    results = await run_langsmith_evaluation(
        dataset_name=dataset_name,
        experiment_prefix="rag-v1",
    )
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Optional, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _get_langsmith_client():
    """Get LangSmith client, ensuring API key is set."""
    try:
        from langsmith import Client
        return Client()
    except ImportError:
        raise ImportError("langsmith not installed. Run: pip install langsmith")


# -----------------------------------------------------------------------------
# Dataset Management
# -----------------------------------------------------------------------------

def create_langsmith_dataset(
    examples: list[dict],
    dataset_name: str,
    description: str = "",
) -> str:
    """Create or update a LangSmith dataset.
    
    Args:
        examples: List of dicts with keys: question, answer (optional), contexts (optional)
        dataset_name: Name for the dataset in LangSmith
        description: Dataset description
        
    Returns:
        Dataset name (for use with evaluate())
    """
    client = _get_langsmith_client()
    
    # Check if dataset exists
    try:
        existing = client.read_dataset(dataset_name=dataset_name)
        logger.info(f"Dataset '{dataset_name}' already exists, updating...")
        # Delete and recreate to update
        client.delete_dataset(dataset_id=existing.id)
    except Exception:
        pass  # Dataset doesn't exist
    
    # Create dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=description or f"RAG evaluation dataset: {dataset_name}",
    )
    
    # Add examples
    for ex in examples:
        inputs = {"question": ex.get("question", ex.get("query", ""))}
        
        # LangSmith requires outputs for experiments to display properly
        # Use reference_answer if available, otherwise use a placeholder
        outputs = {
            "reference_answer": ex.get("reference_answer", ex.get("answer", ""))
        }
        
        metadata = ex.get("metadata", {})
        if ex.get("contexts"):
            metadata["contexts"] = ex.get("contexts")
        
        client.create_example(
            inputs=inputs,
            outputs=outputs,  # Always provide outputs
            metadata=metadata if metadata else None,
            dataset_id=dataset.id,
        )
    
    logger.info(f"Created LangSmith dataset '{dataset_name}' with {len(examples)} examples")
    return dataset_name


def load_questions_to_langsmith(
    questions_file: str,
    dataset_name: Optional[str] = None,
) -> str:
    """Load questions from JSON file and create LangSmith dataset.
    
    Args:
        questions_file: Path to JSON file with questions
        dataset_name: Optional name (defaults to filename)
        
    Returns:
        Dataset name in LangSmith
    """
    from pathlib import Path
    
    with open(questions_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = data if isinstance(data, list) else data.get("questions", [])
    
    examples = []
    for q in questions:
        if isinstance(q, str):
            examples.append({"question": q})
        else:
            examples.append({
                "question": q.get("question", q.get("query", "")),
                "reference_answer": q.get("answer", q.get("reference_answer")),
                "metadata": q.get("metadata", {}),
            })
    
    name = dataset_name or Path(questions_file).stem
    return create_langsmith_dataset(examples, name)


# -----------------------------------------------------------------------------
# Custom Evaluators (LangSmith format)
# -----------------------------------------------------------------------------

def answer_relevance_evaluator(run, example) -> dict:
    """Evaluate if the answer addresses the question.
    
    This is a LangSmith evaluator function that returns a score.
    """
    from langchain_openai import ChatOpenAI
    
    question = example.inputs.get("question", "")
    answer = run.outputs.get("answer", run.outputs.get("output", ""))
    
    if not answer:
        return {"key": "answer_relevance", "score": 0.0, "comment": "No answer provided"}
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""Evaluate if the answer addresses the question. 
Rate from 0.0 (completely irrelevant) to 1.0 (perfectly addresses the question).

Question: {question}

Answer: {answer}

Respond with ONLY a JSON object:
{{"score": <float 0-1>, "reasoning": "<brief explanation>"}}"""
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Parse JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        result = json.loads(content)
        reasoning = result.get("reasoning", "")
        if isinstance(reasoning, list):
            reasoning = "; ".join(str(r) for r in reasoning)
        return {
            "key": "answer_relevance",
            "score": float(result.get("score", 0.5)),
            "comment": str(reasoning),
        }
    except Exception as e:
        return {"key": "answer_relevance", "score": 0.5, "comment": f"Error: {e}"}


def context_relevance_evaluator(run, example) -> dict:
    """Evaluate if retrieved contexts are relevant to the query."""
    from langchain_openai import ChatOpenAI
    
    question = example.inputs.get("question", "")
    contexts = run.outputs.get("contexts", run.outputs.get("sources", []))
    
    if not contexts:
        return {"key": "context_relevance", "score": 0.0, "comment": "No contexts retrieved"}
    
    if isinstance(contexts, list):
        contexts_text = "\n\n".join(
            f"[{i+1}] {c[:500]}..." if len(str(c)) > 500 else f"[{i+1}] {c}"
            for i, c in enumerate(contexts)
        )
    else:
        contexts_text = str(contexts)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""Evaluate if the retrieved contexts are relevant to answering the question.
Rate from 0.0 (no relevant contexts) to 1.0 (all contexts highly relevant).

Question: {question}

Retrieved Contexts:
{contexts_text}

Respond with ONLY a JSON object:
{{"score": <float 0-1>, "reasoning": "<brief explanation>"}}"""
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        result = json.loads(content)
        reasoning = result.get("reasoning", "")
        if isinstance(reasoning, list):
            reasoning = "; ".join(str(r) for r in reasoning)
        return {
            "key": "context_relevance",
            "score": float(result.get("score", 0.5)),
            "comment": str(reasoning),
        }
    except Exception as e:
        return {"key": "context_relevance", "score": 0.5, "comment": f"Error: {e}"}


def faithfulness_evaluator(run, example) -> dict:
    """Evaluate if the answer is grounded in the provided contexts."""
    from langchain_openai import ChatOpenAI
    
    question = example.inputs.get("question", "")
    answer = run.outputs.get("answer", run.outputs.get("output", ""))
    contexts = run.outputs.get("contexts", run.outputs.get("sources", []))
    
    if not contexts:
        return {"key": "faithfulness", "score": 0.0, "comment": "No contexts to verify against"}
    
    if not answer:
        return {"key": "faithfulness", "score": 0.0, "comment": "No answer provided"}
    
    if isinstance(contexts, list):
        contexts_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    else:
        contexts_text = str(contexts)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""Evaluate if the answer is faithful (grounded in) the provided contexts.
A faithful answer only contains claims that can be verified from the contexts.
Rate from 0.0 (many hallucinations) to 1.0 (fully grounded).

Question: {question}

Contexts:
{contexts_text}

Answer: {answer}

Respond with ONLY a JSON object with two keys - score (float) and reasoning (string):
{{"score": 0.8, "reasoning": "Most claims are supported. The claim about X is not found in the context."}}"""
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        result = json.loads(content)
        reasoning = result.get("reasoning", "")
        if isinstance(reasoning, list):
            reasoning = "; ".join(str(r) for r in reasoning)
        return {
            "key": "faithfulness",
            "score": float(result.get("score", 0.5)),
            "comment": str(reasoning),
        }
    except Exception as e:
        return {"key": "faithfulness", "score": 0.5, "comment": f"Error: {e}"}


def correctness_evaluator(run, example) -> dict:
    """Evaluate if the answer matches the reference answer."""
    from langchain_openai import ChatOpenAI
    
    question = example.inputs.get("question", "")
    answer = run.outputs.get("answer", run.outputs.get("output", ""))
    reference = example.outputs.get("reference_answer", "") if example.outputs else ""
    
    if not reference:
        return {"key": "correctness", "score": 0.5, "comment": "No reference answer provided"}
    
    if not answer:
        return {"key": "correctness", "score": 0.0, "comment": "No answer provided"}
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""Compare the generated answer to the reference answer.
Rate from 0.0 (completely wrong) to 1.0 (semantically equivalent).

Question: {question}

Generated Answer: {answer}

Reference Answer: {reference}

Respond with ONLY a JSON object:
{{"score": <float 0-1>, "reasoning": "<explain key differences or similarities>"}}"""
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        result = json.loads(content)
        reasoning = result.get("reasoning", "")
        if isinstance(reasoning, list):
            reasoning = "; ".join(str(r) for r in reasoning)
        return {
            "key": "correctness",
            "score": float(result.get("score", 0.5)),
            "comment": str(reasoning),
        }
    except Exception as e:
        return {"key": "correctness", "score": 0.5, "comment": f"Error: {e}"}


# Map of evaluator names to functions
EVALUATOR_MAP = {
    "answer_relevance": answer_relevance_evaluator,
    "relevance": answer_relevance_evaluator,
    "context_relevance": context_relevance_evaluator,
    "context": context_relevance_evaluator,
    "faithfulness": faithfulness_evaluator,
    "correctness": correctness_evaluator,
}


# -----------------------------------------------------------------------------
# RAG Pipeline Target Function
# -----------------------------------------------------------------------------

def create_rag_target():
    """Create a target function for LangSmith evaluation.
    
    This wraps the RAG pipeline to produce outputs in the expected format.
    """
    import asyncio
    from rag.retrieval.pipeline import RetrievalPipeline
    from rag.retrieval.answer_synthesizer import AnswerSynthesizer
    
    pipeline = None
    synthesizer = None
    
    def rag_target(inputs: dict) -> dict:
        """Target function that runs the RAG pipeline."""
        nonlocal pipeline, synthesizer
        
        # Lazy initialization
        if pipeline is None:
            pipeline = RetrievalPipeline()
            synthesizer = AnswerSynthesizer()
        
        question = inputs.get("question", "")
        
        async def _run():
            result = await pipeline.retrieve(question)
            contexts = [doc.page_content for doc in result.documents]
            answer_result = await synthesizer.synthesize(question, result.documents)
            return {
                "answer": answer_result.answer,
                "contexts": contexts,
                "sources": [
                    doc.metadata.get("source", "")
                    for doc in result.documents
                ],
            }
        
        # Use new event loop to avoid conflicts
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_run())
        finally:
            loop.close()
    
    return rag_target


def create_async_rag_target():
    """Create an async target function for LangSmith evaluation."""
    from rag.retrieval.pipeline import RetrievalPipeline
    from rag.retrieval.answer_synthesizer import AnswerSynthesizer
    
    pipeline = None
    synthesizer = None
    
    async def rag_target(inputs: dict) -> dict:
        """Async target function that runs the RAG pipeline."""
        nonlocal pipeline, synthesizer
        
        # Lazy initialization
        if pipeline is None:
            pipeline = RetrievalPipeline()
            synthesizer = AnswerSynthesizer()
        
        question = inputs.get("question", "")
        
        result = await pipeline.retrieve(question)
        contexts = [doc.page_content for doc in result.documents]
        answer_result = await synthesizer.synthesize(question, result.documents)
        
        return {
            "answer": answer_result.answer,
            "contexts": contexts,
            "sources": [
                doc.metadata.get("source", "")
                for doc in result.documents
            ],
        }
    
    return rag_target


# -----------------------------------------------------------------------------
# Main Evaluation Function
# -----------------------------------------------------------------------------

def run_langsmith_evaluation(
    dataset_name: str,
    experiment_prefix: str = "rag-eval",
    evaluators: Optional[list[str]] = None,
    max_concurrency: int = 5,
) -> dict:
    """Run evaluation using LangSmith's evaluate() function.
    
    Results will appear in the LangSmith UI under Datasets > [dataset_name].
    
    Args:
        dataset_name: Name of LangSmith dataset to evaluate against
        experiment_prefix: Prefix for the experiment name
        evaluators: List of evaluator names to run (default: all)
        max_concurrency: Max concurrent evaluations
        
    Returns:
        Evaluation results summary
    """
    from langsmith import evaluate
    
    # Get evaluator functions
    if evaluators is None:
        evaluators = ["answer_relevance", "context_relevance", "faithfulness"]
    
    eval_functions = []
    for name in evaluators:
        if name in EVALUATOR_MAP:
            eval_functions.append(EVALUATOR_MAP[name])
        else:
            logger.warning(f"Unknown evaluator: {name}")
    
    if not eval_functions:
        raise ValueError("No valid evaluators specified")
    
    # Create target function
    target = create_rag_target()
    
    # Run evaluation
    logger.info(f"Running LangSmith evaluation on dataset '{dataset_name}'")
    logger.info(f"Evaluators: {evaluators}")
    logger.info(f"Results will appear in LangSmith UI")
    
    results = evaluate(
        target,
        data=dataset_name,
        evaluators=eval_functions,
        experiment_prefix=experiment_prefix,
        max_concurrency=max_concurrency,
    )
    
    # Consume the results iterator to actually run the evaluation
    # and collect summary statistics
    all_results = list(results)
    
    # Aggregate scores by evaluator
    scores_by_evaluator = {}
    for result in all_results:
        if hasattr(result, 'evaluation_results') and result.evaluation_results:
            for eval_result in result.evaluation_results.get('results', []):
                key = eval_result.get('key', 'unknown')
                score = eval_result.get('score', 0)
                if key not in scores_by_evaluator:
                    scores_by_evaluator[key] = []
                scores_by_evaluator[key].append(score)
    
    # Calculate summary
    summary = {
        "dataset": dataset_name,
        "experiment_prefix": experiment_prefix,
        "evaluators": evaluators,
        "num_examples": len(all_results),
        "scores": {
            key: {
                "mean": sum(scores) / len(scores) if scores else 0,
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
            }
            for key, scores in scores_by_evaluator.items()
        }
    }
    
    logger.info(f"Evaluation complete! {len(all_results)} examples evaluated.")
    logger.info(f"View results at: https://smith.langchain.com/")
    
    return summary


async def run_langsmith_evaluation_async(
    dataset_name: str,
    experiment_prefix: str = "rag-eval",
    evaluators: Optional[list[str]] = None,
    max_concurrency: int = 5,
) -> dict:
    """Async version of run_langsmith_evaluation."""
    from langsmith import aevaluate
    
    # Get evaluator functions
    if evaluators is None:
        evaluators = ["answer_relevance", "context_relevance", "faithfulness"]
    
    eval_functions = []
    for name in evaluators:
        if name in EVALUATOR_MAP:
            eval_functions.append(EVALUATOR_MAP[name])
        else:
            logger.warning(f"Unknown evaluator: {name}")
    
    if not eval_functions:
        raise ValueError("No valid evaluators specified")
    
    # Create target function
    target = create_async_rag_target()
    
    # Run evaluation
    logger.info(f"Running LangSmith evaluation on dataset '{dataset_name}'")
    logger.info(f"Evaluators: {evaluators}")
    logger.info(f"Results will appear in LangSmith UI")
    
    results = await aevaluate(
        target,
        data=dataset_name,
        evaluators=eval_functions,
        experiment_prefix=experiment_prefix,
        max_concurrency=max_concurrency,
    )
    
    # Consume the async results iterator
    all_results = []
    async for result in results:
        all_results.append(result)
    
    # Aggregate scores by evaluator
    scores_by_evaluator = {}
    for result in all_results:
        if hasattr(result, 'evaluation_results') and result.evaluation_results:
            for eval_result in result.evaluation_results.get('results', []):
                key = eval_result.get('key', 'unknown')
                score = eval_result.get('score', 0)
                if key not in scores_by_evaluator:
                    scores_by_evaluator[key] = []
                scores_by_evaluator[key].append(score)
    
    # Calculate summary
    summary = {
        "dataset": dataset_name,
        "experiment_prefix": experiment_prefix,
        "evaluators": evaluators,
        "num_examples": len(all_results),
        "scores": {
            key: {
                "mean": sum(scores) / len(scores) if scores else 0,
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
            }
            for key, scores in scores_by_evaluator.items()
        }
    }
    
    logger.info(f"Evaluation complete! {len(all_results)} examples evaluated.")
    logger.info(f"View results at: https://smith.langchain.com/")
    
    return summary
