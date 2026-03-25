"""Observability module for LangSmith tracing and evaluation.

This module provides:
- LangSmith tracing configuration
- Custom trace decorators for RAG components
- Evaluation datasets and evaluators
- LangSmith native evaluation (results appear in UI)
"""

from rag.observability.tracing import (
    configure_langsmith,
    get_tracer,
    trace_rag_pipeline,
    trace_retrieval,
    trace_synthesis,
    TracingConfig,
)
from rag.observability.evaluators import (
    RAGEvaluator,
    AnswerRelevanceEvaluator,
    ContextRelevanceEvaluator,
    FaithfulnessEvaluator,
    run_evaluation,
)
from rag.observability.langsmith_evaluators import (
    create_langsmith_dataset,
    load_questions_to_langsmith,
    run_langsmith_evaluation,
    run_langsmith_evaluation_async,
    EVALUATOR_MAP,
)

__all__ = [
    # Tracing
    "configure_langsmith",
    "get_tracer",
    "trace_rag_pipeline",
    "trace_retrieval",
    "trace_synthesis",
    "TracingConfig",
    # Custom Evaluation
    "RAGEvaluator",
    "AnswerRelevanceEvaluator",
    "ContextRelevanceEvaluator",
    "FaithfulnessEvaluator",
    "run_evaluation",
    # LangSmith Native Evaluation
    "create_langsmith_dataset",
    "load_questions_to_langsmith",
    "run_langsmith_evaluation",
    "run_langsmith_evaluation_async",
    "EVALUATOR_MAP",
]
