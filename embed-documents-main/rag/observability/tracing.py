"""LangSmith tracing configuration and utilities.

This module provides:
- Environment-based LangSmith configuration
- Custom trace decorators for RAG pipeline components
- Span context management for nested traces
"""

from __future__ import annotations

import functools
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar, ParamSpec

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class TracingConfig:
    """Configuration for LangSmith tracing."""
    
    api_key: Optional[str] = field(default=None)
    project_name: str = field(default="rag-pipeline")
    endpoint: str = field(default="https://api.smith.langchain.com")
    tracing_enabled: bool = field(default=True)
    
    # Sampling rate (0.0 to 1.0) - useful for high-traffic production
    sample_rate: float = field(default=1.0)
    
    # Tags to add to all traces
    default_tags: list[str] = field(default_factory=list)
    
    # Metadata to add to all traces
    default_metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls) -> "TracingConfig":
        """Load configuration from environment variables."""
        return cls(
            api_key=os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY"),
            project_name=os.getenv("LANGCHAIN_PROJECT", "rag-pipeline"),
            endpoint=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
            tracing_enabled=os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true",
            sample_rate=float(os.getenv("LANGSMITH_SAMPLE_RATE", "1.0")),
        )


# Global tracer instance
_tracer: Optional["LangSmithTracer"] = None


class LangSmithTracer:
    """Wrapper for LangSmith tracing operations."""
    
    def __init__(self, config: Optional[TracingConfig] = None):
        self.config = config or TracingConfig.from_env()
        self._client = None
        self._configured = False
        
    def configure(self) -> bool:
        """Configure LangSmith tracing.
        
        Returns:
            True if configuration succeeded, False otherwise.
        """
        if self._configured:
            return True
            
        if not self.config.tracing_enabled:
            logger.info("LangSmith tracing is disabled")
            return False
            
        if not self.config.api_key:
            logger.warning(
                "LANGCHAIN_API_KEY not set. LangSmith tracing disabled. "
                "Get your API key at https://smith.langchain.com"
            )
            return False
            
        # Set environment variables for LangChain integration
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = self.config.api_key
        os.environ["LANGCHAIN_PROJECT"] = self.config.project_name
        os.environ["LANGCHAIN_ENDPOINT"] = self.config.endpoint
        
        try:
            from langsmith import Client
            self._client = Client(
                api_key=self.config.api_key,
                api_url=self.config.endpoint,
            )
            # Test connection
            self._client.list_projects(limit=1)
            self._configured = True
            logger.info(f"LangSmith tracing configured for project: {self.config.project_name}")
            return True
        except ImportError:
            logger.warning("langsmith package not installed. Run: pip install langsmith")
            return False
        except Exception as e:
            logger.warning(f"Failed to configure LangSmith: {e}")
            return False
    
    @property
    def client(self):
        """Get the LangSmith client."""
        if not self._configured:
            self.configure()
        return self._client
    
    @property
    def is_enabled(self) -> bool:
        """Check if tracing is enabled and configured."""
        return self._configured and self.config.tracing_enabled
    
    @contextmanager
    def trace_span(
        self,
        name: str,
        run_type: str = "chain",
        inputs: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ):
        """Create a traced span context.
        
        Args:
            name: Name of the span
            run_type: Type of run (chain, retriever, llm, tool)
            inputs: Input data to log
            tags: Tags for filtering
            metadata: Additional metadata
            
        Yields:
            Span context for adding outputs
        """
        import random
        
        # Check sampling
        if random.random() > self.config.sample_rate:
            yield _NoOpSpan()
            return
            
        if not self.is_enabled:
            yield _NoOpSpan()
            return
            
        try:
            from langsmith import traceable
            from langsmith.run_trees import RunTree
            
            all_tags = list(self.config.default_tags)
            if tags:
                all_tags.extend(tags)
                
            all_metadata = dict(self.config.default_metadata)
            if metadata:
                all_metadata.update(metadata)
            
            run_tree = RunTree(
                name=name,
                run_type=run_type,
                inputs=inputs or {},
                tags=all_tags,
                extra={"metadata": all_metadata},
            )
            
            start_time = time.perf_counter()
            span = _TracedSpan(run_tree, start_time)
            
            # Yield the span - it handles its own lifecycle via __enter__/__exit__
            yield span
                
        except ImportError:
            yield _NoOpSpan()
        except Exception as e:
            logger.debug(f"Tracing error (non-fatal): {e}")
            yield _NoOpSpan()


class _NoOpSpan:
    """No-op span when tracing is disabled."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
    
    def set_output(self, key: str, value: Any):
        pass
        
    def set_outputs(self, outputs: dict):
        pass
        
    def add_metadata(self, key: str, value: Any):
        pass
        
    @property
    def elapsed_ms(self) -> float:
        return 0.0


class _TracedSpan:
    """Active traced span."""
    
    def __init__(self, run_tree, start_time: float):
        self._run_tree = run_tree
        self._start_time = start_time
        self._outputs: dict[str, Any] = {}
        self._closed = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._closed:
            return False
        self._closed = True
        try:
            if exc_type is not None:
                self._run_tree.end(error=str(exc_val))
            else:
                self._run_tree.end(outputs=self._outputs)
            self._run_tree.post()
        except Exception:
            pass
        return False
        
    def set_output(self, key: str, value: Any):
        """Set a single output value."""
        self._outputs[key] = value
        
    def set_outputs(self, outputs: dict):
        """Set multiple outputs."""
        self._outputs.update(outputs)
        
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the span."""
        if "metadata" not in self._run_tree.extra:
            self._run_tree.extra["metadata"] = {}
        self._run_tree.extra["metadata"][key] = value
        
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.perf_counter() - self._start_time) * 1000


def configure_langsmith(config: Optional[TracingConfig] = None) -> bool:
    """Configure LangSmith tracing globally.
    
    Args:
        config: Optional TracingConfig. If not provided, loads from env vars.
        
    Returns:
        True if configuration succeeded.
    """
    global _tracer
    _tracer = LangSmithTracer(config)
    return _tracer.configure()


def get_tracer() -> LangSmithTracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = LangSmithTracer()
        _tracer.configure()
    return _tracer


def trace_rag_pipeline(
    name: str = "rag_pipeline",
    tags: Optional[list[str]] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to trace a RAG pipeline function.
    
    Example:
        @trace_rag_pipeline(name="query_pipeline")
        async def run_query(query: str) -> dict:
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer()
            with tracer.trace_span(
                name=name,
                run_type="chain",
                inputs={"args": str(args), "kwargs": kwargs},
                tags=tags or ["rag", "pipeline"],
            ) as span:
                result = await func(*args, **kwargs)
                span.set_output("result", str(result)[:1000])  # Truncate for safety
                return result
                
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer()
            with tracer.trace_span(
                name=name,
                run_type="chain",
                inputs={"args": str(args), "kwargs": kwargs},
                tags=tags or ["rag", "pipeline"],
            ) as span:
                result = func(*args, **kwargs)
                span.set_output("result", str(result)[:1000])
                return result
                
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
        
    return decorator


def trace_retrieval(
    name: str = "retrieval",
    tags: Optional[list[str]] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to trace retrieval operations.
    
    Example:
        @trace_retrieval(name="hybrid_search")
        async def search(query: str, k: int) -> list[Document]:
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer()
            with tracer.trace_span(
                name=name,
                run_type="retriever",
                inputs={"query": kwargs.get("query", str(args[0]) if args else "")},
                tags=tags or ["retrieval"],
            ) as span:
                result = await func(*args, **kwargs)
                if hasattr(result, "__len__"):
                    span.set_output("num_documents", len(result))
                return result
                
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer()
            with tracer.trace_span(
                name=name,
                run_type="retriever",
                inputs={"query": kwargs.get("query", str(args[0]) if args else "")},
                tags=tags or ["retrieval"],
            ) as span:
                result = func(*args, **kwargs)
                if hasattr(result, "__len__"):
                    span.set_output("num_documents", len(result))
                return result
                
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
        
    return decorator


def trace_synthesis(
    name: str = "answer_synthesis",
    tags: Optional[list[str]] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to trace answer synthesis (LLM calls).
    
    Example:
        @trace_synthesis(name="generate_answer")
        async def synthesize(query: str, docs: list) -> str:
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer()
            with tracer.trace_span(
                name=name,
                run_type="llm",
                inputs={"query": kwargs.get("query", str(args[0]) if args else "")},
                tags=tags or ["synthesis", "llm"],
            ) as span:
                result = await func(*args, **kwargs)
                if hasattr(result, "answer"):
                    span.set_output("answer_length", len(result.answer))
                return result
                
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer()
            with tracer.trace_span(
                name=name,
                run_type="llm",
                inputs={"query": kwargs.get("query", str(args[0]) if args else "")},
                tags=tags or ["synthesis", "llm"],
            ) as span:
                result = func(*args, **kwargs)
                if hasattr(result, "answer"):
                    span.set_output("answer_length", len(result.answer))
                return result
                
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
        
    return decorator
