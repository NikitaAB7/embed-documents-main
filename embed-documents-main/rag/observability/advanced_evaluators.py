"""Advanced RAG Evaluators for comprehensive evaluation.

This module provides strong evaluators for RAG systems:
- Faithfulness: Are claims grounded in context?
- Embedding Similarity: Semantic distance from reference
- Logical Coherence: Is the answer well-structured and logical?
- Self-Consistency: Does the model give consistent answers?
- Numerical Accuracy: Are numbers/percentages correct?
- Citation Accuracy: Are sources properly cited?
- Ranking Consistency: Is retrieval ranking stable?

Usage:
    from rag.observability.advanced_evaluators import (
        create_faithfulness_evaluator,
        create_embedding_similarity_evaluator,
        ...
    )
"""

import json
import logging
import re
from typing import Any, Optional

from langsmith.schemas import Example, Run

logger = logging.getLogger(__name__)


# =============================================================================
# 1. FAITHFULNESS EVALUATOR
# =============================================================================

def create_faithfulness_evaluator():
    """Evaluate if answer claims are grounded in retrieved context.
    
    This is critical for preventing hallucinations - checks that every
    claim in the answer can be traced back to the retrieved documents.
    """
    def evaluator(run: Run, example: Example) -> dict:
        from langchain_openai import ChatOpenAI
        
        question = example.inputs.get("question", "")
        answer = run.outputs.get("answer", "") if run.outputs else ""
        contexts = run.outputs.get("contexts", []) if run.outputs else []
        
        if not answer:
            return {"key": "faithfulness", "score": 0.0, "comment": "No answer provided"}
        
        if not contexts:
            return {"key": "faithfulness", "score": 0.0, "comment": "No contexts to verify against"}
        
        # Format contexts
        if isinstance(contexts, list):
            contexts_text = "\n\n".join(f"[Context {i+1}]: {c}" for i, c in enumerate(contexts))
        else:
            contexts_text = str(contexts)
        
        prompt = f"""You are a fact-checker evaluating if an answer is faithful to the source documents.

Your task:
1. Extract all factual claims from the answer
2. Check if each claim is supported by the contexts
3. Score based on the proportion of supported claims

Question: {question}

Retrieved Contexts:
{contexts_text}

Answer to evaluate: {answer}

For each claim, determine:
- SUPPORTED: Claim is directly stated or clearly implied in contexts
- PARTIALLY_SUPPORTED: Claim has some basis but adds unsupported details
- UNSUPPORTED: Claim cannot be verified from contexts (hallucination)

Respond with ONLY a JSON object:
{{"score": 0.85, "reasoning": "5/6 claims supported. Claim about X is not in context.", "claims_analysis": "brief summary"}}"""

        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            response = llm.invoke(prompt)
            content = response.content.strip()
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            reasoning = result.get("reasoning", "")
            if isinstance(reasoning, list):
                reasoning = " ".join(str(r) for r in reasoning)
            
            return {
                "key": "faithfulness",
                "score": float(result.get("score", 0.5)),
                "comment": str(reasoning)[:1000],
            }
        except Exception as e:
            logger.error(f"Faithfulness evaluation error: {e}")
            return {"key": "faithfulness", "score": 0.5, "comment": f"Error: {str(e)[:100]}"}
    
    return evaluator


# =============================================================================
# 2. EMBEDDING SIMILARITY EVALUATOR
# =============================================================================

def create_embedding_similarity_evaluator():
    """Evaluate semantic similarity between answer and reference using embeddings.
    
    This provides an objective, non-LLM-judge metric based on vector distance.
    """
    def evaluator(run: Run, example: Example) -> dict:
        from langchain_openai import OpenAIEmbeddings
        import numpy as np
        
        answer = run.outputs.get("answer", "") if run.outputs else ""
        reference = example.outputs.get("reference_answer", "") if example.outputs else ""
        
        if not answer:
            return {"key": "embedding_similarity", "score": 0.0, "comment": "No answer provided"}
        
        if not reference:
            return {"key": "embedding_similarity", "score": 0.5, "comment": "No reference answer"}
        
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            # Get embeddings
            answer_emb = embeddings.embed_query(str(answer))
            reference_emb = embeddings.embed_query(str(reference))
            
            # Cosine similarity
            answer_arr = np.array(answer_emb)
            reference_arr = np.array(reference_emb)
            
            similarity = np.dot(answer_arr, reference_arr) / (
                np.linalg.norm(answer_arr) * np.linalg.norm(reference_arr)
            )
            
            # Convert to 0-1 score (cosine similarity can be -1 to 1)
            score = (similarity + 1) / 2
            
            return {
                "key": "embedding_similarity",
                "score": float(score),
                "comment": f"Cosine similarity: {similarity:.3f}",
            }
        except Exception as e:
            logger.error(f"Embedding similarity error: {e}")
            return {"key": "embedding_similarity", "score": 0.5, "comment": f"Error: {str(e)[:100]}"}
    
    return evaluator


# =============================================================================
# 3. LOGICAL COHERENCE EVALUATOR
# =============================================================================

def create_logical_coherence_evaluator():
    """Evaluate if the answer is logically coherent and well-structured.
    
    Checks for:
    - Internal consistency (no contradictions)
    - Logical flow of ideas
    - Clear structure
    - Appropriate reasoning
    """
    def evaluator(run: Run, example: Example) -> dict:
        from langchain_openai import ChatOpenAI
        
        question = example.inputs.get("question", "")
        answer = run.outputs.get("answer", "") if run.outputs else ""
        
        if not answer:
            return {"key": "logical_coherence", "score": 0.0, "comment": "No answer provided"}
        
        prompt = f"""Evaluate the logical coherence of this answer.

Question: {question}
Answer: {answer}

Evaluate on these criteria:
1. **Internal Consistency**: No contradictions within the answer
2. **Logical Flow**: Ideas connect logically from one to the next
3. **Structure**: Well-organized with clear points
4. **Reasoning**: If conclusions are drawn, they follow from premises
5. **Clarity**: Points are clearly stated without ambiguity

Score:
- 1.0: Perfectly coherent, flawless logic
- 0.8: Minor issues but overall coherent
- 0.6: Some logical gaps or unclear connections
- 0.4: Multiple coherence issues
- 0.2: Largely incoherent
- 0.0: Completely incoherent or contradictory

Respond with ONLY a JSON object:
{{"score": 0.85, "reasoning": "Answer is well-structured with clear logical flow. Minor ambiguity in point 2."}}"""

        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            response = llm.invoke(prompt)
            content = response.content.strip()
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            reasoning = result.get("reasoning", "")
            if isinstance(reasoning, list):
                reasoning = " ".join(str(r) for r in reasoning)
            
            return {
                "key": "logical_coherence",
                "score": float(result.get("score", 0.5)),
                "comment": str(reasoning)[:1000],
            }
        except Exception as e:
            logger.error(f"Logical coherence error: {e}")
            return {"key": "logical_coherence", "score": 0.5, "comment": f"Error: {str(e)[:100]}"}
    
    return evaluator


# =============================================================================
# 4. SELF-CONSISTENCY EVALUATOR
# =============================================================================

def create_self_consistency_evaluator(num_samples: int = 3):
    """Evaluate if the model gives consistent answers across multiple runs.
    
    Runs the same question multiple times and checks if answers agree.
    High consistency = more reliable, low consistency = model is uncertain.
    """
    def evaluator(run: Run, example: Example) -> dict:
        from langchain_openai import ChatOpenAI
        
        question = example.inputs.get("question", "")
        original_answer = run.outputs.get("answer", "") if run.outputs else ""
        contexts = run.outputs.get("contexts", []) if run.outputs else []
        
        if not original_answer:
            return {"key": "self_consistency", "score": 0.0, "comment": "No answer provided"}
        
        if not contexts:
            return {"key": "self_consistency", "score": 0.5, "comment": "No contexts for regeneration"}
        
        # Format contexts
        if isinstance(contexts, list):
            contexts_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        else:
            contexts_text = str(contexts)
        
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # Higher temp for variation
            
            # Generate multiple answers
            answers = [original_answer]
            gen_prompt = f"""Answer this question based on the provided context.

Question: {question}

Context:
{contexts_text}

Provide a concise, factual answer based only on the context."""

            for _ in range(num_samples - 1):
                response = llm.invoke(gen_prompt)
                answers.append(response.content.strip())
            
            # Check consistency between answers
            consistency_prompt = f"""Compare these {len(answers)} answers to the same question and evaluate their consistency.

Question: {question}

Answers:
{chr(10).join(f"Answer {i+1}: {a}" for i, a in enumerate(answers))}

Evaluate:
- Do all answers convey the same core information?
- Are there any contradictions between answers?
- Do they agree on key facts, numbers, and conclusions?

Score:
- 1.0: All answers are essentially identical in meaning
- 0.8: Minor wording differences but same facts
- 0.6: Some variation but core message consistent
- 0.4: Notable disagreements on details
- 0.2: Major contradictions
- 0.0: Completely different answers

Respond with ONLY a JSON object:
{{"score": 0.85, "reasoning": "Answers agree on main points. Minor variation in phrasing."}}"""

            judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            response = judge_llm.invoke(consistency_prompt)
            content = response.content.strip()
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            reasoning = result.get("reasoning", "")
            if isinstance(reasoning, list):
                reasoning = " ".join(str(r) for r in reasoning)
            
            return {
                "key": "self_consistency",
                "score": float(result.get("score", 0.5)),
                "comment": str(reasoning)[:1000],
            }
        except Exception as e:
            logger.error(f"Self-consistency error: {e}")
            return {"key": "self_consistency", "score": 0.5, "comment": f"Error: {str(e)[:100]}"}
    
    return evaluator


# =============================================================================
# 5. NUMERICAL ACCURACY EVALUATOR
# =============================================================================

def create_numerical_accuracy_evaluator():
    """Evaluate accuracy of numbers, percentages, and dates in the answer.
    
    Critical for financial domain - extracts all numerical values and
    compares them against the reference answer.
    """
    def evaluator(run: Run, example: Example) -> dict:
        from langchain_openai import ChatOpenAI
        
        answer = run.outputs.get("answer", "") if run.outputs else ""
        reference = example.outputs.get("reference_answer", "") if example.outputs else ""
        
        if not answer:
            return {"key": "numerical_accuracy", "score": 0.0, "comment": "No answer provided"}
        
        if not reference:
            return {"key": "numerical_accuracy", "score": 0.5, "comment": "No reference for comparison"}
        
        prompt = f"""Extract and compare all numerical values between these two answers.

Reference Answer (ground truth): {reference}

Generated Answer (to evaluate): {answer}

Extract:
- Numbers (integers, decimals)
- Percentages
- Currency amounts
- Dates
- Ratios
- Any quantitative values

Compare each number in the generated answer against the reference:
- CORRECT: Exact match or within acceptable rounding
- CLOSE: Within 5% of correct value
- INCORRECT: Wrong value
- EXTRA: Number not in reference (may be hallucinated)
- MISSING: Important number from reference not included

If no numbers are present in either, score based on whether that's appropriate.

Respond with ONLY a JSON object:
{{"score": 0.9, "reasoning": "4/5 numbers correct. Revenue figure slightly different (within 1%).", "numbers_found": "brief list"}}"""

        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            response = llm.invoke(prompt)
            content = response.content.strip()
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            reasoning = result.get("reasoning", "")
            if isinstance(reasoning, list):
                reasoning = " ".join(str(r) for r in reasoning)
            
            return {
                "key": "numerical_accuracy",
                "score": float(result.get("score", 0.5)),
                "comment": str(reasoning)[:1000],
            }
        except Exception as e:
            logger.error(f"Numerical accuracy error: {e}")
            return {"key": "numerical_accuracy", "score": 0.5, "comment": f"Error: {str(e)[:100]}"}
    
    return evaluator


# =============================================================================
# 6. CITATION ACCURACY EVALUATOR
# =============================================================================

def create_citation_accuracy_evaluator():
    """Evaluate if sources are properly cited and citations are accurate.
    
    Checks:
    - Are claims properly attributed to sources?
    - Do citation references match actual source content?
    - Are all major claims cited?
    """
    def evaluator(run: Run, example: Example) -> dict:
        from langchain_openai import ChatOpenAI
        
        question = example.inputs.get("question", "")
        answer = run.outputs.get("answer", "") if run.outputs else ""
        contexts = run.outputs.get("contexts", []) if run.outputs else []
        sources = run.outputs.get("sources", []) if run.outputs else []
        
        if not answer:
            return {"key": "citation_accuracy", "score": 0.0, "comment": "No answer provided"}
        
        if not contexts and not sources:
            return {"key": "citation_accuracy", "score": 0.5, "comment": "No sources to verify"}
        
        # Format sources
        sources_text = ""
        if contexts:
            sources_text = "\n\n".join(
                f"[Source {i+1}]: {c}" + (f" (from: {sources[i]})" if i < len(sources) else "")
                for i, c in enumerate(contexts)
            )
        
        prompt = f"""Evaluate the citation accuracy of this answer.

Question: {question}

Available Sources:
{sources_text}

Answer: {answer}

Evaluate:
1. **Citation Presence**: Does the answer cite sources where appropriate?
2. **Citation Accuracy**: Do citations point to the correct source?
3. **Claim Coverage**: Are major factual claims properly attributed?
4. **No False Citations**: Are there citations to non-existent sources?

Score:
- 1.0: All claims properly cited with accurate references
- 0.8: Most claims cited, minor attribution issues
- 0.6: Some citations present but incomplete
- 0.4: Few citations, important claims unattributed
- 0.2: Almost no citations despite factual claims
- 0.0: No citations or completely wrong citations

Respond with ONLY a JSON object:
{{"score": 0.75, "reasoning": "Answer cites sources [1] and [2] correctly. Claim about revenue lacks citation."}}"""

        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            response = llm.invoke(prompt)
            content = response.content.strip()
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            reasoning = result.get("reasoning", "")
            if isinstance(reasoning, list):
                reasoning = " ".join(str(r) for r in reasoning)
            
            return {
                "key": "citation_accuracy",
                "score": float(result.get("score", 0.5)),
                "comment": str(reasoning)[:1000],
            }
        except Exception as e:
            logger.error(f"Citation accuracy error: {e}")
            return {"key": "citation_accuracy", "score": 0.5, "comment": f"Error: {str(e)[:100]}"}
    
    return evaluator


# =============================================================================
# 7. RANKING CONSISTENCY EVALUATOR
# =============================================================================

def create_ranking_consistency_evaluator():
    """Evaluate if retrieved document ranking is sensible and consistent.
    
    Checks:
    - Are most relevant docs ranked higher?
    - Is the ranking consistent with document relevance?
    - Does ranking make sense for the query?
    """
    def evaluator(run: Run, example: Example) -> dict:
        from langchain_openai import ChatOpenAI
        
        question = example.inputs.get("question", "")
        contexts = run.outputs.get("contexts", []) if run.outputs else []
        
        if not contexts:
            return {"key": "ranking_consistency", "score": 0.5, "comment": "No contexts to evaluate"}
        
        if len(contexts) < 2:
            return {"key": "ranking_consistency", "score": 1.0, "comment": "Single context, ranking trivial"}
        
        # Format contexts with their positions
        contexts_text = "\n\n".join(
            f"[Rank {i+1}]: {c}" for i, c in enumerate(contexts)
        )
        
        prompt = f"""Evaluate the ranking quality of these retrieved documents.

Question: {question}

Retrieved Documents (in ranked order):
{contexts_text}

Evaluate:
1. **Relevance Order**: Are more relevant documents ranked higher?
2. **Top-K Quality**: Are the top results actually the most relevant?
3. **Ranking Logic**: Does the ordering make sense for this query?
4. **No Major Errors**: Are there highly relevant docs ranked too low?

Score:
- 1.0: Perfect ranking, most relevant at top
- 0.8: Good ranking, minor reordering would improve
- 0.6: Acceptable but some relevant docs ranked too low
- 0.4: Significant ranking issues
- 0.2: Poor ranking, relevant docs buried
- 0.0: Completely wrong ranking

Respond with ONLY a JSON object:
{{"score": 0.8, "reasoning": "Top 2 results highly relevant. Rank 3 could swap with Rank 5 for better ordering.", "suggested_reorder": "optional"}}"""

        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            response = llm.invoke(prompt)
            content = response.content.strip()
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            reasoning = result.get("reasoning", "")
            if isinstance(reasoning, list):
                reasoning = " ".join(str(r) for r in reasoning)
            
            return {
                "key": "ranking_consistency",
                "score": float(result.get("score", 0.5)),
                "comment": str(reasoning)[:1000],
            }
        except Exception as e:
            logger.error(f"Ranking consistency error: {e}")
            return {"key": "ranking_consistency", "score": 0.5, "comment": f"Error: {str(e)[:100]}"}
    
    return evaluator


# =============================================================================
# EVALUATOR REGISTRY
# =============================================================================

ADVANCED_EVALUATOR_MAP = {
    "faithfulness": create_faithfulness_evaluator,
    "embedding_similarity": create_embedding_similarity_evaluator,
    "logical_coherence": create_logical_coherence_evaluator,
    "self_consistency": create_self_consistency_evaluator,
    "numerical_accuracy": create_numerical_accuracy_evaluator,
    "citation_accuracy": create_citation_accuracy_evaluator,
    "ranking_consistency": create_ranking_consistency_evaluator,
}


def get_all_advanced_evaluators():
    """Get all advanced evaluators as a list."""
    return [factory() for factory in ADVANCED_EVALUATOR_MAP.values()]


def get_evaluators_by_name(names: list[str]):
    """Get specific evaluators by name."""
    evaluators = []
    for name in names:
        if name in ADVANCED_EVALUATOR_MAP:
            evaluators.append(ADVANCED_EVALUATOR_MAP[name]())
        else:
            logger.warning(f"Unknown evaluator: {name}")
    return evaluators
