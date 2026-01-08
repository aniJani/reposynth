"""
Adaptive Context Retrieval Module.

This module provides uncertainty-triggered context retrieval for LLM generation.
When the model is uncertain (detected by entropy monitoring), this retriever:
1. Infers what the model is uncertain about (TopicInferrer)
2. Retrieves relevant context from the codebase
3. Returns context to augment generation

Phase 3, Week 6: Topic Inference & Retrieval
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Protocol
import numpy as np

from .topic_inference import TopicInferrer, TopicResult


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    content: str                      # Retrieved context content
    source: str                       # Source of the content (file path, etc.)
    relevance_score: float            # Relevance to query [0, 1]
    token_count: int                  # Approximate token count
    topic: TopicResult                # Topic that triggered retrieval
    retrieval_id: int                 # Unique ID for this retrieval

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(source='{self.source}', "
            f"score={self.relevance_score:.2f}, tokens={self.token_count})"
        )


class BaseRetriever(Protocol):
    """Protocol for base retrievers (existing RepoSynth retrieval)."""

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents."""
        ...


class SimpleRetriever:
    """
    Simple retriever for testing (placeholder for actual retrieval).

    In production, this would be replaced with RepoSynth's actual
    retrieval system (semantic search, BM25, etc.)
    """

    def __init__(self, documents: Optional[Dict[str, str]] = None):
        """
        Initialize with optional document store.

        Args:
            documents: Dict mapping source names to content
        """
        self.documents = documents or {}

    def add_document(self, source: str, content: str):
        """Add a document to the store."""
        self.documents[source] = content

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Simple keyword-based retrieval.

        In production, replace with semantic search.
        """
        if not self.documents:
            return []

        results = []
        query_terms = set(query.lower().split())

        for source, content in self.documents.items():
            content_lower = content.lower()
            # Simple keyword overlap scoring
            score = sum(1 for term in query_terms if term in content_lower)
            score = score / max(len(query_terms), 1)

            if score > 0:
                results.append({
                    'source': source,
                    'content': content,
                    'score': score,
                })

        # Sort by score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]


class AdaptiveContextRetriever:
    """
    Retrieves context when uncertainty is detected during generation.

    This retriever:
    1. Uses TopicInferrer to determine what to search for
    2. Calls the base retriever with the inferred query
    3. Manages retrieval budget (max retrievals per generation)
    4. Tracks retrieval events for analysis

    Example:
        >>> retriever = AdaptiveContextRetriever(
        ...     base_retriever=reposynth_retriever,
        ...     topic_inferrer=TopicInferrer(tokenizer),
        ...     max_retrievals=3
        ... )
        >>> result = retriever.retrieve_on_uncertainty(
        ...     uncertainty_result=monitor.last_result,
        ...     context="In our Flask app, the database is",
        ...     logits=model_logits
        ... )
        >>> if result:
        ...     context += result.content
    """

    def __init__(
        self,
        base_retriever: Optional[BaseRetriever] = None,
        topic_inferrer: Optional[TopicInferrer] = None,
        max_retrievals: int = 3,
        min_confidence: float = 0.3,
        min_relevance: float = 0.2,
        top_k: int = 3,
        tokenizer=None,
    ):
        """
        Initialize adaptive retriever.

        Args:
            base_retriever: Underlying retrieval system
            topic_inferrer: TopicInferrer for query construction
            max_retrievals: Maximum retrievals per generation
            min_confidence: Minimum topic confidence to trigger retrieval
            min_relevance: Minimum relevance score to accept result
            top_k: Number of results to retrieve per query
            tokenizer: Tokenizer for topic inferrer (if not provided separately)
        """
        self.base_retriever = base_retriever or SimpleRetriever()
        self.topic_inferrer = topic_inferrer or TopicInferrer(tokenizer=tokenizer)
        self.max_retrievals = max_retrievals
        self.min_confidence = min_confidence
        self.min_relevance = min_relevance
        self.top_k = top_k

        # State tracking
        self.retrieval_count = 0
        self.retrieval_history: List[RetrievalResult] = []
        self._retrieval_id_counter = 0

    def reset(self):
        """Reset retriever state for new generation."""
        self.retrieval_count = 0
        self.retrieval_history = []
        self.topic_inferrer.reset()

    def retrieve_on_uncertainty(
        self,
        logits: np.ndarray,
        context: str,
        position: int,
        uncertainty_value: Optional[float] = None,
        should_retrieve: bool = True,
    ) -> Optional[RetrievalResult]:
        """
        Retrieve context when uncertainty is detected.

        Args:
            logits: Model output logits
            context: Generated text so far
            position: Token position
            uncertainty_value: Uncertainty score (for logging)
            should_retrieve: Whether retrieval is recommended

        Returns:
            RetrievalResult if successful, None if budget exhausted or no relevant results
        """
        # Check budget
        if self.retrieval_count >= self.max_retrievals:
            return None

        # Check if retrieval is recommended
        if not should_retrieve:
            return None

        # Infer topic
        topic = self.topic_inferrer.infer_topic(
            logits=logits,
            context=context,
            position=position,
            uncertainty_value=uncertainty_value,
        )

        # Check confidence threshold
        if topic.confidence < self.min_confidence:
            return None

        # Retrieve using base retriever
        raw_results = self.base_retriever.retrieve(
            query=topic.query,
            top_k=self.top_k,
        )

        if not raw_results:
            return None

        # Filter by relevance
        best_result = None
        for result in raw_results:
            score = result.get('score', 0)
            if score >= self.min_relevance:
                best_result = result
                break

        if best_result is None:
            return None

        # Create retrieval result
        self._retrieval_id_counter += 1
        retrieval_result = RetrievalResult(
            content=best_result.get('content', ''),
            source=best_result.get('source', 'unknown'),
            relevance_score=best_result.get('score', 0),
            token_count=self._estimate_tokens(best_result.get('content', '')),
            topic=topic,
            retrieval_id=self._retrieval_id_counter,
            metadata={
                'position': position,
                'uncertainty_value': uncertainty_value,
                'query': topic.query,
            }
        )

        # Update state
        self.retrieval_count += 1
        self.retrieval_history.append(retrieval_result)

        return retrieval_result

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Rough estimate: ~4 characters per token on average
        return len(text) // 4

    def get_retrieval_history(self) -> List[RetrievalResult]:
        """Get history of retrievals."""
        return self.retrieval_history.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        if not self.retrieval_history:
            return {
                'total_retrievals': 0,
                'budget_remaining': self.max_retrievals,
            }

        relevance_scores = [r.relevance_score for r in self.retrieval_history]
        confidences = [r.topic.confidence for r in self.retrieval_history]
        token_counts = [r.token_count for r in self.retrieval_history]

        return {
            'total_retrievals': self.retrieval_count,
            'budget_remaining': self.max_retrievals - self.retrieval_count,
            'avg_relevance': np.mean(relevance_scores),
            'avg_confidence': np.mean(confidences),
            'total_tokens_retrieved': sum(token_counts),
            'retrieval_positions': [r.metadata.get('position') for r in self.retrieval_history],
        }


class MultiSourceRetriever(AdaptiveContextRetriever):
    """
    Adaptive retriever that can query multiple sources.

    Sources can include:
    - Local codebase (semantic search)
    - Documentation
    - API references
    - Stack Overflow / external knowledge
    """

    def __init__(
        self,
        retrievers: Dict[str, BaseRetriever],
        topic_inferrer: Optional[TopicInferrer] = None,
        source_priority: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize multi-source retriever.

        Args:
            retrievers: Dict mapping source names to retrievers
            topic_inferrer: TopicInferrer instance
            source_priority: Order to try sources (first = highest priority)
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            base_retriever=None,
            topic_inferrer=topic_inferrer,
            **kwargs
        )
        self.retrievers = retrievers
        self.source_priority = source_priority or list(retrievers.keys())

    def retrieve_on_uncertainty(
        self,
        logits: np.ndarray,
        context: str,
        position: int,
        uncertainty_value: Optional[float] = None,
        should_retrieve: bool = True,
        preferred_source: Optional[str] = None,
    ) -> Optional[RetrievalResult]:
        """
        Retrieve from multiple sources based on priority.

        Args:
            logits: Model output logits
            context: Generated text so far
            position: Token position
            uncertainty_value: Uncertainty score
            should_retrieve: Whether retrieval is recommended
            preferred_source: Optional source to try first

        Returns:
            RetrievalResult from best matching source
        """
        if self.retrieval_count >= self.max_retrievals:
            return None

        if not should_retrieve:
            return None

        # Infer topic
        topic = self.topic_inferrer.infer_topic(
            logits=logits,
            context=context,
            position=position,
            uncertainty_value=uncertainty_value,
        )

        if topic.confidence < self.min_confidence:
            return None

        # Build source order
        sources_to_try = []
        if preferred_source and preferred_source in self.retrievers:
            sources_to_try.append(preferred_source)
        sources_to_try.extend(
            s for s in self.source_priority
            if s not in sources_to_try
        )

        # Try each source
        for source_name in sources_to_try:
            retriever = self.retrievers.get(source_name)
            if retriever is None:
                continue

            raw_results = retriever.retrieve(
                query=topic.query,
                top_k=self.top_k,
            )

            for result in raw_results:
                if result.get('score', 0) >= self.min_relevance:
                    self._retrieval_id_counter += 1
                    retrieval_result = RetrievalResult(
                        content=result.get('content', ''),
                        source=f"{source_name}:{result.get('source', 'unknown')}",
                        relevance_score=result.get('score', 0),
                        token_count=self._estimate_tokens(result.get('content', '')),
                        topic=topic,
                        retrieval_id=self._retrieval_id_counter,
                        metadata={
                            'position': position,
                            'uncertainty_value': uncertainty_value,
                            'query': topic.query,
                            'source_type': source_name,
                        }
                    )

                    self.retrieval_count += 1
                    self.retrieval_history.append(retrieval_result)
                    return retrieval_result

        return None


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_adaptive_retriever(
    base_retriever: Optional[BaseRetriever] = None,
    tokenizer=None,
    max_retrievals: int = 3,
    **kwargs
) -> AdaptiveContextRetriever:
    """
    Factory function to create adaptive retrievers.

    Args:
        base_retriever: Underlying retrieval system
        tokenizer: Tokenizer for topic inference
        max_retrievals: Maximum retrievals per generation
        **kwargs: Additional arguments

    Returns:
        Configured AdaptiveContextRetriever
    """
    topic_inferrer = TopicInferrer(tokenizer=tokenizer)

    return AdaptiveContextRetriever(
        base_retriever=base_retriever,
        topic_inferrer=topic_inferrer,
        max_retrievals=max_retrievals,
        **kwargs
    )
