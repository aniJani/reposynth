"""
Learned Topic Inferrer - Drop-in Replacement for TopicInferrer.

This module provides LearnedTopicInferrer, an API-compatible replacement
for the heuristic TopicInferrer that uses the learned cross-attention
module for file matching.

Phase 3, Week 10: Learned Query Pooling Integration
"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass, field

from .topic_inference import TopicResult, TopicInferrer
from .learned_query import LearnedQueryModule, create_learned_query_module


class LearnedTopicInferrer:
    """
    API-compatible replacement for TopicInferrer using learned cross-attention.

    This class wraps the LearnedQueryModule and provides the same interface
    as TopicInferrer, enabling drop-in replacement in the retrieval pipeline.

    The learned module uses:
    - Cross-attention over confused token embeddings
    - Learnable query vectors for pooling
    - Learned bilinear scoring for file matching

    Fallback: If the learned module fails or produces low-confidence results,
    falls back to the heuristic TopicInferrer.

    Example:
        >>> inferrer = LearnedTopicInferrer(tokenizer=tokenizer)
        >>> inferrer.set_available_files(["src/auth.py", "src/api.py"])
        >>> result = inferrer.infer_topic(
        ...     logits=model_logits,
        ...     context="In our Flask app...",
        ...     position=42,
        ...     original_query="How does auth work?"
        ... )
        >>> print(result.matched_files)  # ["src/auth.py"]
    """

    def __init__(
        self,
        tokenizer=None,
        top_k: int = 20,
        context_window: int = 300,
        min_token_length: int = 2,
        code_filter_threshold: float = 0.3,
        available_files: Optional[List[str]] = None,
        embedder=None,
        vector_similarity_threshold: float = 0.5,
        # Learned module config
        embed_dim: int = 384,
        num_query_vectors: int = 4,
        num_heads: int = 4,
        scoring_method: str = "bilinear",
        score_threshold: float = 0.3,
        model_path: Optional[str] = None,
        fallback_confidence_threshold: float = 0.2,
        use_fallback: bool = True,
    ):
        """
        Initialize learned topic inferrer.

        Args:
            tokenizer: HuggingFace tokenizer for decoding tokens
            top_k: Number of top tokens to consider
            context_window: Characters of context to analyze
            min_token_length: Minimum length for token to be considered
            code_filter_threshold: Threshold for filtering non-code tokens
            available_files: List of file paths in the codebase
            embedder: SentenceTransformer model (shared with learned module)
            vector_similarity_threshold: Minimum cosine similarity for matching

            # Learned module parameters
            embed_dim: Embedding dimension (384 for all-MiniLM-L6-v2)
            num_query_vectors: Number of learnable queries (4 for accuracy)
            num_heads: Number of attention heads (4 for accuracy)
            scoring_method: File scoring method (bilinear recommended)
            score_threshold: Minimum score for file matching
            model_path: Optional path to pretrained weights
            fallback_confidence_threshold: Confidence below which to use fallback
            use_fallback: Whether to use heuristic fallback
        """
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.context_window = context_window
        self.min_token_length = min_token_length
        self.code_filter_threshold = code_filter_threshold
        self.fallback_confidence_threshold = fallback_confidence_threshold
        self.use_fallback = use_fallback

        # Initialize learned module
        config = {
            'embed_dim': embed_dim,
            'num_query_vectors': num_query_vectors,
            'num_heads': num_heads,
            'scoring_method': scoring_method,
            'score_threshold': score_threshold,
            'top_k': 3,
        }
        self.learned_module = create_learned_query_module(
            config=config,
            pretrained_path=model_path,
        )

        # Initialize heuristic fallback
        if use_fallback:
            self.fallback_inferrer = TopicInferrer(
                tokenizer=tokenizer,
                top_k=top_k,
                context_window=context_window,
                min_token_length=min_token_length,
                code_filter_threshold=code_filter_threshold,
                available_files=available_files,
                embedder=embedder,
                vector_similarity_threshold=vector_similarity_threshold,
            )
        else:
            self.fallback_inferrer = None

        # Set available files
        if available_files:
            self.set_available_files(available_files)

        # State tracking
        self.last_result: Optional[TopicResult] = None
        self.history: List[TopicResult] = []
        self._use_learned_count = 0
        self._use_fallback_count = 0

    def set_available_files(self, file_paths: List[str]):
        """
        Set available file paths for matching.

        Args:
            file_paths: List of file paths in the codebase
        """
        self.learned_module.set_available_files(file_paths)

        if self.fallback_inferrer:
            self.fallback_inferrer.set_available_files(file_paths)

    def set_embedder(self, embedder):
        """
        Set the embedder for vector-based matching.

        Args:
            embedder: SentenceTransformer model or compatible encoder
        """
        if self.fallback_inferrer:
            self.fallback_inferrer.set_embedder(embedder)

    def load_weights(self, model_path: str):
        """
        Load pretrained weights for the learned module.

        Args:
            model_path: Path to saved weights
        """
        import torch
        state_dict = torch.load(model_path, map_location='cpu')
        self.learned_module.load_state_dict(state_dict)

    def infer_topic(
        self,
        logits: np.ndarray,
        context: str,
        position: int,
        uncertainty_value: Optional[float] = None,
        original_query: str = "",
    ) -> TopicResult:
        """
        Infer topic using learned cross-attention module.

        This method is API-compatible with TopicInferrer.infer_topic().

        Args:
            logits: Model output logits [vocab_size]
            context: Generated text up to this point
            position: Token position in generation
            uncertainty_value: Optional uncertainty score
            original_query: Original user query

        Returns:
            TopicResult with matched files and metadata
        """
        # 1. Extract confused tokens from logits
        confused_tokens, confused_probs = self._extract_confused_tokens(logits)

        if not confused_tokens:
            # No confused tokens - use fallback or return empty result
            if self.use_fallback and self.fallback_inferrer:
                self._use_fallback_count += 1
                return self.fallback_inferrer.infer_topic(
                    logits, context, position, uncertainty_value, original_query
                )
            else:
                return self._empty_result(context, position, uncertainty_value)

        # 2. Run learned module
        try:
            learned_result = self.learned_module(
                confused_tokens=confused_tokens,
                confused_probs=confused_probs,
                original_query=original_query,
                generated_context=context[-self.context_window:],
                return_attention=True,
            )
        except Exception as e:
            # Module failed - use fallback
            if self.use_fallback and self.fallback_inferrer:
                self._use_fallback_count += 1
                return self.fallback_inferrer.infer_topic(
                    logits, context, position, uncertainty_value, original_query
                )
            else:
                return self._empty_result(context, position, uncertainty_value)

        # 3. Check confidence - fallback if too low
        if (
            learned_result.confidence < self.fallback_confidence_threshold
            and self.use_fallback
            and self.fallback_inferrer
        ):
            self._use_fallback_count += 1
            return self.fallback_inferrer.infer_topic(
                logits, context, position, uncertainty_value, original_query
            )

        self._use_learned_count += 1

        # 4. Build TopicResult (API-compatible format)
        result = self._build_topic_result(
            learned_result=learned_result,
            confused_tokens=confused_tokens,
            context=context,
            position=position,
            uncertainty_value=uncertainty_value,
            original_query=original_query,
        )

        self.last_result = result
        self.history.append(result)

        return result

    def _extract_confused_tokens(
        self,
        logits: np.ndarray,
    ) -> tuple:
        """
        Extract confused tokens and their probabilities from logits.

        Args:
            logits: Model output logits [vocab_size]

        Returns:
            Tuple of (tokens, probabilities)
        """
        if self.tokenizer is None:
            return [], []

        # Get top-k indices
        top_indices = np.argsort(logits)[-self.top_k:][::-1]

        # Softmax to get probabilities
        logits_top = logits[top_indices]
        probs = np.exp(logits_top - np.max(logits_top))
        probs = probs / probs.sum()

        # Decode tokens
        tokens = []
        probabilities = []

        for idx, prob in zip(top_indices, probs):
            try:
                token = self.tokenizer.decode([idx])
                clean = token.strip()

                # Filter by length and content
                if len(clean) >= self.min_token_length and self._is_code_relevant(clean):
                    tokens.append(clean)
                    probabilities.append(float(prob))
            except Exception:
                continue

        return tokens, probabilities

    def _is_code_relevant(self, token: str) -> bool:
        """Check if a token is code-relevant."""
        clean = token.lower()

        # Skip common stopwords
        stopwords = {
            'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall',
            'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else',
            'this', 'that', 'these', 'those', 'it', 'its',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        }

        if clean in stopwords:
            return False

        # Accept tokens that look like code
        import re
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', clean):
            return True

        # Accept tokens with dots (method chains)
        if '.' in token:
            return True

        return False

    def _build_topic_result(
        self,
        learned_result,
        confused_tokens: List[str],
        context: str,
        position: int,
        uncertainty_value: Optional[float],
        original_query: str,
    ) -> TopicResult:
        """
        Build TopicResult from learned module output.

        Maintains API compatibility with heuristic TopicInferrer.
        """
        # Build search query from top tokens
        query_parts = []
        if original_query:
            # Extract key terms from query
            words = original_query.split()[:3]
            query_parts.extend(words)

        # Add top confused tokens
        query_parts.extend(confused_tokens[:5])

        # Deduplicate
        seen = set()
        unique_parts = []
        for part in query_parts:
            if part.lower() not in seen:
                seen.add(part.lower())
                unique_parts.append(part)

        query = ' '.join(unique_parts[:8])

        # Infer type from context (simplified)
        inferred_type = 'api'
        if 'import' in context[-50:]:
            inferred_type = 'import'
        elif 'def ' in context[-50:] or 'function' in context[-50:]:
            inferred_type = 'function'

        return TopicResult(
            query=query,
            confidence=learned_result.confidence,
            top_tokens=confused_tokens[:10],
            extracted_identifiers=[],  # Not used in learned approach
            inferred_type=inferred_type,
            context_snippet=context[-100:],
            matched_files=learned_result.matched_files,
            file_match_scores=learned_result.file_scores,
            metadata={
                'position': position,
                'uncertainty_value': uncertainty_value,
                'original_query': original_query,
                'method': 'learned',
                'token_attention': (
                    learned_result.token_attention.tolist()
                    if learned_result.token_attention is not None
                    else None
                ),
            }
        )

    def _empty_result(
        self,
        context: str,
        position: int,
        uncertainty_value: Optional[float],
    ) -> TopicResult:
        """Return empty result when no inference is possible."""
        return TopicResult(
            query="",
            confidence=0.0,
            top_tokens=[],
            extracted_identifiers=[],
            inferred_type='unknown',
            context_snippet=context[-100:],
            matched_files=[],
            file_match_scores={},
            metadata={
                'position': position,
                'uncertainty_value': uncertainty_value,
                'method': 'empty',
            }
        )

    def reset(self):
        """Reset inferrer state."""
        self.last_result = None
        self.history = []

        if self.fallback_inferrer:
            self.fallback_inferrer.reset()

    def get_history(self) -> List[TopicResult]:
        """Get inference history."""
        return self.history.copy()

    def get_statistics(self) -> Dict:
        """Get usage statistics."""
        total = self._use_learned_count + self._use_fallback_count
        return {
            'total_inferences': total,
            'learned_count': self._use_learned_count,
            'fallback_count': self._use_fallback_count,
            'learned_ratio': (
                self._use_learned_count / total if total > 0 else 0.0
            ),
        }


def create_learned_topic_inferrer(
    tokenizer=None,
    available_files: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    use_fallback: bool = True,
    **kwargs,
) -> LearnedTopicInferrer:
    """
    Factory function to create learned topic inferrers.

    Args:
        tokenizer: HuggingFace tokenizer
        available_files: List of file paths
        model_path: Optional path to pretrained weights
        use_fallback: Whether to use heuristic fallback
        **kwargs: Additional configuration

    Returns:
        Configured LearnedTopicInferrer
    """
    return LearnedTopicInferrer(
        tokenizer=tokenizer,
        available_files=available_files,
        model_path=model_path,
        use_fallback=use_fallback,
        **kwargs,
    )
