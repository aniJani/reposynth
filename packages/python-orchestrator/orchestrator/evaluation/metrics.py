"""
Evaluation Metrics for CCE Research.

Implements 8 metrics for assessing adaptive retrieval quality:
1. answer_correctness - LLM/keyword scoring
2. answer_completeness - semantic similarity
3. hallucination_rate - unsupported claims (FIXED)
4. context_precision - retrieved accuracy
5. context_recall - coverage of ground truth
6. token_efficiency - tokens saved vs baseline
7. spike_precision - trigger accuracy
8. spike_recall - trigger coverage
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
import numpy as np


@dataclass
class EvaluationResult:
    """Result of evaluating a single example."""
    example_id: str
    method: str

    # Answer quality metrics
    answer_correctness: float = 0.0
    answer_completeness: float = 0.0
    hallucination_rate: float = 0.0

    # Context quality metrics
    context_precision: float = 0.0
    context_recall: float = 0.0

    # Efficiency metrics
    token_efficiency: float = 0.0
    tokens_used: int = 0
    baseline_tokens: int = 0

    # Spike detection metrics
    spike_precision: float = 0.0
    spike_recall: float = 0.0

    # Metadata
    generation_time: float = 0.0
    num_retrievals: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_f1_context(self) -> float:
        """Compute F1 score for context retrieval."""
        if self.context_precision + self.context_recall == 0:
            return 0.0
        return 2 * (self.context_precision * self.context_recall) / (
            self.context_precision + self.context_recall
        )

    def get_f1_spike(self) -> float:
        """Compute F1 score for spike detection."""
        if self.spike_precision + self.spike_recall == 0:
            return 0.0
        return 2 * (self.spike_precision * self.spike_recall) / (
            self.spike_precision + self.spike_recall
        )

    def get_composite_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute weighted composite score.
        Default weights prioritize correctness and context quality.
        """
        if weights is None:
            weights = {
                "answer_correctness": 0.25,
                "answer_completeness": 0.15,
                "hallucination_rate": 0.15,  # Inverted: lower is better
                "context_f1": 0.25,
                "token_efficiency": 0.20,
            }

        score = 0.0
        score += weights.get("answer_correctness", 0) * self.answer_correctness
        score += weights.get("answer_completeness", 0) * self.answer_completeness
        score += weights.get("hallucination_rate", 0) * (1 - self.hallucination_rate)
        score += weights.get("context_f1", 0) * self.get_f1_context()
        score += weights.get("token_efficiency", 0) * self.token_efficiency

        return score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "example_id": self.example_id,
            "method": self.method,
            "answer_correctness": self.answer_correctness,
            "answer_completeness": self.answer_completeness,
            "hallucination_rate": self.hallucination_rate,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "context_f1": self.get_f1_context(),
            "token_efficiency": self.token_efficiency,
            "tokens_used": self.tokens_used,
            "spike_precision": self.spike_precision,
            "spike_recall": self.spike_recall,
            "spike_f1": self.get_f1_spike(),
            "composite_score": self.get_composite_score(),
            "generation_time": self.generation_time,
            "num_retrievals": self.num_retrievals,
        }


class EvaluationMetrics:
    """
    Evaluation metrics calculator.

    Uses sentence-transformers for semantic similarity.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize metrics calculator.

        Args:
            embedding_model: SentenceTransformer model name
        """
        self.embedding_model_name = embedding_model
        self._model = None

    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install with: pip install sentence-transformers"
                )
        return self._model

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        return self.model.encode(text, convert_to_numpy=True)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences (claims)."""
        # Simple sentence splitting on . ! ? followed by space or end
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # Filter out very short sentences
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    # =========================================================================
    # Metric 1: Answer Correctness
    # =========================================================================
    def answer_correctness(
        self,
        predicted: str,
        ground_truth: str,
        keywords: Optional[List[str]] = None,
        keyword_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ) -> float:
        """
        Compute answer correctness as weighted combination of keyword and semantic match.

        Args:
            predicted: Generated answer
            ground_truth: Reference answer
            keywords: Important keywords that should appear
            keyword_weight: Weight for keyword matching (default 0.4)
            semantic_weight: Weight for semantic similarity (default 0.6)

        Returns:
            Score between 0 and 1
        """
        # Keyword score
        keyword_score = 0.0
        if keywords:
            predicted_lower = predicted.lower()
            matches = sum(1 for kw in keywords if kw.lower() in predicted_lower)
            keyword_score = matches / len(keywords) if keywords else 0.0

        # Semantic score
        pred_emb = self._get_embedding(predicted)
        truth_emb = self._get_embedding(ground_truth)
        semantic_score = max(0.0, self._cosine_similarity(pred_emb, truth_emb))

        # Weighted combination
        if keywords:
            return keyword_weight * keyword_score + semantic_weight * semantic_score
        else:
            return semantic_score

    # =========================================================================
    # Metric 2: Answer Completeness
    # =========================================================================
    def answer_completeness(self, predicted: str, ground_truth: str) -> float:
        """
        Compute answer completeness as semantic similarity.

        Higher score means the predicted answer covers more of the ground truth.

        Args:
            predicted: Generated answer
            ground_truth: Reference answer

        Returns:
            Score between 0 and 1
        """
        pred_emb = self._get_embedding(predicted)
        truth_emb = self._get_embedding(ground_truth)
        return max(0.0, self._cosine_similarity(pred_emb, truth_emb))

    # =========================================================================
    # Metric 3: Hallucination Rate (FIXED)
    # =========================================================================
    def hallucination_rate(
        self,
        predicted: str,
        context: str,
        keywords: Optional[List[str]] = None,
        similarity_threshold: float = 0.7,
    ) -> float:
        """
        Compute fraction of predicted claims not supported by context.

        A claim is supported if:
        1. It contains at least one keyword from the ground truth, OR
        2. It has semantic similarity > threshold to any context sentence

        Args:
            predicted: Generated answer
            context: Provided context (ground truth answer or retrieved docs)
            keywords: Keywords that indicate supported claims
            similarity_threshold: Threshold for semantic support (default 0.7)

        Returns:
            Fraction of unsupported claims (0 = no hallucination, 1 = all hallucinated)
        """
        # Split predicted into sentences (claims)
        claims = self._split_into_sentences(predicted)
        if not claims:
            return 0.0  # No claims to evaluate

        # Split context into sentences for semantic matching
        context_sentences = self._split_into_sentences(context)
        if not context_sentences:
            context_sentences = [context]  # Use full context if no sentence split

        # Get embeddings for context sentences
        context_embeddings = [self._get_embedding(s) for s in context_sentences]

        # Prepare keywords for matching
        keywords_lower: Set[str] = set()
        if keywords:
            keywords_lower = {kw.lower() for kw in keywords}

        unsupported_count = 0

        for claim in claims:
            claim_lower = claim.lower()
            is_supported = False

            # Check 1: Keyword presence
            if keywords_lower:
                for kw in keywords_lower:
                    if kw in claim_lower:
                        is_supported = True
                        break

            # Check 2: Semantic similarity to context
            if not is_supported and context_embeddings:
                claim_emb = self._get_embedding(claim)
                max_similarity = max(
                    self._cosine_similarity(claim_emb, ctx_emb)
                    for ctx_emb in context_embeddings
                )
                if max_similarity >= similarity_threshold:
                    is_supported = True

            if not is_supported:
                unsupported_count += 1

        return unsupported_count / len(claims)

    # =========================================================================
    # Metric 4: Context Precision
    # =========================================================================
    def context_precision(
        self,
        retrieved_files: List[str],
        ground_truth_files: List[str],
    ) -> float:
        """
        Compute precision of retrieved files.

        Precision = |retrieved ∩ ground_truth| / |retrieved|

        Args:
            retrieved_files: Files retrieved by the system
            ground_truth_files: Files that should have been retrieved

        Returns:
            Precision score between 0 and 1
        """
        if not retrieved_files:
            return 0.0

        # Normalize file paths for comparison
        retrieved_set = {self._normalize_path(f) for f in retrieved_files}
        truth_set = {self._normalize_path(f) for f in ground_truth_files}

        intersection = retrieved_set & truth_set
        return len(intersection) / len(retrieved_set)

    # =========================================================================
    # Metric 5: Context Recall
    # =========================================================================
    def context_recall(
        self,
        retrieved_files: List[str],
        ground_truth_files: List[str],
    ) -> float:
        """
        Compute recall of retrieved files.

        Recall = |retrieved ∩ ground_truth| / |ground_truth|

        Args:
            retrieved_files: Files retrieved by the system
            ground_truth_files: Files that should have been retrieved

        Returns:
            Recall score between 0 and 1
        """
        if not ground_truth_files:
            return 0.0

        # Normalize file paths for comparison
        retrieved_set = {self._normalize_path(f) for f in retrieved_files}
        truth_set = {self._normalize_path(f) for f in ground_truth_files}

        intersection = retrieved_set & truth_set
        return len(intersection) / len(truth_set)

    def _normalize_path(self, path: str) -> str:
        """Normalize file path for comparison."""
        # Remove leading/trailing slashes, convert to forward slashes
        return path.strip().replace("\\", "/").strip("/").lower()

    # =========================================================================
    # Metric 6: Token Efficiency
    # =========================================================================
    def token_efficiency(self, tokens_used: int, baseline_tokens: int) -> float:
        """
        Compute token efficiency relative to baseline.

        Efficiency = 1 - (tokens_used / baseline_tokens)
        Higher is better (using fewer tokens).

        Args:
            tokens_used: Tokens used by adaptive method
            baseline_tokens: Tokens used by full-context baseline

        Returns:
            Efficiency score between 0 and 1
        """
        if baseline_tokens <= 0:
            return 0.0
        if tokens_used >= baseline_tokens:
            return 0.0
        return 1.0 - (tokens_used / baseline_tokens)

    # =========================================================================
    # Metric 7: Spike Precision
    # =========================================================================
    def spike_precision(
        self,
        detected_positions: List[int],
        ground_truth_positions: List[int],
        tolerance: int = 3,
    ) -> float:
        """
        Compute precision of spike detection.

        A detected spike is correct if it's within `tolerance` tokens of a ground truth position.

        Precision = |correct detections| / |detected|

        Args:
            detected_positions: Token positions where spikes were detected
            ground_truth_positions: Token positions where context was actually missing
            tolerance: Window size for matching (default 3 tokens)

        Returns:
            Precision score between 0 and 1
        """
        if not detected_positions:
            return 0.0
        if not ground_truth_positions:
            return 0.0  # No ground truth means we can't evaluate

        correct = 0
        for detected in detected_positions:
            # Check if any ground truth position is within tolerance
            for truth in ground_truth_positions:
                if abs(detected - truth) <= tolerance:
                    correct += 1
                    break

        return correct / len(detected_positions)

    # =========================================================================
    # Metric 8: Spike Recall
    # =========================================================================
    def spike_recall(
        self,
        detected_positions: List[int],
        ground_truth_positions: List[int],
        tolerance: int = 3,
    ) -> float:
        """
        Compute recall of spike detection.

        Recall = |matched ground truths| / |ground truths|

        Args:
            detected_positions: Token positions where spikes were detected
            ground_truth_positions: Token positions where context was actually missing
            tolerance: Window size for matching (default 3 tokens)

        Returns:
            Recall score between 0 and 1
        """
        if not ground_truth_positions:
            return 0.0
        if not detected_positions:
            return 0.0  # No detections means zero recall

        matched = 0
        for truth in ground_truth_positions:
            # Check if any detected position is within tolerance
            for detected in detected_positions:
                if abs(detected - truth) <= tolerance:
                    matched += 1
                    break

        return matched / len(ground_truth_positions)

    # =========================================================================
    # Full Evaluation
    # =========================================================================
    def evaluate(
        self,
        example_id: str,
        generated_answer: str,
        ground_truth_answer: str,
        retrieved_files: List[str],
        ground_truth_files: List[str],
        ground_truth_keywords: Optional[List[str]] = None,
        tokens_used: int = 0,
        baseline_tokens: int = 0,
        generation_time: float = 0.0,
        num_retrievals: int = 0,
        detected_positions: Optional[List[int]] = None,
        ground_truth_positions: Optional[List[int]] = None,
        method: str = "unknown",
    ) -> EvaluationResult:
        """
        Run full evaluation on a single example.

        Args:
            example_id: Identifier for the example
            generated_answer: Answer generated by the system
            ground_truth_answer: Reference answer
            retrieved_files: Files retrieved by the system
            ground_truth_files: Files that should have been retrieved
            ground_truth_keywords: Keywords for correctness/hallucination
            tokens_used: Context tokens used
            baseline_tokens: Baseline token count for efficiency
            generation_time: Time taken to generate
            num_retrievals: Number of retrieval operations
            detected_positions: Positions where spikes were detected
            ground_truth_positions: Positions where context was missing
            method: Name of the method being evaluated

        Returns:
            EvaluationResult with all metrics
        """
        result = EvaluationResult(example_id=example_id, method=method)

        # Answer quality
        result.answer_correctness = self.answer_correctness(
            generated_answer, ground_truth_answer, ground_truth_keywords
        )
        result.answer_completeness = self.answer_completeness(
            generated_answer, ground_truth_answer
        )
        result.hallucination_rate = self.hallucination_rate(
            generated_answer, ground_truth_answer, ground_truth_keywords
        )

        # Context quality
        result.context_precision = self.context_precision(
            retrieved_files, ground_truth_files
        )
        result.context_recall = self.context_recall(
            retrieved_files, ground_truth_files
        )

        # Efficiency
        result.token_efficiency = self.token_efficiency(tokens_used, baseline_tokens)
        result.tokens_used = tokens_used
        result.baseline_tokens = baseline_tokens

        # Spike detection
        if detected_positions is not None and ground_truth_positions is not None:
            result.spike_precision = self.spike_precision(
                detected_positions, ground_truth_positions
            )
            result.spike_recall = self.spike_recall(
                detected_positions, ground_truth_positions
            )

        # Metadata
        result.generation_time = generation_time
        result.num_retrievals = num_retrievals

        return result
