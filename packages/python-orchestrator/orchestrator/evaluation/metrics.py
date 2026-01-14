"""
Evaluation Metrics for CCE Research.

Implements 8 metrics for assessing adaptive retrieval quality:
1. answer_correctness - LLM/keyword scoring
2. answer_completeness - semantic similarity
3. hallucination_rate - unsupported claims
4. context_precision - retrieved accuracy
5. context_recall - coverage of ground truth
6. token_efficiency - tokens saved vs baseline
7. spike_precision - trigger accuracy
8. spike_recall - trigger coverage

Phase 4, Week 8-9: Evaluation Framework
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
import numpy as np


@dataclass
class EvaluationResult:
    """Result of evaluating a single example."""
    example_id: str
    method: str = ""

    # Answer quality metrics
    answer_correctness: float = 0.0      # [0, 1]
    answer_completeness: float = 0.0     # [0, 1]
    hallucination_rate: float = 0.0      # [0, 1] (lower is better)

    # Context quality metrics
    context_precision: float = 0.0       # [0, 1]
    context_recall: float = 0.0          # [0, 1]

    # Efficiency metrics
    token_efficiency: float = 0.0        # [0, 1]
    tokens_used: int = 0
    baseline_tokens: int = 0

    # Spike detection metrics
    spike_precision: float = 0.0         # [0, 1]
    spike_recall: float = 0.0            # [0, 1]

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
            "baseline_tokens": self.baseline_tokens,
            "spike_precision": self.spike_precision,
            "spike_recall": self.spike_recall,
            "spike_f1": self.get_f1_spike(),
            "composite_score": self.get_composite_score(),
            "generation_time": self.generation_time,
            "num_retrievals": self.num_retrievals,
        }

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(id='{self.example_id}', "
            f"correctness={self.answer_correctness:.2f}, "
            f"composite={self.get_composite_score():.2f})"
        )


class EvaluationMetrics:
    """
    Evaluation metrics calculator.

    Uses sentence-transformers for semantic similarity and optionally
    LLM-as-judge for correctness scoring.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_judge_model: Optional[str] = None,
        llm_judge_api_key: Optional[str] = None,
    ):
        """
        Initialize metrics calculator.

        Args:
            embedding_model: SentenceTransformer model name
            llm_judge_model: Optional LLM model for judging (e.g., 'gpt-4')
            llm_judge_api_key: API key for LLM judge
        """
        self.embedding_model_name = embedding_model
        self.llm_judge_model = llm_judge_model
        self.llm_judge_api_key = llm_judge_api_key
        self._model = None

    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                print("Warning: sentence-transformers not available, using fallback methods")
                self._model = None
        return self._model

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text."""
        if self.model is None:
            return None
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
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
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

        Uses LLM-as-judge if available, otherwise falls back to keyword + semantic.

        Args:
            predicted: Generated answer
            ground_truth: Reference answer
            keywords: Important keywords that should appear
            keyword_weight: Weight for keyword matching (default 0.4)
            semantic_weight: Weight for semantic similarity (default 0.6)

        Returns:
            Score between 0 and 1
        """
        # Try LLM judge first
        if self.llm_judge_model and self.llm_judge_api_key:
            llm_score = self._llm_judge_correctness(predicted, ground_truth)
            if llm_score is not None:
                return llm_score

        # Keyword score
        keyword_score = 0.0
        if keywords:
            predicted_lower = predicted.lower()
            matches = sum(1 for kw in keywords if kw.lower() in predicted_lower)
            keyword_score = matches / len(keywords) if keywords else 0.0

        # Semantic score
        pred_emb = self._get_embedding(predicted)
        truth_emb = self._get_embedding(ground_truth)

        if pred_emb is not None and truth_emb is not None:
            semantic_score = max(0.0, self._cosine_similarity(pred_emb, truth_emb))
        else:
            # Fallback to word overlap
            pred_words = set(predicted.lower().split())
            truth_words = set(ground_truth.lower().split())
            semantic_score = len(pred_words & truth_words) / max(len(truth_words), 1)

        # Weighted combination
        if keywords:
            return keyword_weight * keyword_score + semantic_weight * semantic_score
        else:
            return semantic_score

    def _llm_judge_correctness(self, generated: str, ground_truth: str) -> Optional[float]:
        """Use LLM as judge for correctness scoring."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.llm_judge_api_key)

            prompt = f"""Rate how well the generated answer matches the expected answer.
Score from 0.0 to 1.0 where:
- 1.0 = Perfect match, covers all key points
- 0.7 = Good match, covers most key points
- 0.4 = Partial match, some relevant information
- 0.0 = Completely wrong or irrelevant

Expected Answer:
{ground_truth}

Generated Answer:
{generated}

Respond with ONLY a number between 0.0 and 1.0:"""

            response = client.chat.completions.create(
                model=self.llm_judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )

            score_text = response.choices[0].message.content.strip()
            return float(score_text)

        except Exception as e:
            print(f"LLM judge error: {e}")
            return None

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

        if pred_emb is not None and truth_emb is not None:
            return max(0.0, self._cosine_similarity(pred_emb, truth_emb))
        else:
            # Fallback to word overlap
            pred_words = set(predicted.lower().split())
            truth_words = set(ground_truth.lower().split())
            if not truth_words:
                return 0.0
            return len(pred_words & truth_words) / len(truth_words)

    # =========================================================================
    # Metric 3: Hallucination Rate
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
        2. It has semantic similarity > threshold to any context sentence, OR
        3. It contains facts/entities that match the context

        Args:
            predicted: Generated answer
            context: Provided context (ground truth answer or retrieved docs)
            keywords: Keywords that indicate supported claims
            similarity_threshold: Threshold for semantic support (default 0.7)

        Returns:
            Fraction of unsupported claims (0 = no hallucination, 1 = all hallucinated)
        """
        # Extract claims from predicted answer
        claims = self._extract_claims(predicted)
        if not claims:
            return 0.0  # No claims to evaluate

        # Split context into sentences for semantic matching
        context_sentences = self._split_into_sentences(context)
        if not context_sentences:
            context_sentences = [context]

        # Get embeddings for context sentences
        context_embeddings = []
        if self.model is not None:
            context_embeddings = [self._get_embedding(s) for s in context_sentences]
            context_embeddings = [e for e in context_embeddings if e is not None]

        # Prepare keywords for matching
        keywords_lower: Set[str] = set()
        if keywords:
            keywords_lower = {kw.lower() for kw in keywords}

        # Extract facts from context
        context_facts = self._extract_facts(context)

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

            # Check 2: Fact/entity overlap
            if not is_supported:
                claim_facts = self._extract_facts(claim)
                if claim_facts and context_facts:
                    overlap = len(claim_facts & context_facts)
                    if overlap > 0:
                        is_supported = True

            # Check 3: Semantic similarity to context
            if not is_supported and context_embeddings:
                claim_emb = self._get_embedding(claim)
                if claim_emb is not None:
                    max_similarity = max(
                        self._cosine_similarity(claim_emb, ctx_emb)
                        for ctx_emb in context_embeddings
                    )
                    if max_similarity >= similarity_threshold:
                        is_supported = True

            # Check 4: Check for contradictions
            if is_supported:
                contradiction_penalty = self._check_contradiction(claim, context)
                if contradiction_penalty > 0.3:
                    is_supported = False

            if not is_supported:
                unsupported_count += 1

        return unsupported_count / len(claims)

    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text.

        Splits on sentence boundaries and filters out non-claims.
        """
        sentences = re.split(r'[.!?]\n|\n\d+\.\s+|(?<=[.!?])\s+', text)

        claims = []
        filler_patterns = [
            r'^(the |this |a |an )?answer is',
            r'^in summary',
            r'^to summarize',
            r'^overall',
            r'^basically',
            r'^essentially',
        ]

        for sentence in sentences:
            sentence = sentence.strip()

            # Skip short sentences
            if len(sentence) < 15:
                continue

            # Skip questions
            if sentence.endswith('?'):
                continue

            # Skip filler phrases
            is_filler = any(
                re.match(pattern, sentence.lower())
                for pattern in filler_patterns
            )
            if is_filler:
                continue

            # Skip list markers only
            if re.match(r'^[\d\-\*•]+\.?\s*$', sentence):
                continue

            claims.append(sentence)

        return claims

    def _extract_facts(self, text: str) -> Set[str]:
        """
        Extract factual elements from text.

        Looks for technical terms, function names, file paths, etc.
        """
        facts = set()
        text_lower = text.lower()

        # CamelCase words (class names, etc.)
        camel_case = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
        facts.update(w.lower() for w in camel_case)

        # snake_case identifiers
        snake_case = re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', text_lower)
        facts.update(snake_case)

        # Function calls
        func_calls = re.findall(r'\b(\w+)\s*\(', text)
        facts.update(f.lower() for f in func_calls)

        # File paths
        paths = re.findall(r'[\w/\\]+\.\w{2,4}\b', text)
        facts.update(p.lower() for p in paths)

        # HTTP methods and status codes
        http_methods = re.findall(r'\b(GET|POST|PUT|DELETE|PATCH)\b', text)
        facts.update(m.lower() for m in http_methods)

        status_codes = re.findall(r'\b([1-5]\d{2})\b', text)
        facts.update(status_codes)

        # Quoted strings
        quoted = re.findall(r'["\']([^"\']+)["\']', text)
        facts.update(q.lower() for q in quoted if len(q) < 50)

        return facts

    def _check_contradiction(self, claim: str, ground_truth: str) -> float:
        """
        Check if claim contradicts ground truth.

        Returns penalty score [0, 0.5] for contradictions.
        """
        claim_lower = claim.lower()
        truth_lower = ground_truth.lower()

        contradiction_patterns = [
            (r'\bdoes not\b', r'\bdoes\b'),
            (r'\bcannot\b', r'\bcan\b'),
            (r'\bnever\b', r'\balways\b'),
            (r'\bno\b', r'\byes\b'),
            (r'\bwithout\b', r'\bwith\b'),
            (r'\bdisabled\b', r'\benabled\b'),
            (r'\bfalse\b', r'\btrue\b'),
        ]

        penalty = 0.0
        for neg_pattern, pos_pattern in contradiction_patterns:
            claim_has_neg = bool(re.search(neg_pattern, claim_lower))
            truth_has_pos = bool(re.search(pos_pattern, truth_lower))
            claim_has_pos = bool(re.search(pos_pattern, claim_lower))
            truth_has_neg = bool(re.search(neg_pattern, truth_lower))

            if (claim_has_neg and truth_has_pos and not truth_has_neg):
                penalty += 0.15
            elif (claim_has_pos and truth_has_neg and not claim_has_neg):
                penalty += 0.15

        return min(penalty, 0.5)

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

        retrieved_set = self._normalize_paths(retrieved_files)
        truth_set = self._normalize_paths(ground_truth_files)

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
            return 1.0  # No ground truth to retrieve

        retrieved_set = self._normalize_paths(retrieved_files)
        truth_set = self._normalize_paths(ground_truth_files)

        intersection = retrieved_set & truth_set
        return len(intersection) / len(truth_set)

    def _normalize_paths(self, paths: List[str]) -> Set[str]:
        """Normalize file paths for comparison."""
        normalized = set()
        for path in paths:
            # Extract filename and normalize
            name = path.strip().replace("\\", "/").strip("/")
            # Get just the filename for comparison
            basename = name.split("/")[-1]
            normalized.add(basename.lower())
        return normalized

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
            return 0.0

        correct = 0
        for detected in detected_positions:
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

        Args:
            detected_positions: Token positions where spikes were detected
            ground_truth_positions: Token positions where context was actually missing
            tolerance: Window size for matching (default 3 tokens)

        Returns:
            Recall score between 0 and 1
        """
        if not ground_truth_positions:
            return 1.0
        if not detected_positions:
            return 0.0

        matched = 0
        for truth in ground_truth_positions:
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

    # =========================================================================
    # Aggregate Results
    # =========================================================================
    def aggregate_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Aggregate results across multiple examples.

        Args:
            results: List of EvaluationResult objects

        Returns:
            Dictionary with aggregate statistics
        """
        if not results:
            return {}

        metrics = {
            'answer_correctness': [],
            'answer_completeness': [],
            'hallucination_rate': [],
            'context_precision': [],
            'context_recall': [],
            'token_efficiency': [],
            'spike_precision': [],
            'spike_recall': [],
            'composite_score': [],
        }

        for r in results:
            metrics['answer_correctness'].append(r.answer_correctness)
            metrics['answer_completeness'].append(r.answer_completeness)
            metrics['hallucination_rate'].append(r.hallucination_rate)
            metrics['context_precision'].append(r.context_precision)
            metrics['context_recall'].append(r.context_recall)
            metrics['token_efficiency'].append(r.token_efficiency)
            metrics['spike_precision'].append(r.spike_precision)
            metrics['spike_recall'].append(r.spike_recall)
            metrics['composite_score'].append(r.get_composite_score())

        aggregate = {
            'count': len(results),
            'metrics': {},
        }

        for name, values in metrics.items():
            arr = np.array(values)
            aggregate['metrics'][name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'median': float(np.median(arr)),
            }

        # Overall statistics
        aggregate['total_tokens'] = sum(r.tokens_used for r in results)
        aggregate['total_retrievals'] = sum(r.num_retrievals for r in results)
        aggregate['total_time'] = sum(r.generation_time for r in results)

        return aggregate


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_metrics(
    use_llm_judge: bool = False,
    llm_model: str = "gpt-4",
    api_key: Optional[str] = None,
) -> EvaluationMetrics:
    """
    Factory function to create evaluation metrics.

    Args:
        use_llm_judge: Whether to use LLM for judging
        llm_model: LLM model name for judging
        api_key: API key for LLM

    Returns:
        Configured EvaluationMetrics
    """
    return EvaluationMetrics(
        llm_judge_model=llm_model if use_llm_judge else None,
        llm_judge_api_key=api_key if use_llm_judge else None,
    )
