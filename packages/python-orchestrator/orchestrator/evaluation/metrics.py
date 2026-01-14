"""
Evaluation Metrics Module for Adaptive Code Generation.

This module implements 8 evaluation metrics for comprehensive assessment:

1. Answer Correctness - LLM-as-judge scoring [0-1]
2. Answer Completeness - Semantic similarity to ground truth
3. Hallucination Rate - Fraction of unsupported claims
4. Context Precision - |retrieved ∩ truth| / |retrieved|
5. Context Recall - |retrieved ∩ truth| / |truth|
6. Token Efficiency - 1 - (tokens_used / baseline)
7. Spike Precision - Accuracy of uncertainty triggers
8. Spike Recall - Coverage of needed retrievals

Phase 4, Week 8-9: Evaluation Framework
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
import numpy as np
import re


@dataclass
class EvaluationResult:
    """Result of evaluating a single example."""
    example_id: str

    # Core metrics
    answer_correctness: float = 0.0      # [0, 1]
    answer_completeness: float = 0.0     # [0, 1]
    hallucination_rate: float = 0.0      # [0, 1] (lower is better)

    # Context metrics
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
    method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_f1_context(self) -> float:
        """Compute F1 score for context retrieval."""
        if self.context_precision + self.context_recall == 0:
            return 0.0
        return 2 * (self.context_precision * self.context_recall) / \
               (self.context_precision + self.context_recall)

    def get_f1_spike(self) -> float:
        """Compute F1 score for spike detection."""
        if self.spike_precision + self.spike_recall == 0:
            return 0.0
        return 2 * (self.spike_precision * self.spike_recall) / \
               (self.spike_precision + self.spike_recall)

    def get_composite_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute weighted composite score.

        Args:
            weights: Dict mapping metric names to weights (default: equal weights)

        Returns:
            Weighted average score [0, 1]
        """
        if weights is None:
            weights = {
                'answer_correctness': 0.25,
                'answer_completeness': 0.15,
                'hallucination_rate': 0.15,  # Inverted: 1 - rate
                'context_precision': 0.10,
                'context_recall': 0.10,
                'token_efficiency': 0.10,
                'spike_precision': 0.075,
                'spike_recall': 0.075,
            }

        score = 0.0
        score += weights.get('answer_correctness', 0) * self.answer_correctness
        score += weights.get('answer_completeness', 0) * self.answer_completeness
        score += weights.get('hallucination_rate', 0) * (1 - self.hallucination_rate)
        score += weights.get('context_precision', 0) * self.context_precision
        score += weights.get('context_recall', 0) * self.context_recall
        score += weights.get('token_efficiency', 0) * self.token_efficiency
        score += weights.get('spike_precision', 0) * self.spike_precision
        score += weights.get('spike_recall', 0) * self.spike_recall

        return score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'example_id': self.example_id,
            'answer_correctness': self.answer_correctness,
            'answer_completeness': self.answer_completeness,
            'hallucination_rate': self.hallucination_rate,
            'context_precision': self.context_precision,
            'context_recall': self.context_recall,
            'token_efficiency': self.token_efficiency,
            'tokens_used': self.tokens_used,
            'baseline_tokens': self.baseline_tokens,
            'spike_precision': self.spike_precision,
            'spike_recall': self.spike_recall,
            'f1_context': self.get_f1_context(),
            'f1_spike': self.get_f1_spike(),
            'composite_score': self.get_composite_score(),
            'generation_time': self.generation_time,
            'num_retrievals': self.num_retrievals,
            'method': self.method,
        }

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(id='{self.example_id}', "
            f"correctness={self.answer_correctness:.2f}, "
            f"composite={self.get_composite_score():.2f})"
        )


class EvaluationMetrics:
    """
    Compute evaluation metrics for adaptive code generation.

    This class provides methods to compute all 8 evaluation metrics
    used in the research evaluation.

    Example:
        >>> metrics = EvaluationMetrics(embedding_model='all-MiniLM-L6-v2')
        >>> result = metrics.evaluate(
        ...     example_id='test_001',
        ...     generated_answer='The auth uses JWT...',
        ...     ground_truth_answer='Authentication is done via JWT...',
        ...     retrieved_files=['auth.py', 'config.py'],
        ...     ground_truth_files=['auth.py', 'routes.py'],
        ... )
        >>> print(f"Correctness: {result.answer_correctness:.2f}")
    """

    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        llm_judge_model: Optional[str] = None,
        llm_judge_api_key: Optional[str] = None,
    ):
        """
        Initialize evaluation metrics.

        Args:
            embedding_model: Sentence transformer model for embeddings
            llm_judge_model: Model for LLM-as-judge (e.g., 'gpt-4')
            llm_judge_api_key: API key for LLM judge
        """
        self.embedding_model_name = embedding_model
        self.llm_judge_model = llm_judge_model
        self.llm_judge_api_key = llm_judge_api_key

        # Lazy load embedding model
        self._embedding_model = None

    @property
    def embedding_model(self):
        """Lazy load sentence transformer model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                print("Warning: sentence-transformers not available")
                self._embedding_model = None
        return self._embedding_model

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
        spike_positions: Optional[List[int]] = None,
        expected_spike_positions: Optional[List[int]] = None,
        generation_time: float = 0.0,
        num_retrievals: int = 0,
        method: str = "",
    ) -> EvaluationResult:
        """
        Compute all evaluation metrics for a single example.

        Args:
            example_id: Unique identifier for the example
            generated_answer: Model's generated response
            ground_truth_answer: Expected answer
            retrieved_files: Files retrieved by the system
            ground_truth_files: Files that should have been retrieved
            ground_truth_keywords: Key terms that should appear
            tokens_used: Tokens used in context
            baseline_tokens: Tokens used by baseline (for efficiency)
            spike_positions: Positions where retrieval was triggered
            expected_spike_positions: Positions where retrieval should trigger
            generation_time: Time to generate response
            num_retrievals: Number of retrievals performed
            method: Name of the method being evaluated

        Returns:
            EvaluationResult with all metrics
        """
        result = EvaluationResult(
            example_id=example_id,
            tokens_used=tokens_used,
            baseline_tokens=baseline_tokens,
            generation_time=generation_time,
            num_retrievals=num_retrievals,
            method=method,
        )

        # 1. Answer Correctness (LLM judge or keyword-based fallback)
        result.answer_correctness = self.answer_correctness(
            generated=generated_answer,
            ground_truth=ground_truth_answer,
            keywords=ground_truth_keywords,
        )

        # 2. Answer Completeness (semantic similarity)
        result.answer_completeness = self.answer_completeness(
            generated=generated_answer,
            ground_truth=ground_truth_answer,
        )

        # 3. Hallucination Rate
        result.hallucination_rate = self.hallucination_rate(
            generated=generated_answer,
            ground_truth=ground_truth_answer,
            keywords=ground_truth_keywords,
        )

        # 4. Context Precision
        result.context_precision = self.context_precision(
            retrieved=retrieved_files,
            ground_truth=ground_truth_files,
        )

        # 5. Context Recall
        result.context_recall = self.context_recall(
            retrieved=retrieved_files,
            ground_truth=ground_truth_files,
        )

        # 6. Token Efficiency
        result.token_efficiency = self.token_efficiency(
            tokens_used=tokens_used,
            baseline_tokens=baseline_tokens,
        )

        # 7. Spike Precision
        if spike_positions is not None and expected_spike_positions is not None:
            result.spike_precision = self.spike_precision(
                detected=spike_positions,
                expected=expected_spike_positions,
            )

        # 8. Spike Recall
        if spike_positions is not None and expected_spike_positions is not None:
            result.spike_recall = self.spike_recall(
                detected=spike_positions,
                expected=expected_spike_positions,
            )

        return result

    # ========================================================================
    # INDIVIDUAL METRICS
    # ========================================================================

    def answer_correctness(
        self,
        generated: str,
        ground_truth: str,
        keywords: Optional[List[str]] = None,
    ) -> float:
        """
        Compute answer correctness score.

        Uses LLM-as-judge if available, otherwise falls back to
        keyword matching + semantic similarity.

        Args:
            generated: Generated answer
            ground_truth: Expected answer
            keywords: Key terms that should appear

        Returns:
            Correctness score [0, 1]
        """
        if self.llm_judge_model and self.llm_judge_api_key:
            return self._llm_judge_correctness(generated, ground_truth)

        # Fallback: combine keyword matching and semantic similarity
        keyword_score = 0.0
        if keywords:
            keyword_score = self._keyword_overlap(generated, keywords)

        semantic_score = self.answer_completeness(generated, ground_truth)

        # Weight: 40% keywords, 60% semantic
        if keywords:
            return 0.4 * keyword_score + 0.6 * semantic_score
        return semantic_score

    def _llm_judge_correctness(self, generated: str, ground_truth: str) -> float:
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
            return self.answer_completeness(generated, ground_truth)

    def _keyword_overlap(self, text: str, keywords: List[str]) -> float:
        """Compute fraction of keywords present in text."""
        text_lower = text.lower()
        found = sum(1 for kw in keywords if kw.lower() in text_lower)
        return found / len(keywords) if keywords else 0.0

    def answer_completeness(
        self,
        generated: str,
        ground_truth: str,
    ) -> float:
        """
        Compute semantic similarity between generated and ground truth.

        Uses sentence embeddings to measure how complete the answer is.

        Args:
            generated: Generated answer
            ground_truth: Expected answer

        Returns:
            Completeness score [0, 1]
        """
        if not self.embedding_model:
            # Fallback to simple word overlap
            gen_words = set(generated.lower().split())
            truth_words = set(ground_truth.lower().split())
            if not truth_words:
                return 0.0
            overlap = len(gen_words & truth_words)
            return overlap / len(truth_words)

        try:
            embeddings = self.embedding_model.encode(
                [generated, ground_truth],
                convert_to_numpy=True,
            )
            # Cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            # Normalize to [0, 1]
            return float(max(0, (similarity + 1) / 2))
        except Exception:
            return 0.0

    def hallucination_rate(
        self,
        generated: str,
        ground_truth: str,
        keywords: Optional[List[str]] = None,
    ) -> float:
        """
        Estimate hallucination rate in generated answer.

        Measures fraction of claims that are not supported by ground truth.
        Uses multiple signals:
        1. Semantic similarity (stricter threshold)
        2. Entity/fact overlap
        3. Contradiction detection
        4. Keyword validation

        Args:
            generated: Generated answer
            ground_truth: Expected answer
            keywords: Key terms (helps identify valid claims)

        Returns:
            Hallucination rate [0, 1] (lower is better)
        """
        # Extract claims from generated answer
        gen_claims = self._extract_claims(generated)
        if not gen_claims:
            return 0.0

        truth_lower = ground_truth.lower()
        valid_keywords = set(kw.lower() for kw in (keywords or []))

        # Extract facts/entities from ground truth
        truth_facts = self._extract_facts(ground_truth)

        hallucinated = 0
        for claim in gen_claims:
            claim_lower = claim.lower()

            # Score 1: Keyword overlap (valid terms present)
            keyword_score = sum(1 for kw in valid_keywords if kw in claim_lower)
            keyword_score = min(keyword_score / max(len(valid_keywords), 1), 1.0)

            # Score 2: Fact/entity overlap
            claim_facts = self._extract_facts(claim)
            fact_overlap = len(claim_facts & truth_facts)
            fact_score = fact_overlap / max(len(claim_facts), 1) if claim_facts else 0.0

            # Score 3: Semantic similarity (stricter threshold of 0.5)
            semantic_score = 0.0
            if self.embedding_model:
                try:
                    # Compare claim to each ground truth sentence
                    truth_sentences = self._extract_sentences(ground_truth)
                    if truth_sentences:
                        embeddings = self.embedding_model.encode(
                            [claim] + truth_sentences,
                            convert_to_numpy=True,
                        )
                        claim_emb = embeddings[0]
                        max_sim = 0.0
                        for truth_emb in embeddings[1:]:
                            sim = np.dot(claim_emb, truth_emb) / (
                                np.linalg.norm(claim_emb) * np.linalg.norm(truth_emb) + 1e-8
                            )
                            max_sim = max(max_sim, sim)
                        semantic_score = max_sim
                except Exception:
                    semantic_score = 0.0
            else:
                # Fallback: word overlap ratio
                claim_words = set(claim_lower.split())
                truth_words = set(truth_lower.split())
                overlap = len(claim_words & truth_words)
                semantic_score = overlap / max(len(claim_words), 1)

            # Score 4: Check for contradictions
            contradiction_penalty = self._check_contradiction(claim, ground_truth)

            # Combined support score
            # Weighted: 30% keywords, 30% facts, 40% semantic
            support_score = (
                0.30 * keyword_score +
                0.30 * fact_score +
                0.40 * semantic_score
            ) - contradiction_penalty

            # Threshold for hallucination: support score below 0.3
            if support_score < 0.3:
                hallucinated += 1

        return hallucinated / len(gen_claims)

    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text.

        Splits on sentence boundaries and filters out:
        - Very short statements
        - Pure questions
        - Filler phrases
        """
        # Split on sentence boundaries
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

        Looks for:
        - Technical terms (CamelCase, snake_case)
        - Function/method names
        - File paths
        - Numbers with context
        - Quoted strings
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

        # Pattern pairs that indicate contradiction
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

            # Contradiction: claim says "not X" but truth says "X"
            # or claim says "X" but truth says "not X"
            if (claim_has_neg and truth_has_pos and not truth_has_neg):
                penalty += 0.15
            elif (claim_has_pos and truth_has_neg and not claim_has_neg):
                penalty += 0.15

        return min(penalty, 0.5)

    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]\s+|\n', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def context_precision(
        self,
        retrieved: List[str],
        ground_truth: List[str],
    ) -> float:
        """
        Compute context precision.

        Precision = |retrieved ∩ truth| / |retrieved|

        Args:
            retrieved: Files/chunks retrieved
            ground_truth: Files that should be retrieved

        Returns:
            Precision score [0, 1]
        """
        if not retrieved:
            return 0.0

        retrieved_set = self._normalize_paths(retrieved)
        truth_set = self._normalize_paths(ground_truth)

        correct = len(retrieved_set & truth_set)
        return correct / len(retrieved_set)

    def context_recall(
        self,
        retrieved: List[str],
        ground_truth: List[str],
    ) -> float:
        """
        Compute context recall.

        Recall = |retrieved ∩ truth| / |truth|

        Args:
            retrieved: Files/chunks retrieved
            ground_truth: Files that should be retrieved

        Returns:
            Recall score [0, 1]
        """
        if not ground_truth:
            return 1.0  # No ground truth to retrieve

        retrieved_set = self._normalize_paths(retrieved)
        truth_set = self._normalize_paths(ground_truth)

        correct = len(retrieved_set & truth_set)
        return correct / len(truth_set)

    def _normalize_paths(self, paths: List[str]) -> Set[str]:
        """Normalize file paths for comparison."""
        normalized = set()
        for path in paths:
            # Extract filename
            name = path.split('/')[-1].split('\\')[-1]
            normalized.add(name.lower())
        return normalized

    def token_efficiency(
        self,
        tokens_used: int,
        baseline_tokens: int,
    ) -> float:
        """
        Compute token efficiency.

        Efficiency = 1 - (tokens_used / baseline_tokens)

        Higher is better (using fewer tokens than baseline).

        Args:
            tokens_used: Tokens used by adaptive method
            baseline_tokens: Tokens used by baseline (e.g., full context)

        Returns:
            Efficiency score [0, 1]
        """
        if baseline_tokens <= 0:
            return 0.0
        if tokens_used >= baseline_tokens:
            return 0.0

        return 1 - (tokens_used / baseline_tokens)

    def spike_precision(
        self,
        detected: List[int],
        expected: List[int],
        tolerance: int = 3,
    ) -> float:
        """
        Compute spike detection precision.

        Precision = correctly detected / total detected

        Args:
            detected: Positions where retrieval was triggered
            expected: Positions where retrieval should have triggered
            tolerance: Positions within tolerance count as correct

        Returns:
            Precision score [0, 1]
        """
        if not detected:
            return 0.0

        correct = 0
        for d in detected:
            for e in expected:
                if abs(d - e) <= tolerance:
                    correct += 1
                    break

        return correct / len(detected)

    def spike_recall(
        self,
        detected: List[int],
        expected: List[int],
        tolerance: int = 3,
    ) -> float:
        """
        Compute spike detection recall.

        Recall = correctly detected / total expected

        Args:
            detected: Positions where retrieval was triggered
            expected: Positions where retrieval should have triggered
            tolerance: Positions within tolerance count as correct

        Returns:
            Recall score [0, 1]
        """
        if not expected:
            return 1.0  # No expected spikes

        correct = 0
        for e in expected:
            for d in detected:
                if abs(d - e) <= tolerance:
                    correct += 1
                    break

        return correct / len(expected)

    # ========================================================================
    # AGGREGATE METRICS
    # ========================================================================

    def aggregate_results(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, Any]:
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
