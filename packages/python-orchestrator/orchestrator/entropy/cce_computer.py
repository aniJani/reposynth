"""
Contrastive Code Entropy (CCE) Computation Module

This module provides the complete CCE pipeline:
1. Token classification (hybrid keyword + embedding)
2. Logit partitioning (code vs language)
3. Entropy calculation (H_code, H_language)
4. CCE computation (H_code - H_language)

Week 2, Day 5: Hybrid Integration & CCE Computation
"""

from typing import Dict, List, Tuple, Optional, Literal
import numpy as np


# Class-level cache for vocabulary classifications (expensive to compute)
_VOCAB_CLASSIFICATION_CACHE: Dict[str, Dict] = {}


class CCEComputer:
    """
    Complete CCE computation pipeline.

    Integrates token classification and entropy calculation
    to compute Contrastive Code Entropy for uncertainty-guided retrieval.
    """

    def __init__(
        self,
        tokenizer,
        use_hybrid: bool = True,
        embedding_model: str = 'all-MiniLM-L6-v2',
        embedding_margin: float = 0.05,
    ):
        """
        Initialize CCE computer.

        Args:
            tokenizer: HuggingFace tokenizer for decoding token IDs
            use_hybrid: Use hybrid (keyword + embedding) classifier
            embedding_model: SentenceTransformer model name
            embedding_margin: Similarity margin for embedding classification
        """
        self.tokenizer = tokenizer

        # Import classifiers
        if use_hybrid:
            from .token_classifier import HybridClassifier
            self.classifier = HybridClassifier(
                embedding_model=embedding_model,
                embedding_margin=embedding_margin
            )
        else:
            from .token_classifier import KeywordClassifier
            self.classifier = KeywordClassifier()

        # Import entropy calculator
        from .calculator import EntropyCalculator
        self.entropy_calc = EntropyCalculator()

        # Diagnostic data
        self.last_classification: Optional[Dict] = None

    def compute_cce(
        self,
        logits: np.ndarray,
        return_diagnostics: bool = False
    ) -> Dict[str, float]:
        """
        Compute Contrastive Code Entropy for a set of logits.

        Args:
            logits: Array of shape [vocab_size] containing logits for next token
            return_diagnostics: Whether to return detailed diagnostic info

        Returns:
            Dictionary containing:
            - 'contrastive_entropy': CCE = H_code - H_language
            - 'code_entropy': Entropy over code tokens
            - 'language_entropy': Entropy over language tokens
            - 'total_entropy': Shannon entropy over all tokens
            - 'code_prob_mass': Total probability mass on code tokens
            - 'language_prob_mass': Total probability mass on language tokens
            - 'other_prob_mass': Total probability mass on other tokens
            - (if return_diagnostics) 'classification': token classification details
            - (if return_diagnostics) 'top_k_predictions': top-k tokens and probs
        """
        from .calculator import softmax

        # Convert logits to probabilities
        probs = softmax(logits)
        vocab_size = len(logits)

        # Classify all tokens in vocabulary
        classifications = self._classify_vocabulary(vocab_size)

        # Partition logits by classification
        code_indices = classifications['code']
        language_indices = classifications['language']
        other_indices = classifications['other']

        # Extract partitioned logits
        code_logits = logits[code_indices] if len(code_indices) > 0 else np.array([])
        language_logits = logits[language_indices] if len(language_indices) > 0 else np.array([])

        # Compute entropies
        h_total = self.entropy_calc.shannon_entropy(logits)

        h_code = (
            self.entropy_calc.shannon_entropy(code_logits)
            if len(code_logits) > 0
            else 0.0
        )

        h_language = (
            self.entropy_calc.shannon_entropy(language_logits)
            if len(language_logits) > 0
            else 0.0
        )

        # Compute CCE
        cce = h_code - h_language

        # Compute probability masses
        code_prob_mass = float(np.sum(probs[code_indices])) if len(code_indices) > 0 else 0.0
        language_prob_mass = float(np.sum(probs[language_indices])) if len(language_indices) > 0 else 0.0
        other_prob_mass = float(np.sum(probs[other_indices])) if len(other_indices) > 0 else 0.0

        # Build result
        result = {
            'contrastive_entropy': float(cce),
            'code_entropy': float(h_code),
            'language_entropy': float(h_language),
            'total_entropy': float(h_total),
            'code_prob_mass': code_prob_mass,
            'language_prob_mass': language_prob_mass,
            'other_prob_mass': other_prob_mass,
        }

        # Add diagnostics if requested
        if return_diagnostics:
            # Get top-k predictions
            from .calculator import get_top_k_predictions
            top_k_indices, top_k_probs = get_top_k_predictions(logits, k=10)

            top_k_tokens = [
                self.tokenizer.decode([idx])
                for idx in top_k_indices
            ]

            top_k_classes = [
                classifications['token_to_class'].get(idx, 'other')
                for idx in top_k_indices
            ]

            result['classification'] = {
                'code_count': len(code_indices),
                'language_count': len(language_indices),
                'other_count': len(other_indices),
                'code_coverage': len(code_indices) / vocab_size,
                'language_coverage': len(language_indices) / vocab_size,
                'other_coverage': len(other_indices) / vocab_size,
            }

            result['top_k_predictions'] = [
                {
                    'token': token,
                    'token_id': int(idx),
                    'probability': float(prob),
                    'classification': cls,
                }
                for token, idx, prob, cls in zip(
                    top_k_tokens, top_k_indices, top_k_probs, top_k_classes
                )
            ]

        # Store for inspection
        self.last_classification = classifications

        return result

    def _classify_vocabulary(self, vocab_size: int) -> Dict:
        """
        Classify all tokens in the vocabulary.

        Uses class-level caching to avoid re-classifying 150K+ tokens
        on every CCE computation.

        Args:
            vocab_size: Size of vocabulary

        Returns:
            Dictionary with:
            - 'code': array of indices for code tokens
            - 'language': array of indices for language tokens
            - 'other': array of indices for other tokens
            - 'token_to_class': mapping from token_id to class
        """
        global _VOCAB_CLASSIFICATION_CACHE

        # Build cache key from tokenizer and classifier config
        tokenizer_id = getattr(self.tokenizer, 'name_or_path', str(id(self.tokenizer)))
        classifier_type = type(self.classifier).__name__
        cache_key = f"{tokenizer_id}_{classifier_type}_{vocab_size}"

        # Return cached result if available
        if cache_key in _VOCAB_CLASSIFICATION_CACHE:
            return _VOCAB_CLASSIFICATION_CACHE[cache_key]

        # Compute classification (expensive - only done once per tokenizer)
        print(f"Classifying {vocab_size} tokens (will be cached)...")

        code_indices = []
        language_indices = []
        other_indices = []
        token_to_class = {}

        for token_id in range(vocab_size):
            # Decode token
            token_str = self.tokenizer.decode([token_id])

            # Classify token
            classification = self.classifier.classify(token_str)

            # Store
            token_to_class[token_id] = classification

            if classification == 'code':
                code_indices.append(token_id)
            elif classification == 'language':
                language_indices.append(token_id)
            else:
                other_indices.append(token_id)

        result = {
            'code': np.array(code_indices),
            'language': np.array(language_indices),
            'other': np.array(other_indices),
            'token_to_class': token_to_class,
        }

        # Cache result
        _VOCAB_CLASSIFICATION_CACHE[cache_key] = result
        print(f"Vocabulary cached: {len(code_indices)} code, {len(language_indices)} language, {len(other_indices)} other")

        return result

    def get_classifier_stats(self) -> Dict[str, int]:
        """Get diagnostic statistics from the classifier."""
        if hasattr(self.classifier, 'get_stats'):
            return self.classifier.get_stats()
        return {}

    def reset_classifier_stats(self):
        """Reset diagnostic statistics."""
        if hasattr(self.classifier, 'reset_stats'):
            self.classifier.reset_stats()


# ============================================================================
# FUNCTIONAL INTERFACE
# ============================================================================

def compute_cce_from_logits(
    logits: np.ndarray,
    tokenizer,
    use_hybrid: bool = True,
    embedding_model: str = 'all-MiniLM-L6-v2',
    embedding_margin: float = 0.05,
    return_diagnostics: bool = False
) -> Dict[str, float]:
    """
    Compute CCE from logits (convenience function).

    Args:
        logits: Array of shape [vocab_size] containing logits
        tokenizer: HuggingFace tokenizer
        use_hybrid: Use hybrid classifier (default True)
        embedding_model: SentenceTransformer model name
        embedding_margin: Similarity margin for embeddings
        return_diagnostics: Return detailed diagnostics

    Returns:
        Dictionary with CCE and related metrics
    """
    computer = CCEComputer(
        tokenizer=tokenizer,
        use_hybrid=use_hybrid,
        embedding_model=embedding_model,
        embedding_margin=embedding_margin
    )

    return computer.compute_cce(logits, return_diagnostics=return_diagnostics)
