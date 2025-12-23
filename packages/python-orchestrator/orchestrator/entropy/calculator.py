"""
Entropy calculation module for measuring uncertainty in LLM outputs.

This module provides various entropy metrics for quantifying uncertainty
in language model predictions, including:
- Shannon entropy (standard information-theoretic measure)
- Normalized entropy (scaled to [0, 1])
- Probability differential (UnCert-CoT style: 1 - max(P))
- Top-k entropy (entropy over top-k tokens only)

All functions operate on logits (unnormalized log-probabilities) from LLM output.
"""

import numpy as np
from typing import Union, Tuple
from scipy.stats import entropy as scipy_entropy


class EntropyCalculator:
    """
    Compute various entropy metrics from logits.

    This class provides static methods for calculating different entropy metrics
    from model logits. All methods handle numerical stability through scipy's
    entropy implementation.

    Example:
        >>> logits = np.array([2.0, 1.0, 0.5, 0.1])
        >>> h = EntropyCalculator.shannon_entropy(logits)
        >>> print(f"Entropy: {h:.3f} bits")
    """

    @staticmethod
    def shannon_entropy(logits: np.ndarray) -> float:
        """
        Compute standard Shannon entropy from logits.

        H = -Σ p(x) log₂ p(x)

        Shannon entropy measures the average information content or uncertainty
        in a probability distribution. Higher entropy indicates more uncertainty.

        Args:
            logits: Array of shape (vocab_size,) containing unnormalized log-probabilities

        Returns:
            Entropy in bits (using log base 2)

        Raises:
            ValueError: If logits is empty or contains invalid values

        Example:
            >>> # Uniform distribution (high entropy)
            >>> uniform_logits = np.ones(100)
            >>> h_uniform = EntropyCalculator.shannon_entropy(uniform_logits)
            >>> # Peaked distribution (low entropy)
            >>> peaked_logits = np.array([10.0, 0.1, 0.1, 0.1])
            >>> h_peaked = EntropyCalculator.shannon_entropy(peaked_logits)
            >>> assert h_uniform > h_peaked
        """
        if logits.size == 0:
            raise ValueError("Logits array cannot be empty")

        if not np.all(np.isfinite(logits)):
            raise ValueError("Logits contains non-finite values (inf or nan)")

        # Convert logits to probabilities using softmax
        # Subtract max for numerical stability
        logits_stable = logits - np.max(logits)
        probs = np.exp(logits_stable) / np.sum(np.exp(logits_stable))

        # Compute Shannon entropy using scipy (handles edge cases)
        # base=2 gives entropy in bits
        return float(scipy_entropy(probs, base=2))

    @staticmethod
    def normalized_entropy(logits: np.ndarray) -> float:
        """
        Compute normalized entropy (scaled to [0, 1]).

        H_norm = H / log₂(V)

        where V is the vocabulary size. This normalization makes entropy
        comparable across different vocabulary sizes.

        Args:
            logits: Array of shape (vocab_size,) containing unnormalized log-probabilities

        Returns:
            Normalized entropy in range [0, 1], where:
            - 0 = completely certain (one token has probability 1)
            - 1 = completely uncertain (uniform distribution)

        Raises:
            ValueError: If logits is empty or contains invalid values

        Example:
            >>> logits = np.random.randn(32000)  # CodeLlama vocab size
            >>> h_norm = EntropyCalculator.normalized_entropy(logits)
            >>> assert 0 <= h_norm <= 1
        """
        if logits.size == 0:
            raise ValueError("Logits array cannot be empty")

        h = EntropyCalculator.shannon_entropy(logits)
        vocab_size = logits.shape[0]
        max_entropy = np.log2(vocab_size)

        return h / max_entropy

    @staticmethod
    def probability_differential(logits: np.ndarray) -> float:
        """
        Compute probability differential (UnCert-CoT style).

        PD = 1 - max(P)

        This metric measures uncertainty as the complement of the maximum
        probability. It's simpler than entropy but effective for detecting
        low-confidence predictions.

        Args:
            logits: Array of shape (vocab_size,) containing unnormalized log-probabilities

        Returns:
            Probability differential in range [0, 1], where:
            - 0 = completely certain (max probability = 1)
            - ~1 = completely uncertain (probabilities spread out)

        Raises:
            ValueError: If logits is empty or contains invalid values

        Reference:
            UnCert-CoT (arXiv:2503.15341) - Line boundary measurement approach

        Example:
            >>> # High confidence case
            >>> peaked = np.array([10.0, 0.0, 0.0, 0.0])
            >>> pd_peaked = EntropyCalculator.probability_differential(peaked)
            >>> assert pd_peaked < 0.1  # Close to 0
            >>>
            >>> # Low confidence case
            >>> uniform = np.ones(100)
            >>> pd_uniform = EntropyCalculator.probability_differential(uniform)
            >>> assert pd_uniform > 0.9  # Close to 1
        """
        if logits.size == 0:
            raise ValueError("Logits array cannot be empty")

        if not np.all(np.isfinite(logits)):
            raise ValueError("Logits contains non-finite values (inf or nan)")

        # Convert to probabilities
        logits_stable = logits - np.max(logits)
        probs = np.exp(logits_stable) / np.sum(np.exp(logits_stable))

        # Compute 1 - max(P)
        max_prob = np.max(probs)
        return float(1.0 - max_prob)

    @staticmethod
    def top_k_entropy(logits: np.ndarray, k: int = 10) -> float:
        """
        Compute entropy over top-k tokens only.

        H_k = -Σ p(x) log₂ p(x) for x in top-k tokens

        This variant focuses on the most likely tokens, ignoring the long tail
        of low-probability tokens. Useful for analyzing uncertainty among
        plausible predictions.

        Args:
            logits: Array of shape (vocab_size,) containing unnormalized log-probabilities
            k: Number of top tokens to consider (default: 10)

        Returns:
            Entropy computed over top-k tokens in bits

        Raises:
            ValueError: If k <= 0, k > vocab_size, or logits is invalid

        Example:
            >>> logits = np.random.randn(32000)
            >>> # Entropy over top-10 tokens
            >>> h_10 = EntropyCalculator.top_k_entropy(logits, k=10)
            >>> # Entropy over top-100 tokens
            >>> h_100 = EntropyCalculator.top_k_entropy(logits, k=100)
            >>> # More tokens => potentially higher entropy
            >>> assert h_100 >= h_10
        """
        if logits.size == 0:
            raise ValueError("Logits array cannot be empty")

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        if k > logits.shape[0]:
            raise ValueError(f"k ({k}) cannot exceed vocabulary size ({logits.shape[0]})")

        # Get top-k indices
        top_k_indices = np.argpartition(logits, -k)[-k:]
        top_k_logits = logits[top_k_indices]

        # Compute entropy over top-k logits
        return EntropyCalculator.shannon_entropy(top_k_logits)


# Convenience functions (functional interface)

def shannon_entropy(logits: np.ndarray) -> float:
    """Compute Shannon entropy. See EntropyCalculator.shannon_entropy for details."""
    return EntropyCalculator.shannon_entropy(logits)


def normalized_entropy(logits: np.ndarray) -> float:
    """Compute normalized entropy. See EntropyCalculator.normalized_entropy for details."""
    return EntropyCalculator.normalized_entropy(logits)


def probability_differential(logits: np.ndarray) -> float:
    """Compute probability differential. See EntropyCalculator.probability_differential for details."""
    return EntropyCalculator.probability_differential(logits)


def top_k_entropy(logits: np.ndarray, k: int = 10) -> float:
    """Compute top-k entropy. See EntropyCalculator.top_k_entropy for details."""
    return EntropyCalculator.top_k_entropy(logits, k)


# Utility functions

def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Compute softmax (normalized probabilities) from logits.

    Args:
        logits: Array of unnormalized log-probabilities

    Returns:
        Probability distribution (sums to 1)

    Example:
        >>> logits = np.array([2.0, 1.0, 0.1])
        >>> probs = softmax(logits)
        >>> assert np.isclose(np.sum(probs), 1.0)
    """
    logits_stable = logits - np.max(logits)
    exp_logits = np.exp(logits_stable)
    return exp_logits / np.sum(exp_logits)


def get_top_k_predictions(
    logits: np.ndarray,
    k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get top-k token indices and their probabilities.

    Args:
        logits: Array of shape (vocab_size,)
        k: Number of top predictions to return

    Returns:
        Tuple of (indices, probabilities), both arrays of length k
        sorted by descending probability

    Example:
        >>> logits = np.random.randn(1000)
        >>> indices, probs = get_top_k_predictions(logits, k=5)
        >>> assert len(indices) == 5
        >>> assert len(probs) == 5
        >>> assert np.all(np.diff(probs) <= 0)  # Descending order
    """
    if k <= 0 or k > len(logits):
        raise ValueError(f"k must be in range (0, {len(logits)}], got {k}")

    probs = softmax(logits)
    top_k_indices = np.argsort(probs)[-k:][::-1]  # Descending order
    top_k_probs = probs[top_k_indices]

    return top_k_indices, top_k_probs
