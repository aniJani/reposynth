"""
Unit tests for entropy calculator module.

Tests cover:
- Shannon entropy calculation
- Normalized entropy
- Probability differential
- Top-k entropy
- Edge cases and error handling
- Known distributions
"""

import unittest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "python-orchestrator"))

from orchestrator.entropy.calculator import (
    EntropyCalculator,
    shannon_entropy,
    normalized_entropy,
    probability_differential,
    top_k_entropy,
    softmax,
    get_top_k_predictions
)


class TestEntropyCalculator(unittest.TestCase):
    """Test cases for EntropyCalculator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.RandomState(42)  # Reproducible random numbers

    # Shannon Entropy Tests

    def test_shannon_entropy_uniform_distribution(self):
        """Test Shannon entropy on uniform distribution (maximum entropy)."""
        # Uniform distribution over 100 items
        logits = np.ones(100)
        h = EntropyCalculator.shannon_entropy(logits)

        # Expected: log2(100) ≈ 6.644 bits
        expected = np.log2(100)
        assert_almost_equal(h, expected, decimal=5)

    def test_shannon_entropy_peaked_distribution(self):
        """Test Shannon entropy on peaked distribution (low entropy)."""
        # One very high logit, rest very low
        logits = np.array([10.0, 0.1, 0.1, 0.1, 0.1])
        h = EntropyCalculator.shannon_entropy(logits)

        # Should be close to 0 (very certain)
        self.assertLess(h, 0.5)

    def test_shannon_entropy_binary_distribution(self):
        """Test Shannon entropy on binary distribution."""
        # Equal probability binary choice
        logits = np.array([0.0, 0.0])
        h = EntropyCalculator.shannon_entropy(logits)

        # Expected: log2(2) = 1 bit
        assert_almost_equal(h, 1.0, decimal=5)

    def test_shannon_entropy_deterministic(self):
        """Test Shannon entropy on deterministic distribution."""
        # One logit extremely high
        logits = np.array([100.0, 0.0, 0.0, 0.0])
        h = EntropyCalculator.shannon_entropy(logits)

        # Should be very close to 0
        self.assertLess(h, 0.01)

    def test_shannon_entropy_numerical_stability(self):
        """Test Shannon entropy with extreme values."""
        # Very large positive and negative logits
        logits = np.array([1000.0, -1000.0, 500.0, -500.0])
        h = EntropyCalculator.shannon_entropy(logits)

        # Should not overflow or produce nan
        self.assertTrue(np.isfinite(h))
        self.assertGreaterEqual(h, 0)

    # Normalized Entropy Tests

    def test_normalized_entropy_range(self):
        """Test that normalized entropy is in [0, 1]."""
        for _ in range(10):
            logits = self.rng.randn(100)
            h_norm = EntropyCalculator.normalized_entropy(logits)

            self.assertGreaterEqual(h_norm, 0.0)
            self.assertLessEqual(h_norm, 1.0)

    def test_normalized_entropy_uniform(self):
        """Test normalized entropy on uniform distribution (should be 1.0)."""
        logits = np.ones(1000)
        h_norm = EntropyCalculator.normalized_entropy(logits)

        assert_almost_equal(h_norm, 1.0, decimal=5)

    def test_normalized_entropy_peaked(self):
        """Test normalized entropy on peaked distribution (should be ~0)."""
        logits = np.array([100.0] + [0.0] * 999)
        h_norm = EntropyCalculator.normalized_entropy(logits)

        self.assertLess(h_norm, 0.01)

    # Probability Differential Tests

    def test_probability_differential_certain(self):
        """Test probability differential when model is certain."""
        # Very peaked distribution
        logits = np.array([10.0, 0.0, 0.0, 0.0])
        pd = EntropyCalculator.probability_differential(logits)

        # Should be close to 0 (1 - max_prob, where max_prob ≈ 1)
        self.assertLess(pd, 0.01)

    def test_probability_differential_uncertain(self):
        """Test probability differential when model is uncertain."""
        # Uniform distribution
        logits = np.ones(100)
        pd = EntropyCalculator.probability_differential(logits)

        # Should be close to 1 - 1/100 = 0.99
        expected = 1.0 - 1.0/100
        assert_almost_equal(pd, expected, decimal=5)

    def test_probability_differential_binary(self):
        """Test probability differential on binary choice."""
        # 50-50 split
        logits = np.array([0.0, 0.0])
        pd = EntropyCalculator.probability_differential(logits)

        # Should be 1 - 0.5 = 0.5
        assert_almost_equal(pd, 0.5, decimal=5)

    # Top-K Entropy Tests

    def test_top_k_entropy_basic(self):
        """Test top-k entropy computation."""
        logits = self.rng.randn(1000)
        h_10 = EntropyCalculator.top_k_entropy(logits, k=10)

        # Entropy should be positive
        self.assertGreater(h_10, 0)

        # Should be less than log2(10)
        self.assertLess(h_10, np.log2(10))

    def test_top_k_entropy_increasing_k(self):
        """Test that entropy increases (or stays same) with larger k."""
        logits = self.rng.randn(1000)

        h_10 = EntropyCalculator.top_k_entropy(logits, k=10)
        h_50 = EntropyCalculator.top_k_entropy(logits, k=50)
        h_100 = EntropyCalculator.top_k_entropy(logits, k=100)

        # More tokens => higher (or equal) entropy
        self.assertGreaterEqual(h_50, h_10 * 0.99)  # Allow tiny numerical error
        self.assertGreaterEqual(h_100, h_50 * 0.99)

    def test_top_k_entropy_k_equals_vocab(self):
        """Test top-k entropy when k equals vocabulary size."""
        logits = self.rng.randn(100)

        h_all = EntropyCalculator.top_k_entropy(logits, k=100)
        h_shannon = EntropyCalculator.shannon_entropy(logits)

        # Should be identical
        assert_almost_equal(h_all, h_shannon, decimal=5)

    # Edge Cases and Error Handling

    def test_empty_logits_raises(self):
        """Test that empty logits raise ValueError."""
        empty_logits = np.array([])

        with self.assertRaises(ValueError):
            EntropyCalculator.shannon_entropy(empty_logits)

        with self.assertRaises(ValueError):
            EntropyCalculator.normalized_entropy(empty_logits)

        with self.assertRaises(ValueError):
            EntropyCalculator.probability_differential(empty_logits)

    def test_invalid_logits_raises(self):
        """Test that invalid logits (inf/nan) raise ValueError."""
        invalid_logits = np.array([1.0, 2.0, np.inf, 3.0])

        with self.assertRaises(ValueError):
            EntropyCalculator.shannon_entropy(invalid_logits)

        nan_logits = np.array([1.0, np.nan, 2.0])

        with self.assertRaises(ValueError):
            EntropyCalculator.probability_differential(nan_logits)

    def test_top_k_invalid_k_raises(self):
        """Test that invalid k values raise ValueError."""
        logits = self.rng.randn(100)

        # k = 0
        with self.assertRaises(ValueError):
            EntropyCalculator.top_k_entropy(logits, k=0)

        # Negative k
        with self.assertRaises(ValueError):
            EntropyCalculator.top_k_entropy(logits, k=-5)

        # k > vocab_size
        with self.assertRaises(ValueError):
            EntropyCalculator.top_k_entropy(logits, k=200)

    # Functional Interface Tests

    def test_functional_interface_shannon(self):
        """Test functional interface matches class method."""
        logits = self.rng.randn(100)

        h_class = EntropyCalculator.shannon_entropy(logits)
        h_func = shannon_entropy(logits)

        assert_almost_equal(h_class, h_func)

    def test_functional_interface_normalized(self):
        """Test functional interface for normalized entropy."""
        logits = self.rng.randn(100)

        h_class = EntropyCalculator.normalized_entropy(logits)
        h_func = normalized_entropy(logits)

        assert_almost_equal(h_class, h_func)

    def test_functional_interface_prob_diff(self):
        """Test functional interface for probability differential."""
        logits = self.rng.randn(100)

        pd_class = EntropyCalculator.probability_differential(logits)
        pd_func = probability_differential(logits)

        assert_almost_equal(pd_class, pd_func)

    def test_functional_interface_top_k(self):
        """Test functional interface for top-k entropy."""
        logits = self.rng.randn(100)

        h_class = EntropyCalculator.top_k_entropy(logits, k=10)
        h_func = top_k_entropy(logits, k=10)

        assert_almost_equal(h_class, h_func)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.RandomState(42)

    def test_softmax_sums_to_one(self):
        """Test that softmax output sums to 1."""
        logits = self.rng.randn(100)
        probs = softmax(logits)

        assert_almost_equal(np.sum(probs), 1.0, decimal=10)

    def test_softmax_all_positive(self):
        """Test that softmax output is all positive."""
        logits = self.rng.randn(100)
        probs = softmax(logits)

        self.assertTrue(np.all(probs >= 0))

    def test_softmax_numerical_stability(self):
        """Test softmax with extreme values."""
        # Very large positive values
        logits = np.array([1000.0, 999.0, 998.0])
        probs = softmax(logits)

        # Should not overflow
        self.assertTrue(np.all(np.isfinite(probs)))
        assert_almost_equal(np.sum(probs), 1.0, decimal=5)

    def test_get_top_k_predictions_descending(self):
        """Test that top-k predictions are in descending order."""
        logits = self.rng.randn(1000)
        indices, probs = get_top_k_predictions(logits, k=10)

        # Check descending order
        for i in range(len(probs) - 1):
            self.assertGreaterEqual(probs[i], probs[i+1])

    def test_get_top_k_predictions_correct_size(self):
        """Test that top-k returns correct number of predictions."""
        logits = self.rng.randn(1000)

        for k in [1, 5, 10, 50, 100]:
            indices, probs = get_top_k_predictions(logits, k=k)
            self.assertEqual(len(indices), k)
            self.assertEqual(len(probs), k)

    def test_get_top_k_predictions_probabilities_sum(self):
        """Test that top-k probabilities sum to less than or equal to 1."""
        logits = self.rng.randn(1000)
        indices, probs = get_top_k_predictions(logits, k=100)

        # Sum of top-k probs should be <= 1
        self.assertLessEqual(np.sum(probs), 1.0 + 1e-10)  # Allow tiny numerical error

    def test_get_top_k_invalid_k(self):
        """Test that invalid k raises ValueError."""
        logits = self.rng.randn(100)

        with self.assertRaises(ValueError):
            get_top_k_predictions(logits, k=0)

        with self.assertRaises(ValueError):
            get_top_k_predictions(logits, k=200)


class TestKnownDistributions(unittest.TestCase):
    """Test entropy calculations on known probability distributions."""

    def test_uniform_distribution_entropy(self):
        """Test that uniform distribution has maximum entropy."""
        # Uniform over N items has entropy log2(N)
        for n in [2, 10, 100, 1000]:
            logits = np.ones(n)
            h = shannon_entropy(logits)
            expected = np.log2(n)

            assert_almost_equal(h, expected, decimal=5,
                              err_msg=f"Failed for n={n}")

    def test_biased_coin_entropy(self):
        """Test entropy of biased coin flip."""
        # Bias p=0.7, q=0.3
        # H = -0.7*log2(0.7) - 0.3*log2(0.3) ≈ 0.881 bits
        p = 0.7
        logits = np.array([np.log(p), np.log(1-p)])
        h = shannon_entropy(logits)

        expected = -p * np.log2(p) - (1-p) * np.log2(1-p)
        assert_almost_equal(h, expected, decimal=3)

    def test_four_way_equal_choice(self):
        """Test entropy of four equal choices."""
        # H = log2(4) = 2 bits
        logits = np.zeros(4)  # Equal probabilities
        h = shannon_entropy(logits)

        assert_almost_equal(h, 2.0, decimal=5)


class TestPerformance(unittest.TestCase):
    """Test that entropy calculations are fast enough."""

    def test_shannon_entropy_performance(self):
        """Test Shannon entropy performance (<1ms for vocab_size=32000)."""
        import time

        logits = np.random.randn(32000)  # CodeLlama vocab size

        start = time.perf_counter()
        for _ in range(100):
            _ = shannon_entropy(logits)
        end = time.perf_counter()

        avg_time_ms = (end - start) * 1000 / 100

        # Should be much less than 1ms per call
        self.assertLess(avg_time_ms, 1.0,
                       msg=f"Average time: {avg_time_ms:.3f}ms (target: <1ms)")


if __name__ == '__main__':
    unittest.main()
