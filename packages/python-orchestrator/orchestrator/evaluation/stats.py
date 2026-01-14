"""
Statistical Analysis for CCE Evaluation.

Provides significance testing for comparing methods:
- Paired t-tests
- Effect sizes (Cohen's d)
- Confidence intervals
- Summary tables
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats


@dataclass
class ComparisonResult:
    """Result of comparing two methods."""
    method_a: str
    method_b: str
    metric: str

    # Descriptive stats
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    n_samples: int

    # Statistical tests
    t_statistic: float
    p_value: float
    cohens_d: float

    # Confidence interval for difference
    ci_lower: float
    ci_upper: float
    confidence: float

    @property
    def is_significant(self) -> bool:
        """Check if difference is statistically significant at p < 0.05."""
        return self.p_value < 0.05

    @property
    def effect_size_interpretation(self) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(self.cohens_d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method_a": self.method_a,
            "method_b": self.method_b,
            "metric": self.metric,
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "std_a": round(self.std_a, 4),
            "std_b": round(self.std_b, 4),
            "n_samples": self.n_samples,
            "t_statistic": round(self.t_statistic, 4),
            "p_value": round(self.p_value, 6),
            "cohens_d": round(self.cohens_d, 4),
            "effect_size": self.effect_size_interpretation,
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "significant": self.is_significant,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        direction = "better" if self.mean_a > self.mean_b else "worse"
        sig = "significant" if self.is_significant else "not significant"
        return (
            f"{self.metric}: {self.method_a} vs {self.method_b}\n"
            f"  {self.method_a}: {self.mean_a:.3f} ± {self.std_a:.3f}\n"
            f"  {self.method_b}: {self.mean_b:.3f} ± {self.std_b:.3f}\n"
            f"  Difference: {self.mean_a - self.mean_b:+.3f} ({direction})\n"
            f"  t({self.n_samples-1}) = {self.t_statistic:.3f}, p = {self.p_value:.4f} ({sig})\n"
            f"  Cohen's d = {self.cohens_d:.3f} ({self.effect_size_interpretation} effect)\n"
            f"  95% CI: [{self.ci_lower:.3f}, {self.ci_upper:.3f}]"
        )


class StatisticalAnalysis:
    """
    Statistical analysis utilities for comparing evaluation results.
    """

    def __init__(self, confidence: float = 0.95):
        """
        Initialize statistical analysis.

        Args:
            confidence: Confidence level for intervals (default 0.95)
        """
        self.confidence = confidence

    def paired_t_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
    ) -> Tuple[float, float]:
        """
        Perform paired t-test.

        Args:
            scores_a: Scores from method A
            scores_b: Scores from method B (same examples)

        Returns:
            Tuple of (t_statistic, p_value)
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Score lists must have same length for paired test")
        if len(scores_a) < 2:
            return 0.0, 1.0

        t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
        return float(t_stat), float(p_val)

    def cohens_d(
        self,
        scores_a: List[float],
        scores_b: List[float],
    ) -> float:
        """
        Compute Cohen's d effect size for paired samples.

        Uses the standard deviation of differences for paired data.

        Args:
            scores_a: Scores from method A
            scores_b: Scores from method B

        Returns:
            Cohen's d effect size
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Score lists must have same length")
        if len(scores_a) < 2:
            return 0.0

        a = np.array(scores_a)
        b = np.array(scores_b)
        differences = a - b

        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)

        if std_diff == 0:
            return 0.0 if mean_diff == 0 else float("inf")

        return float(mean_diff / std_diff)

    def confidence_interval(
        self,
        scores: List[float],
        confidence: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for mean.

        Args:
            scores: List of scores
            confidence: Confidence level (default: instance default)

        Returns:
            Tuple of (lower, upper) bounds
        """
        if confidence is None:
            confidence = self.confidence

        if len(scores) < 2:
            mean = np.mean(scores) if scores else 0.0
            return (mean, mean)

        arr = np.array(scores)
        mean = np.mean(arr)
        std_err = stats.sem(arr)

        # t-distribution for small samples
        df = len(arr) - 1
        t_crit = stats.t.ppf((1 + confidence) / 2, df)
        margin = t_crit * std_err

        return (float(mean - margin), float(mean + margin))

    def confidence_interval_difference(
        self,
        scores_a: List[float],
        scores_b: List[float],
        confidence: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for difference of means (paired).

        Args:
            scores_a: Scores from method A
            scores_b: Scores from method B
            confidence: Confidence level

        Returns:
            Tuple of (lower, upper) bounds for mean difference
        """
        if confidence is None:
            confidence = self.confidence

        if len(scores_a) != len(scores_b):
            raise ValueError("Score lists must have same length")

        differences = [a - b for a, b in zip(scores_a, scores_b)]
        return self.confidence_interval(differences, confidence)

    def compare_methods(
        self,
        method_a: str,
        scores_a: List[float],
        method_b: str,
        scores_b: List[float],
        metric: str,
    ) -> ComparisonResult:
        """
        Full comparison between two methods.

        Args:
            method_a: Name of first method
            scores_a: Scores from first method
            method_b: Name of second method
            scores_b: Scores from second method
            metric: Name of the metric being compared

        Returns:
            ComparisonResult with all statistics
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Score lists must have same length for paired comparison")

        a = np.array(scores_a)
        b = np.array(scores_b)

        t_stat, p_val = self.paired_t_test(scores_a, scores_b)
        d = self.cohens_d(scores_a, scores_b)
        ci_lower, ci_upper = self.confidence_interval_difference(scores_a, scores_b)

        return ComparisonResult(
            method_a=method_a,
            method_b=method_b,
            metric=metric,
            mean_a=float(np.mean(a)),
            mean_b=float(np.mean(b)),
            std_a=float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            std_b=float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
            n_samples=len(a),
            t_statistic=t_stat,
            p_value=p_val,
            cohens_d=d,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence=self.confidence,
        )

    def compare_all_to_baseline(
        self,
        baseline_name: str,
        baseline_scores: Dict[str, List[float]],
        method_scores: Dict[str, Dict[str, List[float]]],
    ) -> Dict[str, List[ComparisonResult]]:
        """
        Compare all methods to a baseline across all metrics.

        Args:
            baseline_name: Name of baseline method
            baseline_scores: Dict mapping metric -> scores for baseline
            method_scores: Dict mapping method -> metric -> scores

        Returns:
            Dict mapping method -> list of ComparisonResults
        """
        results = {}

        for method_name, scores_by_metric in method_scores.items():
            if method_name == baseline_name:
                continue

            method_results = []
            for metric, method_vals in scores_by_metric.items():
                if metric in baseline_scores:
                    baseline_vals = baseline_scores[metric]
                    if len(method_vals) == len(baseline_vals):
                        comparison = self.compare_methods(
                            method_name, method_vals,
                            baseline_name, baseline_vals,
                            metric,
                        )
                        method_results.append(comparison)

            results[method_name] = method_results

        return results

    def significance_summary(
        self,
        comparisons: List[ComparisonResult],
    ) -> str:
        """
        Generate summary table of significance tests.

        Args:
            comparisons: List of comparison results

        Returns:
            Formatted summary string
        """
        if not comparisons:
            return "No comparisons to summarize."

        lines = ["=" * 80]
        lines.append("STATISTICAL SIGNIFICANCE SUMMARY")
        lines.append("=" * 80)
        lines.append(
            f"{'Metric':<20} {'Method A':<15} {'Method B':<15} "
            f"{'Δ Mean':>10} {'p-value':>10} {'d':>8} {'Sig?':>6}"
        )
        lines.append("-" * 80)

        for comp in comparisons:
            diff = comp.mean_a - comp.mean_b
            sig = "Yes" if comp.is_significant else "No"
            lines.append(
                f"{comp.metric:<20} {comp.method_a:<15} {comp.method_b:<15} "
                f"{diff:>+10.3f} {comp.p_value:>10.4f} {comp.cohens_d:>8.2f} {sig:>6}"
            )

        lines.append("=" * 80)

        # Summary counts
        n_sig = sum(1 for c in comparisons if c.is_significant)
        lines.append(f"\nSignificant results: {n_sig}/{len(comparisons)} (p < 0.05)")

        return "\n".join(lines)

    def to_dataframe(self, comparisons: List[ComparisonResult]):
        """
        Convert comparisons to pandas DataFrame.

        Args:
            comparisons: List of comparison results

        Returns:
            pandas DataFrame
        """
        try:
            import pandas as pd
            rows = [c.to_dict() for c in comparisons]
            return pd.DataFrame(rows)
        except ImportError:
            raise ImportError("pandas required for DataFrame output")
