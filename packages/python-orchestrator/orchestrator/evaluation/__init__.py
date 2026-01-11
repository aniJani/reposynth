"""
Evaluation Framework for RepoSynth CCE Research.

This module provides:
- BenchmarkDataset: Structured evaluation examples
- EvaluationMetrics: 8 metrics for assessment
- BaselineRunner: Run baseline methods
- ExperimentRunner: Full experiment orchestration
- StatisticalAnalysis: Significance testing
"""

from .benchmark import (
    BenchmarkExample,
    BenchmarkDataset,
    Difficulty,
    Category,
    create_benchmark,
    create_sample_benchmark,
)

from .metrics import (
    EvaluationMetrics,
    EvaluationResult,
)

from .runner import (
    BaselineRunner,
    ExperimentRunner,
    BaselineMethod,
)

from .stats import (
    StatisticalAnalysis,
)

__all__ = [
    # Benchmark
    "BenchmarkExample",
    "BenchmarkDataset",
    "Difficulty",
    "Category",
    "create_benchmark",
    "create_sample_benchmark",
    # Metrics
    "EvaluationMetrics",
    "EvaluationResult",
    # Runner
    "BaselineRunner",
    "ExperimentRunner",
    "BaselineMethod",
    # Stats
    "StatisticalAnalysis",
]
