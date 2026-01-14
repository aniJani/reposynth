"""
Evaluation Framework for RepoSynth CCE Research.

This module provides:
- BenchmarkDataset: Structured evaluation examples
- EvaluationMetrics: 8 metrics for assessment
- BaselineRunner: Run baseline methods
- ExperimentRunner: Full experiment orchestration
- StatisticalAnalysis: Significance testing

Phase 4, Week 8-9: Evaluation Framework

Metrics implemented:
1. answer_correctness - LLM-as-judge scoring
2. answer_completeness - Semantic similarity to ground truth
3. hallucination_rate - Fraction of unsupported claims
4. context_precision - Retrieved vs ground truth files
5. context_recall - Coverage of ground truth files
6. token_efficiency - Tokens saved vs baseline
7. spike_precision - Accuracy of uncertainty detection
8. spike_recall - Coverage of uncertainty detection
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
    ExperimentConfig,
    EvaluationRetriever,
)

from .stats import (
    StatisticalAnalysis,
)

# Import benchmark generator utilities if available
try:
    from .benchmark_generator import generate_full_benchmark, get_mock_codebase
    _HAS_BENCHMARK_GENERATOR = True
except ImportError:
    _HAS_BENCHMARK_GENERATOR = False

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
    "ExperimentConfig",
    "EvaluationRetriever",
    # Stats
    "StatisticalAnalysis",
]

# Add benchmark generator exports if available
if _HAS_BENCHMARK_GENERATOR:
    __all__.extend(["generate_full_benchmark", "get_mock_codebase"])

__version__ = '0.1.0'
