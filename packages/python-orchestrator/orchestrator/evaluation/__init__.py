"""
Evaluation Module for Adaptive Code Generation Research.

This package provides:
- BenchmarkDataset: Structured dataset for evaluation
- EvaluationMetrics: 8 metrics for comprehensive evaluation
- BaselineRunner: Run baselines for comparison
- ExperimentRunner: Full experiment orchestration

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

__all__ = [
    # Core classes
    'BenchmarkDataset',
    'BenchmarkExample',
    'EvaluationMetrics',
    'EvaluationResult',
    # Runners
    'BaselineRunner',
    'ExperimentRunner',
    # Utilities
    'create_benchmark',
    'load_benchmark',
    # Benchmark generator
    'generate_full_benchmark',
    'get_mock_codebase',
]

__version__ = '0.1.0'

# Lazy imports
def __getattr__(name):
    if name in ['BenchmarkDataset', 'BenchmarkExample', 'create_benchmark', 'load_benchmark']:
        from .benchmark import BenchmarkDataset, BenchmarkExample, create_benchmark, load_benchmark
        globals().update({
            'BenchmarkDataset': BenchmarkDataset,
            'BenchmarkExample': BenchmarkExample,
            'create_benchmark': create_benchmark,
            'load_benchmark': load_benchmark,
        })
        return globals()[name]

    if name in ['EvaluationMetrics', 'EvaluationResult']:
        from .metrics import EvaluationMetrics, EvaluationResult
        globals().update({
            'EvaluationMetrics': EvaluationMetrics,
            'EvaluationResult': EvaluationResult,
        })
        return globals()[name]

    if name in ['BaselineRunner', 'ExperimentRunner']:
        from .runner import BaselineRunner, ExperimentRunner
        globals().update({
            'BaselineRunner': BaselineRunner,
            'ExperimentRunner': ExperimentRunner,
        })
        return globals()[name]

    if name in ['generate_full_benchmark', 'get_mock_codebase']:
        from .benchmark_generator import generate_full_benchmark, get_mock_codebase
        globals().update({
            'generate_full_benchmark': generate_full_benchmark,
            'get_mock_codebase': get_mock_codebase,
        })
        return globals()[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
