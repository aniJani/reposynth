"""
Tests for the evaluation module.

Run with: pytest tests/evaluation/test_evaluation.py -v
"""

import pytest
from pathlib import Path


class TestBenchmark:
    """Tests for benchmark module."""

    def test_imports(self):
        """Test that all benchmark classes can be imported."""
        from orchestrator.evaluation import (
            BenchmarkExample,
            BenchmarkDataset,
            Difficulty,
            Category,
            create_benchmark,
            create_sample_benchmark,
        )
        assert BenchmarkExample is not None
        assert BenchmarkDataset is not None

    def test_create_sample_benchmark(self):
        """Test sample benchmark creation."""
        from orchestrator.evaluation import create_sample_benchmark

        benchmark = create_sample_benchmark()
        assert len(benchmark) == 3
        assert benchmark.name == "sample_benchmark"

    def test_benchmark_filtering(self):
        """Test filtering by difficulty and category."""
        from orchestrator.evaluation import create_sample_benchmark, Difficulty, Category

        benchmark = create_sample_benchmark()

        # Filter by difficulty
        easy = benchmark.filter(difficulty=Difficulty.EASY)
        assert len(easy) == 2

        # Filter by category
        arch = benchmark.filter(category=Category.ARCHITECTURE)
        assert len(arch) == 1

    def test_benchmark_validation(self):
        """Test benchmark validation."""
        from orchestrator.evaluation import create_sample_benchmark

        benchmark = create_sample_benchmark()
        errors = benchmark.validate()
        assert len(errors) == 0

    def test_benchmark_statistics(self):
        """Test benchmark statistics."""
        from orchestrator.evaluation import create_sample_benchmark

        benchmark = create_sample_benchmark()
        stats = benchmark.get_statistics()

        assert stats["total_examples"] == 3
        assert "by_difficulty" in stats
        assert "by_category" in stats


class TestMetrics:
    """Tests for metrics module."""

    def test_imports(self):
        """Test that metrics can be imported."""
        from orchestrator.evaluation import EvaluationMetrics, EvaluationResult
        assert EvaluationMetrics is not None
        assert EvaluationResult is not None

    def test_context_precision(self):
        """Test context precision calculation."""
        from orchestrator.evaluation import EvaluationMetrics

        metrics = EvaluationMetrics()

        # Perfect precision
        prec = metrics.context_precision(
            ["a.py", "b.py"],
            ["a.py", "b.py"]
        )
        assert prec == 1.0

        # 50% precision
        prec = metrics.context_precision(
            ["a.py", "c.py"],
            ["a.py", "b.py"]
        )
        assert prec == 0.5

        # 0% precision
        prec = metrics.context_precision(
            ["x.py", "y.py"],
            ["a.py", "b.py"]
        )
        assert prec == 0.0

    def test_context_recall(self):
        """Test context recall calculation."""
        from orchestrator.evaluation import EvaluationMetrics

        metrics = EvaluationMetrics()

        # Perfect recall
        rec = metrics.context_recall(
            ["a.py", "b.py"],
            ["a.py", "b.py"]
        )
        assert rec == 1.0

        # 50% recall
        rec = metrics.context_recall(
            ["a.py"],
            ["a.py", "b.py"]
        )
        assert rec == 0.5

    def test_token_efficiency(self):
        """Test token efficiency calculation."""
        from orchestrator.evaluation import EvaluationMetrics

        metrics = EvaluationMetrics()

        # 75% efficiency (used only 25% of baseline)
        eff = metrics.token_efficiency(250, 1000)
        assert eff == 0.75

        # 0% efficiency (used all)
        eff = metrics.token_efficiency(1000, 1000)
        assert eff == 0.0

    def test_spike_precision(self):
        """Test spike precision calculation."""
        from orchestrator.evaluation import EvaluationMetrics

        metrics = EvaluationMetrics()

        # Perfect precision (all detections match)
        prec = metrics.spike_precision(
            detected_positions=[10, 20, 30],
            ground_truth_positions=[10, 20, 30],
            tolerance=3
        )
        assert prec == 1.0

        # Partial precision (1 of 2 match)
        prec = metrics.spike_precision(
            detected_positions=[10, 50],
            ground_truth_positions=[12, 20],
            tolerance=3
        )
        assert prec == 0.5

    def test_spike_recall(self):
        """Test spike recall calculation."""
        from orchestrator.evaluation import EvaluationMetrics

        metrics = EvaluationMetrics()

        # Perfect recall
        rec = metrics.spike_recall(
            detected_positions=[10, 20, 30],
            ground_truth_positions=[10, 20, 30],
            tolerance=3
        )
        assert rec == 1.0

        # 50% recall (1 of 2 ground truths found)
        rec = metrics.spike_recall(
            detected_positions=[10],
            ground_truth_positions=[10, 20],
            tolerance=3
        )
        assert rec == 0.5

    def test_evaluation_result(self):
        """Test EvaluationResult methods."""
        from orchestrator.evaluation import EvaluationResult

        result = EvaluationResult(
            example_id="test_001",
            method="cce_adaptive",
            answer_correctness=0.8,
            answer_completeness=0.9,
            context_precision=0.75,
            context_recall=1.0,
            token_efficiency=0.5,
        )

        # F1 context
        f1 = result.get_f1_context()
        assert 0.8 < f1 < 0.9  # F1 of 0.75 and 1.0

        # Composite score
        score = result.get_composite_score()
        assert 0.0 < score < 1.0


class TestStats:
    """Tests for statistical analysis module."""

    def test_imports(self):
        """Test that stats can be imported."""
        from orchestrator.evaluation import StatisticalAnalysis
        assert StatisticalAnalysis is not None

    def test_paired_t_test(self):
        """Test paired t-test."""
        from orchestrator.evaluation.stats import StatisticalAnalysis

        stats = StatisticalAnalysis()

        # Same values should give p=1
        scores_a = [0.5, 0.6, 0.7, 0.8, 0.9]
        scores_b = [0.5, 0.6, 0.7, 0.8, 0.9]
        t_stat, p_val = stats.paired_t_test(scores_a, scores_b)
        assert p_val == 1.0

        # Different values should give p < 1
        scores_b = [0.4, 0.5, 0.6, 0.7, 0.8]
        t_stat, p_val = stats.paired_t_test(scores_a, scores_b)
        assert p_val < 1.0

    def test_cohens_d(self):
        """Test Cohen's d effect size."""
        from orchestrator.evaluation.stats import StatisticalAnalysis

        stats = StatisticalAnalysis()

        # No difference
        d = stats.cohens_d([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        assert d == 0.0

        # Large difference
        d = stats.cohens_d([0.9, 0.9, 0.9], [0.1, 0.1, 0.1])
        assert d > 0.8  # Large effect

    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        from orchestrator.evaluation.stats import StatisticalAnalysis

        stats = StatisticalAnalysis(confidence=0.95)

        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        lower, upper = stats.confidence_interval(scores)

        # Mean is 0.7
        assert lower < 0.7 < upper
        assert lower > 0.0
        assert upper < 1.0


class TestRunner:
    """Tests for experiment runner."""

    def test_imports(self):
        """Test that runner can be imported."""
        from orchestrator.evaluation import (
            BaselineRunner,
            ExperimentRunner,
            BaselineMethod,
        )
        from orchestrator.evaluation.runner import EvaluationRetriever

        assert BaselineRunner is not None
        assert BaselineMethod is not None
        assert EvaluationRetriever is not None

    def test_baseline_methods(self):
        """Test all baseline methods are defined."""
        from orchestrator.evaluation import BaselineMethod

        methods = list(BaselineMethod)
        assert len(methods) == 7

        # Check specific methods exist
        assert BaselineMethod.NO_CONTEXT in methods
        assert BaselineMethod.BM25 in methods
        assert BaselineMethod.FULL_CONTEXT in methods
        assert BaselineMethod.RANDOM in methods
        assert BaselineMethod.EMBEDDING in methods
        assert BaselineMethod.REPOSYNTH_BASE in methods
        assert BaselineMethod.UNCERT_COT in methods

    def test_evaluation_retriever(self):
        """Test EvaluationRetriever with mock documents."""
        from orchestrator.evaluation.runner import EvaluationRetriever

        documents = {
            "auth.py": "def login(): pass",
            "db.py": "def query(): pass",
            "api.py": "def get_user(): pass",
        }

        retriever = EvaluationRetriever(documents=documents)

        # Test simple retrieval
        results = retriever.retrieve("login auth", top_k=2)
        assert len(results) <= 2

        # Test random retrieval
        random_results = retriever.retrieve_random(top_k=2)
        assert len(random_results) == 2

        # Test full context
        all_results = retriever.retrieve_all()
        assert len(all_results) == 3


class TestIntegration:
    """Integration tests for full evaluation pipeline."""

    def test_full_evaluation_pipeline(self):
        """Test complete evaluation on sample benchmark."""
        from orchestrator.evaluation import (
            create_sample_benchmark,
            EvaluationMetrics,
            BaselineMethod,
        )
        from orchestrator.evaluation.runner import BaselineRunner, EvaluationRetriever

        # Setup
        documents = {
            "src/auth/jwt.py": "import jwt\ndef create_token(user_id): pass\ndef verify_token(token): pass",
            "src/api/routes.py": "@app.route('/login')\ndef login(): token = create_token(user_id)",
            "src/db/models.py": "class User: id = Column(Integer)",
            "src/config.py": "SECRET_KEY = os.environ.get('SECRET_KEY')",
        }

        benchmark = create_sample_benchmark()
        retriever = EvaluationRetriever(documents=documents)
        metrics = EvaluationMetrics()
        runner = BaselineRunner(retriever=retriever, documents=documents)

        # Run on first example
        example = benchmark[0]
        result = runner.run(example, BaselineMethod.BM25, max_tokens=50)

        # Verify result structure
        assert "answer" in result
        assert "retrieved_files" in result
        assert "tokens_used" in result
        assert "generation_time" in result

        # Evaluate
        eval_result = metrics.evaluate(
            example_id=example.id,
            generated_answer=result["answer"],
            ground_truth_answer=example.ground_truth_answer,
            retrieved_files=result["retrieved_files"],
            ground_truth_files=example.ground_truth_files,
            ground_truth_keywords=example.ground_truth_keywords,
            tokens_used=result["tokens_used"],
            baseline_tokens=1000,
            generation_time=result["generation_time"],
            method=BaselineMethod.BM25.value,
        )

        # Verify evaluation result
        assert eval_result.example_id == example.id
        assert eval_result.method == "bm25"
        assert 0.0 <= eval_result.answer_correctness <= 1.0
        assert 0.0 <= eval_result.context_precision <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
