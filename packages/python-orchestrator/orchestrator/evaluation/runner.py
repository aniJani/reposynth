"""
Experiment Runner for CCE Evaluation.

Provides:
- BaselineRunner: Run individual baseline methods
- ExperimentRunner: Full experiment orchestration
- EvaluationRetriever: Wrapper around RepoSynth retrieval
"""

import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Protocol
import numpy as np

from .benchmark import BenchmarkDataset, BenchmarkExample
from .metrics import EvaluationMetrics, EvaluationResult
from .stats import StatisticalAnalysis, ComparisonResult


class BaselineMethod(Enum):
    """Available baseline methods for comparison."""
    NO_CONTEXT = "no_context"           # Generate without any context
    BM25 = "bm25"                       # Keyword-based retrieval
    FULL_CONTEXT = "full_context"       # Include all documents
    RANDOM = "random"                   # Random file selection
    EMBEDDING = "embedding"             # Pure semantic search
    REPOSYNTH_BASE = "reposynth_base"   # RepoSynth hybrid (no entropy)
    UNCERT_COT = "uncert_cot"           # UnCert-CoT line boundary


class RetrieverProtocol(Protocol):
    """Protocol for retriever implementations."""

    def retrieve(self, query: str, top_k: int = 3, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents.

        Returns list of dicts with keys: source, content, score
        """
        ...


class EvaluationRetriever:
    """
    Wraps RepoSynth retrieval for evaluation framework.

    Integrates with hybrid_search from orchestrator.retriever.
    """

    def __init__(
        self,
        pack_dir: Optional[Path] = None,
        name_registry: Optional[Dict[str, Any]] = None,
        documents: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize retriever.

        Args:
            pack_dir: Path to pack directory with vectors.faiss
            name_registry: Pre-loaded name registry
            documents: Dict mapping file paths to content (for testing)
        """
        self.pack_dir = pack_dir
        self.name_registry = name_registry
        self.documents = documents or {}

        # Try to load registry if not provided
        if self.name_registry is None and self.pack_dir:
            registry_path = self.pack_dir / "name_registry.json"
            if registry_path.exists():
                with open(registry_path, "r") as f:
                    self.name_registry = json.load(f)

    def retrieve(self, query: str, top_k: int = 3, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using RepoSynth hybrid search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of dicts with source, content, score
        """
        results = []

        # Try RepoSynth hybrid search if available
        if self.pack_dir and self.name_registry:
            try:
                from orchestrator.retriever import hybrid_search
                raw_results = hybrid_search(
                    query, self.pack_dir, max_items=top_k, registry=self.name_registry
                )

                for r in raw_results:
                    file_path = r.get("file_path", "")
                    content = self._load_content(file_path)
                    results.append({
                        "source": file_path,
                        "content": content,
                        "score": self._compute_score(query, content),
                    })
            except Exception as e:
                print(f"Warning: hybrid_search failed: {e}")

        # Fallback to simple document matching
        if not results and self.documents:
            results = self._simple_retrieve(query, top_k)

        return results[:top_k]

    def retrieve_semantic(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Pure semantic (embedding) retrieval."""
        if not self.pack_dir:
            return []

        try:
            from orchestrator.retriever import semantic_search
            raw_results = semantic_search(query, self.pack_dir, max_items=top_k)
            return [
                {
                    "source": r.get("file_path", ""),
                    "content": self._load_content(r.get("file_path", "")),
                    "score": 0.5,  # semantic_search doesn't return scores
                }
                for r in raw_results
            ]
        except Exception:
            return []

    def retrieve_keyword(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Pure keyword (BM25-style) retrieval."""
        if not self.name_registry:
            return self._simple_retrieve(query, top_k)

        try:
            from orchestrator.retriever import retrieve_relevant_symbols
            raw_results = retrieve_relevant_symbols(query, self.name_registry, max_items=top_k)
            return [
                {
                    "source": r.get("file_path", ""),
                    "content": self._load_content(r.get("file_path", "")),
                    "score": 0.5,
                }
                for r in raw_results
            ]
        except Exception:
            return self._simple_retrieve(query, top_k)

    def retrieve_random(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """Random file selection."""
        if self.documents:
            keys = list(self.documents.keys())
            selected = random.sample(keys, min(top_k, len(keys)))
            return [
                {
                    "source": k,
                    "content": self.documents[k],
                    "score": 0.0,
                }
                for k in selected
            ]
        elif self.name_registry:
            all_files = list(set(
                v.get("file_path", "") for v in self.name_registry.values()
            ))
            selected = random.sample(all_files, min(top_k, len(all_files)))
            return [
                {
                    "source": f,
                    "content": self._load_content(f),
                    "score": 0.0,
                }
                for f in selected
            ]
        return []

    def retrieve_all(self) -> List[Dict[str, Any]]:
        """Return all documents (full context)."""
        if self.documents:
            return [
                {"source": k, "content": v, "score": 1.0}
                for k, v in self.documents.items()
            ]
        elif self.name_registry:
            seen = set()
            results = []
            for v in self.name_registry.values():
                fpath = v.get("file_path", "")
                if fpath not in seen:
                    results.append({
                        "source": fpath,
                        "content": self._load_content(fpath),
                        "score": 1.0,
                    })
                    seen.add(fpath)
            return results
        return []

    def _load_content(self, file_path: str) -> str:
        """Load file content."""
        if file_path in self.documents:
            return self.documents[file_path]

        # Try to load from pack_dir
        if self.pack_dir:
            full_path = self.pack_dir / file_path
            if full_path.exists():
                try:
                    return full_path.read_text(encoding="utf-8")
                except Exception:
                    pass

        return ""

    def _simple_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Simple keyword matching on documents."""
        if not self.documents:
            return []

        query_words = set(query.lower().split())
        scored = []

        for path, content in self.documents.items():
            content_lower = content.lower()
            path_lower = path.lower()
            score = sum(1 for w in query_words if w in content_lower or w in path_lower)
            if score > 0:
                scored.append((score, path, content))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [
            {"source": path, "content": content, "score": score / len(query_words)}
            for score, path, content in scored[:top_k]
        ]

    def _compute_score(self, query: str, content: str) -> float:
        """Compute simple relevance score."""
        if not content:
            return 0.0
        query_words = set(query.lower().split())
        content_lower = content.lower()
        matches = sum(1 for w in query_words if w in content_lower)
        return matches / len(query_words) if query_words else 0.0


class BaselineRunner:
    """
    Runs baseline methods for comparison.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        retriever: Optional[EvaluationRetriever] = None,
        documents: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize baseline runner.

        Args:
            model: Language model for generation
            tokenizer: Tokenizer for the model
            retriever: EvaluationRetriever instance
            documents: Dict mapping file paths to content
        """
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever or EvaluationRetriever(documents=documents)
        self.documents = documents or {}

    def run(
        self,
        example: BenchmarkExample,
        method: BaselineMethod,
        max_tokens: int = 100,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Run a baseline method on an example.

        Args:
            example: Benchmark example to evaluate
            method: Baseline method to use
            max_tokens: Maximum tokens to generate
            top_k: Number of files to retrieve

        Returns:
            Dict with answer, retrieved_files, tokens_used, generation_time
        """
        start_time = time.time()

        if method == BaselineMethod.NO_CONTEXT:
            result = self._run_no_context(example, max_tokens)
        elif method == BaselineMethod.BM25:
            result = self._run_bm25(example, max_tokens, top_k)
        elif method == BaselineMethod.FULL_CONTEXT:
            result = self._run_full_context(example, max_tokens)
        elif method == BaselineMethod.RANDOM:
            result = self._run_random(example, max_tokens, top_k)
        elif method == BaselineMethod.EMBEDDING:
            result = self._run_embedding(example, max_tokens, top_k)
        elif method == BaselineMethod.REPOSYNTH_BASE:
            result = self._run_reposynth_base(example, max_tokens, top_k)
        elif method == BaselineMethod.UNCERT_COT:
            result = self._run_uncert_cot(example, max_tokens, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")

        result["generation_time"] = time.time() - start_time
        result["method"] = method.value
        return result

    def _run_no_context(self, example: BenchmarkExample, max_tokens: int) -> Dict[str, Any]:
        """Generate without any context."""
        prompt = f"Question: {example.query}\n\nAnswer:"
        answer = self._generate(prompt, max_tokens)
        return {
            "answer": answer,
            "retrieved_files": [],
            "tokens_used": len(prompt.split()),  # Approximate
            "num_retrievals": 0,
        }

    def _run_bm25(self, example: BenchmarkExample, max_tokens: int, top_k: int) -> Dict[str, Any]:
        """Generate with BM25 retrieval."""
        retrieved = self.retriever.retrieve_keyword(example.query, top_k)
        return self._generate_with_context(example, retrieved, max_tokens)

    def _run_full_context(self, example: BenchmarkExample, max_tokens: int) -> Dict[str, Any]:
        """Generate with all documents as context."""
        retrieved = self.retriever.retrieve_all()
        return self._generate_with_context(example, retrieved, max_tokens)

    def _run_random(self, example: BenchmarkExample, max_tokens: int, top_k: int) -> Dict[str, Any]:
        """Generate with random file selection."""
        retrieved = self.retriever.retrieve_random(top_k)
        return self._generate_with_context(example, retrieved, max_tokens)

    def _run_embedding(self, example: BenchmarkExample, max_tokens: int, top_k: int) -> Dict[str, Any]:
        """Generate with pure semantic retrieval."""
        retrieved = self.retriever.retrieve_semantic(example.query, top_k)
        return self._generate_with_context(example, retrieved, max_tokens)

    def _run_reposynth_base(self, example: BenchmarkExample, max_tokens: int, top_k: int) -> Dict[str, Any]:
        """Generate with RepoSynth hybrid retrieval (no entropy triggering)."""
        retrieved = self.retriever.retrieve(example.query, top_k)
        return self._generate_with_context(example, retrieved, max_tokens)

    def _run_uncert_cot(self, example: BenchmarkExample, max_tokens: int, top_k: int) -> Dict[str, Any]:
        """
        Generate with UnCert-CoT style (line boundary measurement).

        This simulates the UnCert-CoT approach where uncertainty is only
        measured at line boundaries, not every token.
        """
        # For baseline comparison, this behaves like hybrid retrieval
        # The difference would be in the adaptive generator configuration
        retrieved = self.retriever.retrieve(example.query, top_k)
        result = self._generate_with_context(example, retrieved, max_tokens)
        result["measurement_strategy"] = "line_boundary"
        return result

    def _generate_with_context(
        self,
        example: BenchmarkExample,
        retrieved: List[Dict[str, Any]],
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Generate answer with retrieved context."""
        # Build context
        context_parts = []
        for doc in retrieved:
            source = doc.get("source", "unknown")
            content = doc.get("content", "")
            context_parts.append(f"File: {source}\n```\n{content}\n```")

        context = "\n\n".join(context_parts)
        prompt = f"Context:\n{context}\n\nQuestion: {example.query}\n\nAnswer:"

        answer = self._generate(prompt, max_tokens)

        return {
            "answer": answer,
            "retrieved_files": [doc.get("source", "") for doc in retrieved],
            "tokens_used": self._count_tokens(context),
            "num_retrievals": 1,
        }

    def _generate(self, prompt: str, max_tokens: int) -> str:
        """Generate text from prompt."""
        if self.model is None or self.tokenizer is None:
            # Mock generation for testing
            return f"[Mock answer for: {prompt[:50]}...]"

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if hasattr(self.model, "device"):
                inputs = inputs.to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            generated = outputs[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(generated, skip_special_tokens=True)

        except Exception as e:
            return f"[Generation error: {e}]"

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        # Fallback: approximate 4 chars per token
        return len(text) // 4


@dataclass
class ExperimentConfig:
    """Configuration for experiment run."""
    methods: List[BaselineMethod] = field(default_factory=lambda: list(BaselineMethod))
    max_tokens: int = 100
    top_k: int = 3
    seed: int = 42
    save_results: bool = True
    output_dir: Optional[Path] = None


class ExperimentRunner:
    """
    Orchestrates full experiment runs.
    """

    def __init__(
        self,
        benchmark: BenchmarkDataset,
        retriever: EvaluationRetriever,
        model=None,
        tokenizer=None,
        metrics: Optional[EvaluationMetrics] = None,
    ):
        """
        Initialize experiment runner.

        Args:
            benchmark: Benchmark dataset
            retriever: EvaluationRetriever instance
            model: Language model
            tokenizer: Tokenizer
            metrics: EvaluationMetrics instance
        """
        self.benchmark = benchmark
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = metrics or EvaluationMetrics()
        self.baseline_runner = BaselineRunner(model, tokenizer, retriever)
        self.stats = StatisticalAnalysis()

    def run_all_methods(
        self,
        config: ExperimentConfig,
    ) -> Dict[str, List[EvaluationResult]]:
        """
        Run all methods on all examples.

        Args:
            config: Experiment configuration

        Returns:
            Dict mapping method name to list of EvaluationResults
        """
        random.seed(config.seed)
        results: Dict[str, List[EvaluationResult]] = {}

        for method in config.methods:
            method_results = []
            print(f"Running {method.value}...")

            for example in self.benchmark:
                raw_result = self.baseline_runner.run(
                    example, method, config.max_tokens, config.top_k
                )

                # Evaluate
                eval_result = self.metrics.evaluate(
                    example_id=example.id,
                    generated_answer=raw_result["answer"],
                    ground_truth_answer=example.ground_truth_answer,
                    retrieved_files=raw_result["retrieved_files"],
                    ground_truth_files=example.ground_truth_files,
                    ground_truth_keywords=example.ground_truth_keywords,
                    tokens_used=raw_result["tokens_used"],
                    baseline_tokens=1000,  # Placeholder
                    generation_time=raw_result["generation_time"],
                    num_retrievals=raw_result["num_retrievals"],
                    ground_truth_positions=example.ground_truth_missing_positions,
                    method=method.value,
                )
                method_results.append(eval_result)

            results[method.value] = method_results
            print(f"  Completed {len(method_results)} examples")

        return results

    def compare_to_baseline(
        self,
        results: Dict[str, List[EvaluationResult]],
        baseline: str = "no_context",
    ) -> List[ComparisonResult]:
        """
        Compare all methods to a baseline.

        Args:
            results: Results from run_all_methods
            baseline: Baseline method name

        Returns:
            List of ComparisonResults
        """
        if baseline not in results:
            raise ValueError(f"Baseline {baseline} not in results")

        comparisons = []
        baseline_results = results[baseline]

        metrics_to_compare = [
            "answer_correctness",
            "answer_completeness",
            "context_precision",
            "context_recall",
            "token_efficiency",
        ]

        for method, method_results in results.items():
            if method == baseline:
                continue

            for metric in metrics_to_compare:
                baseline_scores = [getattr(r, metric) for r in baseline_results]
                method_scores = [getattr(r, metric) for r in method_results]

                comparison = self.stats.compare_methods(
                    method, method_scores,
                    baseline, baseline_scores,
                    metric,
                )
                comparisons.append(comparison)

        return comparisons

    def aggregate_results(
        self,
        results: Dict[str, List[EvaluationResult]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute aggregate statistics for each method.

        Args:
            results: Results from run_all_methods

        Returns:
            Dict mapping method -> metric -> mean value
        """
        aggregates = {}

        for method, method_results in results.items():
            if not method_results:
                continue

            aggregates[method] = {
                "answer_correctness": np.mean([r.answer_correctness for r in method_results]),
                "answer_completeness": np.mean([r.answer_completeness for r in method_results]),
                "hallucination_rate": np.mean([r.hallucination_rate for r in method_results]),
                "context_precision": np.mean([r.context_precision for r in method_results]),
                "context_recall": np.mean([r.context_recall for r in method_results]),
                "context_f1": np.mean([r.get_f1_context() for r in method_results]),
                "token_efficiency": np.mean([r.token_efficiency for r in method_results]),
                "composite_score": np.mean([r.get_composite_score() for r in method_results]),
                "generation_time": np.mean([r.generation_time for r in method_results]),
            }

        return aggregates

    def save_results(
        self,
        results: Dict[str, List[EvaluationResult]],
        output_dir: Path,
    ) -> None:
        """
        Save results to files.

        Args:
            results: Results from run_all_methods
            output_dir: Directory to save to
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Raw results
        raw_data = {
            method: [r.to_dict() for r in method_results]
            for method, method_results in results.items()
        }
        with open(output_dir / "raw_results.json", "w") as f:
            json.dump(raw_data, f, indent=2)

        # Aggregates
        aggregates = self.aggregate_results(results)
        with open(output_dir / "aggregates.json", "w") as f:
            json.dump(aggregates, f, indent=2)

        # CSV summary
        try:
            import pandas as pd
            rows = []
            for method, metrics in aggregates.items():
                row = {"method": method, **metrics}
                rows.append(row)
            df = pd.DataFrame(rows)
            df.to_csv(output_dir / "summary.csv", index=False)
        except ImportError:
            pass

        print(f"Results saved to {output_dir}")
