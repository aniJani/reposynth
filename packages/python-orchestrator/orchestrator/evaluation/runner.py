"""
Experiment Runner Module for Evaluation.

This module provides orchestration for running experiments:
- BaselineRunner: Run individual baselines
- ExperimentRunner: Full experimental pipeline

Phase 4, Week 9: Baseline Comparisons

Baselines:
1. No Context - Generate without any retrieved context
2. Full Context - Include all available context
3. Random Context - Random file selection
4. BM25 Retrieval - Traditional keyword-based retrieval
5. Embedding Retrieval - Semantic similarity retrieval
6. RepoSynth Base - RepoSynth without entropy monitoring
7. UnCert-CoT - Line boundary measurement (baseline paper)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from enum import Enum
import time
import json
import os

from .benchmark import BenchmarkDataset, BenchmarkExample
from .metrics import EvaluationMetrics, EvaluationResult


class BaselineMethod(Enum):
    """Available baseline methods."""
    NO_CONTEXT = "no_context"
    FULL_CONTEXT = "full_context"
    RANDOM_CONTEXT = "random_context"
    BM25 = "bm25"
    EMBEDDING = "embedding"
    REPOSYNTH_BASE = "reposynth_base"
    UNCERT_COT = "uncert_cot"
    CCE_ADAPTIVE = "cce_adaptive"  # Our method


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    name: str
    methods: List[BaselineMethod]
    dataset_path: Optional[str] = None
    output_dir: str = "results"

    # Generation settings
    max_tokens: int = 256
    temperature: float = 0.0

    # Retrieval settings
    max_retrievals: int = 5
    top_k: int = 3

    # Evaluation settings
    use_llm_judge: bool = False
    llm_judge_model: str = "gpt-4"
    llm_api_key: Optional[str] = None

    # Checkpointing
    checkpoint_every: int = 10
    resume_from: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'methods': [m.value for m in self.methods],
            'dataset_path': self.dataset_path,
            'output_dir': self.output_dir,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'max_retrievals': self.max_retrievals,
            'top_k': self.top_k,
            'use_llm_judge': self.use_llm_judge,
        }


@dataclass
class ExperimentResults:
    """Results from an experiment run."""
    config: ExperimentConfig
    results_by_method: Dict[str, List[EvaluationResult]]
    aggregates_by_method: Dict[str, Dict]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_comparison_table(self) -> Dict[str, Dict[str, float]]:
        """Get comparison table of methods."""
        table = {}
        for method, aggregate in self.aggregates_by_method.items():
            table[method] = {
                metric: stats['mean']
                for metric, stats in aggregate.get('metrics', {}).items()
            }
        return table

    def save(self, path: str):
        """Save results to JSON."""
        data = {
            'config': self.config.to_dict(),
            'results_by_method': {
                method: [r.to_dict() for r in results]
                for method, results in self.results_by_method.items()
            },
            'aggregates_by_method': self.aggregates_by_method,
            'metadata': self.metadata,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


class BaselineRunner:
    """
    Run baseline methods for comparison.

    Each baseline generates answers for benchmark examples,
    allowing fair comparison with the CCE adaptive method.

    Example:
        >>> runner = BaselineRunner(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     documents=document_store,
        ... )
        >>> result = runner.run(
        ...     example=benchmark_example,
        ...     method=BaselineMethod.BM25,
        ... )
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        documents: Optional[Dict[str, str]] = None,
        embedding_model: str = 'all-MiniLM-L6-v2',
    ):
        """
        Initialize baseline runner.

        Args:
            model: Language model for generation
            tokenizer: Tokenizer
            documents: Document store {path: content}
            embedding_model: Model for embedding retrieval
        """
        self.model = model
        self.tokenizer = tokenizer
        self.documents = documents or {}
        self.embedding_model_name = embedding_model

        # Lazy load embedding model
        self._embedding_model = None
        self._doc_embeddings = None

    @property
    def embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                pass
        return self._embedding_model

    def run(
        self,
        example: BenchmarkExample,
        method: BaselineMethod,
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """
        Run a baseline method on an example.

        Args:
            example: Benchmark example to evaluate
            method: Baseline method to use
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with generated answer, retrieved files, etc.
        """
        start_time = time.time()

        if method == BaselineMethod.NO_CONTEXT:
            result = self._run_no_context(example, max_tokens)
        elif method == BaselineMethod.FULL_CONTEXT:
            result = self._run_full_context(example, max_tokens)
        elif method == BaselineMethod.RANDOM_CONTEXT:
            result = self._run_random_context(example, max_tokens)
        elif method == BaselineMethod.BM25:
            result = self._run_bm25(example, max_tokens)
        elif method == BaselineMethod.EMBEDDING:
            result = self._run_embedding(example, max_tokens)
        else:
            result = self._run_no_context(example, max_tokens)

        result['generation_time'] = time.time() - start_time
        result['method'] = method.value

        return result

    def _run_no_context(
        self,
        example: BenchmarkExample,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Generate without any context."""
        prompt = example.query
        if example.initial_context:
            prompt = f"{example.initial_context}\n\n{prompt}"

        answer = self._generate(prompt, max_tokens)

        return {
            'answer': answer,
            'retrieved_files': [],
            'tokens_used': len(self.tokenizer.encode(prompt)) if self.tokenizer else 0,
            'num_retrievals': 0,
        }

    def _run_full_context(
        self,
        example: BenchmarkExample,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Generate with all available context."""
        context_parts = []
        retrieved_files = []

        for path, content in self.documents.items():
            context_parts.append(f"# {path}\n{content}")
            retrieved_files.append(path)

        context = "\n\n".join(context_parts)
        prompt = f"{context}\n\n{example.query}"

        answer = self._generate(prompt, max_tokens)

        return {
            'answer': answer,
            'retrieved_files': retrieved_files,
            'tokens_used': len(self.tokenizer.encode(prompt)) if self.tokenizer else 0,
            'num_retrievals': len(retrieved_files),
        }

    def _run_random_context(
        self,
        example: BenchmarkExample,
        max_tokens: int,
        num_files: int = 2,
    ) -> Dict[str, Any]:
        """Generate with randomly selected context."""
        import random

        paths = list(self.documents.keys())
        selected = random.sample(paths, min(num_files, len(paths)))

        context_parts = []
        for path in selected:
            context_parts.append(f"# {path}\n{self.documents[path]}")

        context = "\n\n".join(context_parts)
        prompt = f"{context}\n\n{example.query}"

        answer = self._generate(prompt, max_tokens)

        return {
            'answer': answer,
            'retrieved_files': selected,
            'tokens_used': len(self.tokenizer.encode(prompt)) if self.tokenizer else 0,
            'num_retrievals': len(selected),
        }

    def _run_bm25(
        self,
        example: BenchmarkExample,
        max_tokens: int,
        top_k: int = 2,
    ) -> Dict[str, Any]:
        """Generate with BM25 retrieved context."""
        # Simple keyword-based retrieval
        query_terms = set(example.query.lower().split())

        scores = []
        for path, content in self.documents.items():
            content_lower = content.lower()
            score = sum(1 for term in query_terms if term in content_lower)
            scores.append((path, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [path for path, _ in scores[:top_k]]

        context_parts = []
        for path in selected:
            context_parts.append(f"# {path}\n{self.documents[path]}")

        context = "\n\n".join(context_parts)
        prompt = f"{context}\n\n{example.query}"

        answer = self._generate(prompt, max_tokens)

        return {
            'answer': answer,
            'retrieved_files': selected,
            'tokens_used': len(self.tokenizer.encode(prompt)) if self.tokenizer else 0,
            'num_retrievals': len(selected),
        }

    def _run_embedding(
        self,
        example: BenchmarkExample,
        max_tokens: int,
        top_k: int = 2,
    ) -> Dict[str, Any]:
        """Generate with embedding-based retrieved context."""
        if not self.embedding_model:
            return self._run_bm25(example, max_tokens, top_k)

        # Embed query
        import numpy as np
        query_emb = self.embedding_model.encode([example.query])[0]

        # Embed documents if not cached
        if self._doc_embeddings is None:
            doc_texts = list(self.documents.values())
            self._doc_embeddings = self.embedding_model.encode(doc_texts)

        # Compute similarities
        paths = list(self.documents.keys())
        similarities = []
        for i, path in enumerate(paths):
            sim = np.dot(query_emb, self._doc_embeddings[i]) / (
                np.linalg.norm(query_emb) * np.linalg.norm(self._doc_embeddings[i])
            )
            similarities.append((path, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        selected = [path for path, _ in similarities[:top_k]]

        context_parts = []
        for path in selected:
            context_parts.append(f"# {path}\n{self.documents[path]}")

        context = "\n\n".join(context_parts)
        prompt = f"{context}\n\n{example.query}"

        answer = self._generate(prompt, max_tokens)

        return {
            'answer': answer,
            'retrieved_files': selected,
            'tokens_used': len(self.tokenizer.encode(prompt)) if self.tokenizer else 0,
            'num_retrievals': len(selected),
        }

    def _generate(self, prompt: str, max_tokens: int) -> str:
        """Generate text with the model."""
        if self.model is None or self.tokenizer is None:
            return "[Model not available]"

        try:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated = outputs[0][inputs['input_ids'].shape[1]:]
            return self.tokenizer.decode(generated, skip_special_tokens=True)

        except Exception as e:
            return f"[Generation error: {e}]"


class ExperimentRunner:
    """
    Run full experiments across methods and examples.

    Orchestrates running multiple baselines on a benchmark dataset,
    computing metrics, and aggregating results.

    Example:
        >>> runner = ExperimentRunner(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     documents=doc_store,
        ... )
        >>> results = runner.run_experiment(
        ...     dataset=benchmark,
        ...     methods=[BaselineMethod.BM25, BaselineMethod.CCE_ADAPTIVE],
        ... )
        >>> print(results.get_comparison_table())
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        documents: Optional[Dict[str, str]] = None,
        adaptive_generator=None,
    ):
        """
        Initialize experiment runner.

        Args:
            model: Language model
            tokenizer: Tokenizer
            documents: Document store
            adaptive_generator: AdaptiveGenerator for CCE method
        """
        self.baseline_runner = BaselineRunner(
            model=model,
            tokenizer=tokenizer,
            documents=documents,
        )
        self.adaptive_generator = adaptive_generator
        self.metrics = EvaluationMetrics()
        self.documents = documents or {}

    def run_experiment(
        self,
        dataset: BenchmarkDataset,
        methods: List[BaselineMethod],
        config: Optional[ExperimentConfig] = None,
        progress_callback: Optional[Callable] = None,
    ) -> ExperimentResults:
        """
        Run full experiment.

        Args:
            dataset: Benchmark dataset
            methods: Methods to evaluate
            config: Experiment configuration
            progress_callback: Callback for progress updates

        Returns:
            ExperimentResults with all results and aggregates
        """
        if config is None:
            config = ExperimentConfig(
                name="experiment",
                methods=methods,
            )

        results_by_method: Dict[str, List[EvaluationResult]] = {
            m.value: [] for m in methods
        }

        total = len(dataset) * len(methods)
        current = 0

        for example in dataset:
            for method in methods:
                current += 1
                if progress_callback:
                    progress_callback(current, total, method.value, example.id)

                # Run method
                if method == BaselineMethod.CCE_ADAPTIVE:
                    run_result = self._run_adaptive(example, config.max_tokens)
                else:
                    run_result = self.baseline_runner.run(
                        example=example,
                        method=method,
                        max_tokens=config.max_tokens,
                    )

                # Compute baseline tokens (full context)
                baseline_tokens = sum(
                    len(self.baseline_runner.tokenizer.encode(c))
                    for c in self.documents.values()
                ) if self.baseline_runner.tokenizer else 0

                # Evaluate
                eval_result = self.metrics.evaluate(
                    example_id=example.id,
                    generated_answer=run_result['answer'],
                    ground_truth_answer=example.ground_truth_answer,
                    retrieved_files=run_result['retrieved_files'],
                    ground_truth_files=example.ground_truth_files,
                    ground_truth_keywords=example.ground_truth_keywords,
                    tokens_used=run_result['tokens_used'],
                    baseline_tokens=baseline_tokens,
                    generation_time=run_result['generation_time'],
                    num_retrievals=run_result['num_retrievals'],
                    method=method.value,
                )

                results_by_method[method.value].append(eval_result)

        # Aggregate results
        aggregates = {}
        for method, results in results_by_method.items():
            aggregates[method] = self.metrics.aggregate_results(results)

        return ExperimentResults(
            config=config,
            results_by_method=results_by_method,
            aggregates_by_method=aggregates,
            metadata={
                'dataset_size': len(dataset),
                'num_methods': len(methods),
            },
        )

    def _run_adaptive(
        self,
        example: BenchmarkExample,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Run CCE adaptive method."""
        if self.adaptive_generator is None:
            return {
                'answer': "[AdaptiveGenerator not configured]",
                'retrieved_files': [],
                'tokens_used': 0,
                'generation_time': 0,
                'num_retrievals': 0,
            }

        result = self.adaptive_generator.generate(
            query=example.query,
            initial_context=example.initial_context,
        )

        return {
            'answer': result.response,
            'retrieved_files': result.context_sources,
            'tokens_used': result.total_context_tokens,
            'generation_time': result.generation_time,
            'num_retrievals': result.total_retrievals,
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_experiment(
    name: str,
    methods: Optional[List[str]] = None,
    **kwargs
) -> ExperimentConfig:
    """
    Create experiment configuration.

    Args:
        name: Experiment name
        methods: List of method names (strings)
        **kwargs: Additional config options

    Returns:
        ExperimentConfig
    """
    if methods is None:
        methods = ['bm25', 'embedding', 'cce_adaptive']

    method_enums = [BaselineMethod(m) for m in methods]

    return ExperimentConfig(
        name=name,
        methods=method_enums,
        **kwargs,
    )
