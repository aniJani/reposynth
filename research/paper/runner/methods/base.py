"""Common interfaces and dataclasses for all method implementations.

A method's job: given a Question and a Context (model + retrieval indices),
return a Result. The runner handles everything else (judging, checkpointing,
seeding, logging).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Question:
    qid: str
    repo: str
    query: str
    gold_answer: str
    gold_files: list[str]
    gold_keywords: list[str] = field(default_factory=list)
    gold_missing_positions: list[int] = field(default_factory=list)
    difficulty: str = "medium"
    category: str = "api_usage"
    hop_count: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        return cls(
            qid=d["id"],
            repo=d.get("repo", "unknown"),
            query=d["query"],
            gold_answer=d.get("ground_truth_answer", ""),
            gold_files=list(d.get("ground_truth_files", [])),
            gold_keywords=list(d.get("ground_truth_keywords", [])),
            gold_missing_positions=list(d.get("ground_truth_missing_positions", [])),
            difficulty=d.get("difficulty", "medium"),
            category=d.get("category", "api_usage"),
            hop_count=int(d.get("hop_count", 1)),
        )


@dataclass
class Result:
    """One row of paper_results.json."""

    qid: str
    method: str
    seed: int
    tau: float | None = None
    partition: str | None = None
    layers: list | None = None
    max_hops: int | None = None

    status: str = "ok"
    error: str | None = None

    correctness: float = 0.0
    retrievals: int = 0
    context_p: float = 0.0
    context_r: float = 0.0
    context_f1: float = 0.0
    tokens_used: int = 0
    tokens_generated: int = 0
    tokens_retrieved: int = 0
    hallucination: float = 0.0
    latency_ms: int = 0
    peak_vram_mb: int = 0

    cce_trace: list = field(default_factory=list)
    retrieved_files: list = field(default_factory=list)
    generated_text: str = ""

    judge_score: float = 0.0
    judge_explanation: str = ""
    judge_cached: bool = False

    def to_row(self) -> dict:
        d = asdict(self)
        return d


class MethodContext:
    """Bag of shared resources passed to every method.

    Avoids reloading the LLM, the retriever indexes, etc., per call.
    Methods read what they need; ignore what they don't.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        embedder=None,
        repo_files: dict | None = None,
        embeddings_index=None,
        bm25_index=None,
        token_classifier=None,
        rng=None,
        device: str = "cuda",
        max_input_tokens: int = 4096,
        max_new_tokens: int = 150,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.repo_files = repo_files or {}
        self.embeddings_index = embeddings_index
        self.bm25_index = bm25_index
        self.token_classifier = token_classifier
        self.rng = rng
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens


class BaseMethod(ABC):
    """Subclass and implement run(). Each method is a stateless transform."""

    name: str = "base"

    def __init__(self, ctx: MethodContext, config: dict):
        self.ctx = ctx
        self.config = config

    @abstractmethod
    def run(self, question: Question) -> Result:
        """Generate an answer for the question and populate a Result."""
        raise NotImplementedError

    def _empty_result(self, qid: str, seed: int) -> Result:
        return Result(
            qid=qid,
            method=self.name,
            seed=seed,
            tau=self.config.get("tau"),
            partition=self.config.get("partition"),
            layers=list(self.config["layers"]) if self.config.get("layers") else None,
            max_hops=self.config.get("max_hops"),
        )
