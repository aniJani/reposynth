"""Content-hash-keyed LLM-as-judge cache.

Every (question, generated_answer, gold_answer) triple is judged at most
once, ever, across all methods and seeds. Cache survives restarts and is
shared across runs in the same project. This is the single biggest cost
saver in the experiment (judge calls dominate API spend).

Usage:
    judge = Judge(model="gpt-4o-mini-2024-07-18", cache_path="...")
    score, explanation = judge.score(question, generated, gold)
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any


JUDGE_PROMPT = """You are an expert code reviewer evaluating an LLM's answer to a question about a Python codebase.

QUESTION:
{question}

REFERENCE (GROUND TRUTH) ANSWER:
{gold}

CANDIDATE ANSWER (from the LLM under evaluation):
{generated}

Score the candidate answer for correctness on a continuous scale from 0.0 to 1.0:
- 1.0  = factually equivalent to the reference; all key facts present and accurate
- 0.7  = mostly correct with minor omissions or imprecision; no factual errors
- 0.4  = partially correct; either missing major facts OR contains a factual error
- 0.1  = wrong direction or hallucinated content
- 0.0  = empty, off-topic, or completely incorrect

Return STRICTLY a single JSON object on one line, no prose, no markdown:
{{"score": <float>, "explanation": "<one-sentence justification>"}}
"""


def prompt_sha() -> str:
    return hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest()[:16]


def _content_key(question: str, generated: str, gold: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt_sha().encode("utf-8"))
    h.update(b"\x00")
    h.update(question.encode("utf-8"))
    h.update(b"\x00")
    h.update(generated.encode("utf-8"))
    h.update(b"\x00")
    h.update(gold.encode("utf-8"))
    return h.hexdigest()


class Judge:
    def __init__(
        self,
        model: str = "gpt-4o-mini-2024-07-18",
        cache_path: str | Path = "research/paper/judge_cache.json",
        api_key_env: str = "OPENAI_API_KEY",
        max_retries: int = 5,
        retry_base_seconds: float = 2.0,
    ):
        self.model = model
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, dict[str, Any]] = self._load_cache()
        self._api_key = os.environ.get(api_key_env)
        self._max_retries = max_retries
        self._retry_base = retry_base_seconds
        self._dirty = False
        self._last_flush = 0.0

    def _load_cache(self) -> dict:
        if not self.cache_path.exists():
            return {}
        try:
            with open(self.cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            with open(tmp, "w") as f:
                json.dump(self._cache, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.cache_path)
            self._dirty = False
            self._last_flush = time.time()

    def maybe_flush(self, every_seconds: float = 30.0) -> None:
        if time.time() - self._last_flush >= every_seconds:
            self._flush()

    def n_cached(self) -> int:
        return len(self._cache)

    def score(
        self,
        question: str,
        generated: str,
        gold: str,
        offline: bool = False,
    ) -> dict[str, Any]:
        """Return {"score": float, "explanation": str, "cached": bool}.

        If offline=True, fall back to a cheap heuristic (used by --dry-run).
        Raises if no API key and not offline and not cached.
        """
        key = _content_key(question, generated, gold, self.model)

        with self._lock:
            if key in self._cache:
                hit = dict(self._cache[key])
                hit["cached"] = True
                return hit

        if offline:
            return _heuristic_score(generated, gold) | {"cached": False}

        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set and judge call not cached. "
                "Either set the env var or run with --dry-run."
            )

        result = self._call_api(question, generated, gold)
        with self._lock:
            self._cache[key] = result
            self._dirty = True
        self.maybe_flush()
        return result | {"cached": False}

    def _call_api(self, question: str, generated: str, gold: str) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai package not installed: pip install openai") from e

        client = OpenAI(api_key=self._api_key)
        prompt = JUDGE_PROMPT.format(question=question, gold=gold, generated=generated)

        last_err = None
        for attempt in range(self._max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=200,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                obj = json.loads(content)
                score = float(obj.get("score", 0.0))
                score = max(0.0, min(1.0, score))
                return {
                    "score": score,
                    "explanation": str(obj.get("explanation", ""))[:500],
                    "judge_model": self.model,
                    "prompt_sha": prompt_sha(),
                }
            except Exception as e:
                last_err = e
                wait = self._retry_base * (2 ** attempt)
                time.sleep(wait)
        raise RuntimeError(f"judge API failed after {self._max_retries} retries: {last_err}")

    def close(self) -> None:
        self._dirty = True
        self._flush()


def _heuristic_score(generated: str, gold: str) -> dict[str, Any]:
    """Very rough offline fallback for --dry-run only. Do not use for real results."""
    g = generated.lower().split()
    r = set(gold.lower().split())
    if not g:
        return {"score": 0.0, "explanation": "empty generation (heuristic)", "judge_model": "heuristic"}
    overlap = sum(1 for w in g if w in r) / max(len(g), 1)
    return {
        "score": min(1.0, overlap * 1.5),
        "explanation": "heuristic word overlap (offline mode)",
        "judge_model": "heuristic",
        "prompt_sha": "heuristic",
    }
