"""Static-pre-retrieval baselines: no_context, random, bm25, embedding, full_context, reposynth_base.

Each is a thin BaseMethod subclass. The retrieval indices and the generation
function come from MethodContext. Method bodies are intentionally simple —
they exist to be honest baselines, not to be optimized.
"""

from __future__ import annotations

import random as _random
import time
from typing import Iterable

from .base import BaseMethod, MethodContext, Question, Result


TOP_K = 5


def _f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def _context_metrics(retrieved: list[str], gold: list[str]) -> tuple[float, float, float]:
    if not retrieved:
        return 0.0, 0.0, 0.0
    retrieved_set = set(retrieved)
    gold_set = set(gold)
    if not gold_set:
        return 0.0, 0.0, 0.0
    tp = len(retrieved_set & gold_set)
    p = tp / len(retrieved_set)
    r = tp / len(gold_set)
    return p, r, _f1(p, r)


def _generate(ctx: MethodContext, prompt: str, seed: int) -> tuple[str, int, int, int]:
    """Call the model; return (text, tokens_generated, latency_ms, peak_vram_mb).

    Returns synthetic values if no model is loaded (dry-run).
    """
    t0 = time.time()
    if ctx.model is None or ctx.tokenizer is None:
        return f"[DRY-RUN OUTPUT for prompt of length {len(prompt)}]", 0, 0, 0

    import torch

    inputs = ctx.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=ctx.max_input_tokens,
    ).to(ctx.device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        out = ctx.model.generate(
            **inputs,
            max_new_tokens=ctx.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=ctx.tokenizer.eos_token_id,
        )

    n_input = inputs["input_ids"].shape[1]
    gen_ids = out[0, n_input:]
    text = ctx.tokenizer.decode(gen_ids, skip_special_tokens=True)
    n_generated = int(gen_ids.shape[0])
    latency = int((time.time() - t0) * 1000)
    peak_vram = int(torch.cuda.max_memory_allocated() // (1024 * 1024)) if torch.cuda.is_available() else 0
    return text, n_generated, latency, peak_vram


def _build_prompt(query: str, files: dict[str, str]) -> str:
    if not files:
        return f"Question: {query}\n\nAnswer:"
    blocks = "\n\n".join(f"# File: {p}\n{c}" for p, c in files.items())
    return f"Context:\n{blocks}\n\nQuestion: {query}\n\nAnswer:"


def _count_tokens(ctx: MethodContext, text: str) -> int:
    if ctx.tokenizer is None:
        return len(text) // 4
    return len(ctx.tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------- methods


class NoContextMethod(BaseMethod):
    name = "no_context"

    def run(self, q: Question) -> Result:
        r = self._empty_result(q.qid, self.config["seed"])
        prompt = _build_prompt(q.query, {})
        text, n_gen, ms, vram = _generate(self.ctx, prompt, self.config["seed"])
        r.generated_text = text
        r.tokens_used = _count_tokens(self.ctx, prompt)
        r.tokens_generated = n_gen
        r.tokens_retrieved = 0
        r.latency_ms = ms
        r.peak_vram_mb = vram
        r.retrievals = 0
        r.retrieved_files = []
        return r


class RandomMethod(BaseMethod):
    name = "random"

    def run(self, q: Question) -> Result:
        r = self._empty_result(q.qid, self.config["seed"])
        rng = self.ctx.rng or _random.Random(self.config["seed"])
        all_paths = list(self.ctx.repo_files.keys())
        rng.shuffle(all_paths)
        chosen = all_paths[:TOP_K]
        files = {p: self.ctx.repo_files[p] for p in chosen}
        prompt = _build_prompt(q.query, files)
        text, n_gen, ms, vram = _generate(self.ctx, prompt, self.config["seed"])
        p, rec, f1 = _context_metrics(chosen, q.gold_files)
        r.generated_text = text
        r.retrievals = 1
        r.retrieved_files = chosen
        r.context_p, r.context_r, r.context_f1 = p, rec, f1
        r.tokens_used = _count_tokens(self.ctx, prompt)
        r.tokens_generated = n_gen
        r.tokens_retrieved = sum(_count_tokens(self.ctx, c) for c in files.values())
        r.latency_ms = ms
        r.peak_vram_mb = vram
        return r


class BM25Method(BaseMethod):
    name = "bm25"

    def run(self, q: Question) -> Result:
        r = self._empty_result(q.qid, self.config["seed"])
        scored = self.ctx.bm25_index.top_k(q.query, k=TOP_K)
        chosen = [p for p, _ in scored]
        files = {p: self.ctx.repo_files[p] for p in chosen}
        prompt = _build_prompt(q.query, files)
        text, n_gen, ms, vram = _generate(self.ctx, prompt, self.config["seed"])
        p, rec, f1 = _context_metrics(chosen, q.gold_files)
        r.generated_text = text
        r.retrievals = 1
        r.retrieved_files = chosen
        r.context_p, r.context_r, r.context_f1 = p, rec, f1
        r.tokens_used = _count_tokens(self.ctx, prompt)
        r.tokens_generated = n_gen
        r.tokens_retrieved = sum(_count_tokens(self.ctx, c) for c in files.values())
        r.latency_ms = ms
        r.peak_vram_mb = vram
        return r


class EmbeddingMethod(BaseMethod):
    name = "embedding"

    def run(self, q: Question) -> Result:
        r = self._empty_result(q.qid, self.config["seed"])
        scored = self.ctx.embeddings_index.top_k(q.query, k=TOP_K)
        chosen = [p for p, _ in scored]
        files = {p: self.ctx.repo_files[p] for p in chosen}
        prompt = _build_prompt(q.query, files)
        text, n_gen, ms, vram = _generate(self.ctx, prompt, self.config["seed"])
        p, rec, f1 = _context_metrics(chosen, q.gold_files)
        r.generated_text = text
        r.retrievals = 1
        r.retrieved_files = chosen
        r.context_p, r.context_r, r.context_f1 = p, rec, f1
        r.tokens_used = _count_tokens(self.ctx, prompt)
        r.tokens_generated = n_gen
        r.tokens_retrieved = sum(_count_tokens(self.ctx, c) for c in files.values())
        r.latency_ms = ms
        r.peak_vram_mb = vram
        return r


class FullContextMethod(BaseMethod):
    """Pack as many files as fit under the model context; gold-files first, then frequency-weighted."""

    name = "full_context"

    def run(self, q: Question) -> Result:
        r = self._empty_result(q.qid, self.config["seed"])
        all_paths = list(self.ctx.repo_files.keys())
        ordered = sorted(all_paths, key=lambda p: (p not in q.gold_files, p))
        chosen, used = [], 0
        budget = max(0, self.ctx.max_input_tokens - 256)
        for p in ordered:
            t = _count_tokens(self.ctx, self.ctx.repo_files[p])
            if used + t > budget:
                break
            chosen.append(p)
            used += t
        files = {p: self.ctx.repo_files[p] for p in chosen}
        prompt = _build_prompt(q.query, files)
        text, n_gen, ms, vram = _generate(self.ctx, prompt, self.config["seed"])
        p, rec, f1 = _context_metrics(chosen, q.gold_files)
        r.generated_text = text
        r.retrievals = 1
        r.retrieved_files = chosen
        r.context_p, r.context_r, r.context_f1 = p, rec, f1
        r.tokens_used = _count_tokens(self.ctx, prompt)
        r.tokens_generated = n_gen
        r.tokens_retrieved = used
        r.latency_ms = ms
        r.peak_vram_mb = vram
        return r


class ReposynthBaseMethod(BaseMethod):
    """Hybrid keyword+semantic upfront retrieval — the existing RepoSynth retriever, no entropy."""

    name = "reposynth_base"

    def run(self, q: Question) -> Result:
        r = self._empty_result(q.qid, self.config["seed"])
        chosen = self._hybrid_retrieve(q.query)
        files = {p: self.ctx.repo_files[p] for p in chosen if p in self.ctx.repo_files}
        prompt = _build_prompt(q.query, files)
        text, n_gen, ms, vram = _generate(self.ctx, prompt, self.config["seed"])
        p, rec, f1 = _context_metrics(list(files.keys()), q.gold_files)
        r.generated_text = text
        r.retrievals = 1
        r.retrieved_files = list(files.keys())
        r.context_p, r.context_r, r.context_f1 = p, rec, f1
        r.tokens_used = _count_tokens(self.ctx, prompt)
        r.tokens_generated = n_gen
        r.tokens_retrieved = sum(_count_tokens(self.ctx, c) for c in files.values())
        r.latency_ms = ms
        r.peak_vram_mb = vram
        return r

    def _hybrid_retrieve(self, query: str) -> list[str]:
        """Wire to the existing reposynth retriever in
        packages/python-orchestrator/orchestrator/retriever.py.
        Stub returns BM25 ∪ embedding top-K as a sensible default."""
        bm25_hits = [p for p, _ in self.ctx.bm25_index.top_k(query, k=TOP_K)] if self.ctx.bm25_index else []
        emb_hits = [p for p, _ in self.ctx.embeddings_index.top_k(query, k=TOP_K)] if self.ctx.embeddings_index else []
        seen, out = set(), []
        for p in bm25_hits + emb_hits:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out[:TOP_K]
