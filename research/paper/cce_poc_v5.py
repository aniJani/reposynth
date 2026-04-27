"""V5 PoC: full-study runner for one model on RepoBench cross_file_first.

What v5 does (per model, single execution):
- Load tianyang/repobench_python_v1.1 cross_file_first split, n=TARGET_N stratified.
- Phase 1: never-retrieve generation + per-token features (all N tasks).
- Phase 2: always-retrieve generation with retrieval context (all N tasks).
- Phase 3: filter to tasks where always_correct=1 (the subset where retrieval
  was sufficient; the remaining phases operate on this subset).
- Phase 4: 5 sampled generations per filtered task (for semantic entropy).
- Phase 5: NLI clustering (DeBERTa-large-MNLI fp32) → NLI-SE features.
- Phase 6: Embedding-SE features.
- Phase 7: Per-arm AUC (univariate + LOO-LR for multivariate) on the filtered
  subset, with `needs_retrieval = NOT initial_correct` as the binary label.
  Bootstrap 95% CIs on every AUC.
- Phase 8: End-to-end gated retrieval accuracy/cost curves on the filtered
  subset for each gating signal.
- Save results__{slug}.json.

Cache keys are content-stable (md5 of repo + question prefix), not sequential
IDs that change with TARGET_N. The v4 collision bug (n=50 cache contaminating
an n=250 re-run) cannot recur.

Usage:
    python research/paper/cce_poc_v5.py \\
        --model "Qwen/Qwen2.5-Coder-7B-Instruct" \\
        --n-tasks 250 \\
        --out-dir /content/drive/MyDrive/cce_poc_v5
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.paper.runner.cce_features import (
    CCEFeatureExtractor, build_partition_indices,
)
from research.paper.runner.semantic_entropy import semantic_entropy
from research.paper.cce_poc_v3 import (
    setup_model_and_embedder, probe_layers_for, build_chat_prompt,
    generate_with_features, generate_simple, sample_generations,
    build_chunks_and_index, retrieve,
)


# ───────────────────────────────────────────── ablation arms (v3 + NLI-SE)

EMB_SE_FEATURES = (
    "semantic_entropy", "semantic_entropy_norm",
    "n_clusters", "largest_cluster_frac",
)
NLI_SE_FEATURES = (
    "nli_se", "nli_se_norm",
    "nli_n_clusters", "nli_largest_cluster_frac",
)
SAPLMA_FEATURES = ("hs_mean_norm", "hs_std_norm", "hs_max_norm")
FLARE_FEATURES = ("cce_mean", "cce_max", "cce_std", "cce_spikes")
CCE_NEW_FEATURES = (
    "real_cce_mean", "real_cce_max", "real_cce_std", "real_cce_spikes",
    "h_code_mean", "h_code_max", "h_lang_mean", "h_lang_max",
)
V6_FEATURES = SAPLMA_FEATURES + FLARE_FEATURES + (
    "attn_mean", "attn_max", "attn_std", "response_length",
)

ARMS = {
    "saplma_only":       SAPLMA_FEATURES,
    "flare_only":        FLARE_FEATURES,
    "v6_full":           V6_FEATURES,
    "cce_only":          CCE_NEW_FEATURES,
    "embedding_se_only": EMB_SE_FEATURES,
    "nli_se_only":       NLI_SE_FEATURES,
    "flare_plus_emb_se": FLARE_FEATURES + EMB_SE_FEATURES,
    "flare_plus_nli_se": FLARE_FEATURES + NLI_SE_FEATURES,
    "all_features":      V6_FEATURES + CCE_NEW_FEATURES + EMB_SE_FEATURES + NLI_SE_FEATURES,
}

UNIVARIATE_SIGNALS = [
    ("nli_se",            "NLI-SE"),
    ("semantic_entropy",  "embedding-SE"),
    ("cce_max",           "FLARE max-ent"),
    ("cce_mean",          "FLARE mean-ent"),
    ("cce_spikes",        "FLARE spikes"),
    ("hs_max_norm",       "SAPLMA hs_max"),
    ("real_cce_max",      "CCE max"),
    ("response_length",   "length (control)"),
]


# ───────────────────────────────────────────── stable cache keys

def task_key(t: dict) -> str:
    """Content-stable cache key. Independent of TARGET_N or sample ordering."""
    payload = f"{t['repo']}||{t['question'][:500]}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]


# ───────────────────────────────────────────── benchmark loader

def normalize_code(s: str) -> str:
    """Strip markdown fences + common boilerplate from a model response."""
    if not s:
        return ""
    s = s.strip()
    if s.startswith("```"):
        # Strip ```python or ``` opener
        s = re.sub(r"^```(?:python|py)?\s*\n?", "", s)
        # Strip trailing fence
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def make_repobench_verify(ground_truth: str):
    """Lenient code-aware substring verify.

    Tries:
      1. case-insensitive substring on the first 60 chars of ground truth.
      2. whitespace-collapsed match (handles formatting differences).
    Conservative — only credits matches that contain the GT keywords.
    """
    needle_full = (ground_truth or "").strip()
    needle = needle_full[:60].strip()
    needle_lo = needle.lower()
    needle_no_ws = "".join(needle.split()).lower()

    def verify(answer: str) -> bool:
        if not answer or not needle:
            return False
        normalized = normalize_code(answer).lower()
        if needle_lo in normalized:
            return True
        normalized_no_ws = "".join(normalized.split()).lower()
        if needle_no_ws and needle_no_ws in normalized_no_ws:
            return True
        return False

    return verify


def _flatten_context_field(value) -> str:
    """RepoBench's `context` field can be str, list[str], or list[dict-or-tuple].

    Normalizes any of those to a single string. List entries are joined with
    blank lines; dicts use a 'path'+'content' or 'filepath'+'code' shape.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                p = item.get("path") or item.get("filepath") or item.get("file") or ""
                c = (item.get("content") or item.get("code")
                     or item.get("text") or "")
                parts.append(f"# {p}\n{c}" if p else str(c))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                p, c = item
                parts.append(f"# {p}\n{c}")
            else:
                parts.append(str(item))
        return "\n\n".join(p for p in parts if p)
    return str(value)


def load_repobench(n_tasks: int, split: str, seed: int = 42) -> list[dict]:
    """Load + adapt RepoBench Python cross_file_* into our task dict format.

    Stratifies by repo so no single repository dominates. Returns at most
    n_tasks dicts with keys: id, repo, question, context, ground_truth, verify.
    """
    from datasets import load_dataset
    ds = load_dataset("tianyang/repobench_python_v1.1", split=split)
    print(f"[load] tianyang/repobench_python_v1.1 :{split}  n_total={len(ds)}", flush=True)

    rng = random.Random(seed)
    by_repo: dict[str, list[dict]] = {}
    skipped = {"no_next_line": 0, "no_cropped": 0, "no_context": 0}
    inspected_first = False
    for row in ds:
        d = dict(row)
        if not inspected_first:
            print(f"[load] field types: " + ", ".join(
                f"{k}={type(v).__name__}" for k, v in d.items()
            ), flush=True)
            inspected_first = True
        gt = (d.get("next_line") or "").strip()
        if not gt:
            skipped["no_next_line"] += 1
            continue
        in_file = _flatten_context_field(
            d.get("cropped_code") or d.get("all_code") or ""
        )
        if not in_file.strip():
            skipped["no_cropped"] += 1
            continue
        cross_file = _flatten_context_field(d.get("context"))
        if not cross_file.strip():
            skipped["no_context"] += 1
            continue
        repo = d.get("repo_name") or f"repo_{len(by_repo)}"
        question = (
            "Complete the next line of this Python code. Return ONLY the next line "
            "of code (no commentary, no explanation, no markdown).\n\n"
            f"In-file context (the cursor is at the end):\n{in_file[-2000:]}"
        )
        t = {
            "repo": repo,
            "question": question,
            "context": cross_file,
            "ground_truth": gt,
        }
        t["key"] = task_key(t)
        t["verify"] = make_repobench_verify(gt)
        by_repo.setdefault(repo, []).append(t)
    if any(skipped.values()):
        print(f"[load] skipped during adapter: {skipped}", flush=True)

    # Stratified sample.
    n_repos = len(by_repo) or 1
    per_repo = max(1, n_tasks // n_repos)
    for tlist in by_repo.values():
        rng.shuffle(tlist)

    subset: list[dict] = []
    # Round-robin take so smaller repos aren't starved.
    while len(subset) < n_tasks:
        added_any = False
        for tlist in by_repo.values():
            if not tlist:
                continue
            if sum(1 for t in subset if t["repo"] == tlist[0]["repo"]) < per_repo and tlist:
                subset.append(tlist.pop(0))
                added_any = True
                if len(subset) >= n_tasks:
                    break
        if not added_any:
            break
    subset = subset[:n_tasks]
    for i, t in enumerate(subset):
        t["id"] = i + 1

    print(f"[load] adapted {len(subset)} tasks from {len(by_repo)} repos "
          f"(per-repo cap {per_repo})", flush=True)
    print(f"[load] per-repo distribution: {dict(Counter(t['repo'] for t in subset))}",
          flush=True)
    return subset


# ───────────────────────────────────────────── checkpoint helpers

def load_cache(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ───────────────────────────────────────────── NLI clustering (fp32 DeBERTa)

class NLIClusterer:
    """Bidirectional-entailment clustering with DeBERTa-large-MNLI in fp32.

    DeBERTa attention is not fp16-safe (silent NaN). Inference is wrapped
    in torch.autocast(enabled=False).
    """

    def __init__(self, model_name: str = "microsoft/deberta-large-mnli"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name).float().cuda()
        self.model.eval()
        labels = self.model.config.id2label
        self.entail_idx = next(i for i, l in labels.items() if "entail" in l.lower())

    def entail(self, premise: str, hypothesis: str) -> float:
        if premise.strip() == hypothesis.strip():
            return 1.0
        torch = self.torch
        enc = self.tok(premise, hypothesis, return_tensors="pt",
                       truncation=True, max_length=512).to("cuda")
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            logits = self.model(**enc).logits
        probs = torch.softmax(logits.float(), dim=-1)
        return float(probs[0, self.entail_idx])

    def cluster(self, samples: list[str], threshold: float = 0.5) -> list[list[int]]:
        n = len(samples)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(n):
            for j in range(i + 1, n):
                if (self.entail(samples[i], samples[j]) >= threshold
                        and self.entail(samples[j], samples[i]) >= threshold):
                    union(i, j)
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        return list(groups.values())

    def free(self) -> None:
        try:
            del self.model
            self.torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass


def shannon_entropy_bits(sizes: list[int]) -> float:
    total = sum(sizes)
    if total <= 0:
        return 0.0
    return -sum((s / total) * math.log2(s / total) for s in sizes if s > 0)


def nli_features(samples: list[str], clusterer: NLIClusterer) -> dict:
    clusters = clusterer.cluster(samples)
    sizes = [len(c) for c in clusters]
    se_bits = shannon_entropy_bits(sizes)
    max_bits = math.log2(len(samples)) if len(samples) > 1 else 1.0
    return {
        "nli_se": se_bits,
        "nli_se_norm": se_bits / max_bits if max_bits > 0 else 0.0,
        "nli_n_clusters": len(clusters),
        "nli_largest_cluster_frac": max(sizes) / len(samples) if samples else 0.0,
    }


# ───────────────────────────────────────────── per-task pipeline

def run_phase1_never_retrieve(subset, model, tokenizer, code_ids, lang_ids,
                               cache_path: Path, max_new_tokens: int = 200):
    cache = load_cache(cache_path)
    t0 = time.time()
    for i, t in enumerate(subset, 1):
        k = t["key"]
        if k in cache:
            continue
        prompt = build_chat_prompt(tokenizer, t["repo"], t["question"], context=None)
        extractor = CCEFeatureExtractor(code_ids=code_ids, lang_ids=lang_ids)
        answer, feats = generate_with_features(
            model, tokenizer, prompt, extractor, max_new_tokens=max_new_tokens)
        cache[k] = {
            "key": k,
            "id": t["id"],
            "repo": t["repo"],
            "question_preview": t["question"][:200],
            "answer": answer,
            "correct": bool(t["verify"](answer)),
            "features": feats,
        }
        if i % 5 == 0 or i == len(subset):
            save_cache(cache_path, cache)
            elapsed = (time.time() - t0) / 60
            n_correct = sum(1 for v in cache.values() if v.get("correct"))
            print(f"  phase1 {i}/{len(subset)}  ({elapsed:.1f} min)  "
                  f"running acc {n_correct}/{len(cache)} = "
                  f"{n_correct/max(len(cache),1):.3f}", flush=True)
    return cache


def run_phase2_always_retrieve(subset, model, tokenizer, embedder,
                                cache_path: Path,
                                max_ctx_chars: int = 16000,
                                max_new_tokens: int = 200):
    cache = load_cache(cache_path)
    t0 = time.time()
    for i, t in enumerate(subset, 1):
        k = t["key"]
        if k in cache:
            continue
        ctx_raw = t.get("context") or ""
        if not ctx_raw:
            ctx = None
        elif len(ctx_raw) <= max_ctx_chars:
            ctx = ctx_raw
        else:
            chunks, embs = build_chunks_and_index({t["repo"]: ctx_raw}, embedder,
                                                   chunk_size=80)
            ctx = retrieve(t["question"], chunks, embs, embedder, top_k=3,
                           target_file=None)
        prompt = build_chat_prompt(tokenizer, t["repo"], t["question"], context=ctx)
        answer = generate_simple(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        cache[k] = {
            "key": k,
            "id": t["id"],
            "repo": t["repo"],
            "answer": answer,
            "correct": bool(t["verify"](answer)),
            "context_chars_used": len(ctx) if ctx else 0,
        }
        if i % 5 == 0 or i == len(subset):
            save_cache(cache_path, cache)
            elapsed = (time.time() - t0) / 60
            n_correct = sum(1 for v in cache.values() if v.get("correct"))
            print(f"  phase2 {i}/{len(subset)}  ({elapsed:.1f} min)  "
                  f"always-retrieve acc {n_correct}/{len(cache)} = "
                  f"{n_correct/max(len(cache),1):.3f}", flush=True)
    return cache


def run_phase4_se_samples(filtered, model, tokenizer, embedder,
                           cache_path: Path,
                           n_samples: int = 5,
                           max_new_tokens: int = 120,
                           temperature: float = 0.7):
    cache = load_cache(cache_path)
    t0 = time.time()
    for i, t in enumerate(filtered, 1):
        k = t["key"]
        if k in cache:
            continue
        prompt = build_chat_prompt(tokenizer, t["repo"], t["question"], context=None)
        samples = sample_generations(model, tokenizer, prompt, n=n_samples,
                                     max_new_tokens=max_new_tokens,
                                     temperature=temperature)
        emb_se = semantic_entropy(samples, embedder, sim_threshold=0.85)
        cache[k] = {
            "key": k,
            "id": t["id"],
            "samples": samples,
            "embedding_se": emb_se,
        }
        if i % 5 == 0 or i == len(filtered):
            save_cache(cache_path, cache)
            elapsed = (time.time() - t0) / 60
            print(f"  phase4 SE samples {i}/{len(filtered)}  ({elapsed:.1f} min)",
                  flush=True)
    return cache


def run_phase5_nli(filtered, se_cache, cache_path: Path):
    cache = load_cache(cache_path)
    todo = [t for t in filtered if t["key"] not in cache and t["key"] in se_cache]
    if not todo:
        print(f"  phase5 NLI: {len(cache)} cached, nothing to do", flush=True)
        return cache
    clusterer = NLIClusterer()
    t0 = time.time()
    for i, t in enumerate(todo, 1):
        k = t["key"]
        samples = se_cache[k]["samples"]
        feats = nli_features(samples, clusterer)
        cache[k] = {"key": k, "id": t["id"], **feats}
        if i % 5 == 0 or i == len(todo):
            save_cache(cache_path, cache)
            print(f"  phase5 NLI {i}/{len(todo)}  ({(time.time()-t0)/60:.1f} min)",
                  flush=True)
    clusterer.free()
    return cache


# ───────────────────────────────────────────── analysis: AUC + bootstrap

def bootstrap_auc(scores: np.ndarray, labels: np.ndarray,
                   n_iter: int = 1000, alpha: float = 0.05,
                   seed: int = 42) -> dict:
    """Percentile-bootstrap CI for ROC-AUC.

    Returns {auc, ci_lo, ci_hi}. NaN if degenerate.
    """
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2 or len(scores) < 5:
        return {"auc": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    try:
        point = float(roc_auc_score(labels, scores))
    except ValueError:
        return {"auc": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}

    rng = np.random.default_rng(seed)
    n = len(scores)
    boots = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        s = scores[idx]
        l = labels[idx]
        if len(np.unique(l)) < 2:
            continue
        try:
            boots.append(roc_auc_score(l, s))
        except ValueError:
            continue
    if not boots:
        return {"auc": point, "ci_lo": float("nan"), "ci_hi": float("nan")}
    boots = np.array(boots)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"auc": point, "ci_lo": lo, "ci_hi": hi, "n_bootstrap": len(boots)}


def loo_lr_oof_scores(features: list[dict], feat_keys: tuple,
                       labels: np.ndarray) -> np.ndarray:
    """Leave-one-out logistic regression. Returns out-of-fold positive-class probs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut

    X = np.array([[f.get(k, 0.0) for k in feat_keys] for f in features], dtype=float)
    n = len(X)
    out = np.zeros(n)
    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        y_tr = labels[tr]
        if len(np.unique(y_tr)) < 2:
            out[te[0]] = float(y_tr[0])
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X[tr], y_tr)
        out[te[0]] = float(clf.predict_proba(X[te])[0, 1])
    return out


def gated_retrieval_curve(scores: np.ndarray,
                           init_correct: np.ndarray,
                           always_correct: np.ndarray,
                           n_thresholds: int = 50) -> list[dict]:
    """Sweep thresholds; report (retrieval_rate, accuracy) at each.

    Decision: gate fires iff score >= threshold. If fires → use always-retrieve
    answer (correct iff always_correct=1). If not → use never-retrieve answer
    (correct iff init_correct=1).
    """
    if len(scores) == 0:
        return []
    # Use unique score quantiles + extremes for thresholds.
    thresholds = np.unique(np.concatenate([
        np.array([-np.inf]),
        np.quantile(scores, np.linspace(0, 1, n_thresholds)),
        np.array([np.inf]),
    ]))
    curve = []
    for T in thresholds:
        fires = scores >= T
        final = (fires & always_correct) | (~fires & init_correct)
        curve.append({
            "threshold": float(T) if np.isfinite(T) else (1e9 if T > 0 else -1e9),
            "retrieval_rate": float(fires.mean()),
            "accuracy": float(final.mean()),
        })
    return curve


# ───────────────────────────────────────────── CLI

def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--n-tasks", type=int, default=250)
    p.add_argument("--split", default="cross_file_first")
    p.add_argument("--out-dir", default="/content/drive/MyDrive/cce_poc_v5")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--max-new-tokens-sample", type=int, default=120)
    p.add_argument("--max-ctx-chars", type=int, default=16000)
    p.add_argument("--n-samples-se", type=int, default=5)
    p.add_argument("--se-temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(args.model)

    # Per-model cache paths (content-stable keys mean they survive re-runs).
    P1 = out_dir / f"phase1_never_retrieve__{slug}.json"
    P2 = out_dir / f"phase2_always_retrieve__{slug}.json"
    P4 = out_dir / f"phase4_se_samples__{slug}.json"
    P5 = out_dir / f"phase5_nli__{slug}.json"
    RESULTS = out_dir / f"results__{slug}.json"

    print(f"\n{'='*70}\nCCE PoC v5 — {args.model}\n{'='*70}", flush=True)

    # ── Load benchmark ────────────────────────────────────
    subset = load_repobench(n_tasks=args.n_tasks, split=args.split, seed=args.seed)
    if not subset:
        print("[fatal] no tasks loaded", flush=True)
        return 1

    # ── Load model ────────────────────────────────────────
    model, tokenizer, embedder = setup_model_and_embedder(args.model)
    code_ids, lang_ids = build_partition_indices(tokenizer, partition="lenient")
    print(f"[setup] model loaded; |code_ids|={len(code_ids)}, "
          f"|lang_ids|={len(lang_ids)}", flush=True)

    # ── Phase 1: never-retrieve ──────────────────────────
    print(f"\n[phase 1] never-retrieve generation (n={len(subset)})", flush=True)
    phase1 = run_phase1_never_retrieve(
        subset, model, tokenizer, code_ids, lang_ids,
        cache_path=P1, max_new_tokens=args.max_new_tokens,
    )

    # ── Phase 2: always-retrieve ─────────────────────────
    print(f"\n[phase 2] always-retrieve generation", flush=True)
    phase2 = run_phase2_always_retrieve(
        subset, model, tokenizer, embedder,
        cache_path=P2,
        max_ctx_chars=args.max_ctx_chars,
        max_new_tokens=args.max_new_tokens,
    )

    # ── Phase 3: filter ──────────────────────────────────
    filtered = [t for t in subset
                if t["key"] in phase2 and phase2[t["key"]].get("correct")]
    n_total = len(subset)
    n_filtered = len(filtered)
    print(f"\n[phase 3] filter to always_correct=1: "
          f"{n_filtered}/{n_total} = {n_filtered/n_total:.1%}", flush=True)
    if n_filtered < 5:
        print("[fatal] filtered subset too small for meaningful analysis", flush=True)
        return 2

    # ── Phase 4: SE samples ──────────────────────────────
    print(f"\n[phase 4] SE samples on filtered subset", flush=True)
    phase4 = run_phase4_se_samples(
        filtered, model, tokenizer, embedder,
        cache_path=P4,
        n_samples=args.n_samples_se,
        max_new_tokens=args.max_new_tokens_sample,
        temperature=args.se_temperature,
    )

    # Free the LM before loading DeBERTa.
    try:
        import torch
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass

    # ── Phase 5: NLI clustering ──────────────────────────
    print(f"\n[phase 5] NLI clustering (DeBERTa-large-MNLI fp32)", flush=True)
    phase5 = run_phase5_nli(filtered, phase4, cache_path=P5)

    # ── Build per-task feature dicts (filtered subset) ───
    feats_by_key: dict[str, dict] = {}
    for t in filtered:
        k = t["key"]
        f = dict(phase1[k]["features"])  # SAPLMA + FLARE + CCE
        emb_d = phase4.get(k, {}).get("embedding_se", {})
        for ek in EMB_SE_FEATURES:
            f[ek] = float(emb_d.get(ek, 0.0))
        nli_d = phase5.get(k, {})
        for nk in NLI_SE_FEATURES:
            f[nk] = float(nli_d.get(nk, 0.0))
        feats_by_key[k] = f

    # Labels: needs_retrieval = NOT initial_correct (always_correct is 1 by filter).
    keys_ordered = [t["key"] for t in filtered]
    init_correct = np.array([phase1[k]["correct"] for k in keys_ordered], dtype=bool)
    always_correct_filt = np.ones(len(keys_ordered), dtype=bool)
    needs_retrieval = (~init_correct).astype(int)
    n_pos = int(needs_retrieval.sum())

    print(f"\n[analysis] filtered n={len(keys_ordered)}, "
          f"needs_retrieval positives={n_pos} "
          f"({n_pos/len(keys_ordered):.1%})", flush=True)

    # ── Phase 7: per-arm AUC + bootstrap CI ──────────────
    print(f"\n[phase 7] per-arm classification AUC on filtered subset "
          f"(bootstrap n={args.bootstrap_iters})", flush=True)

    # Univariate signals.
    univariate_results = {}
    for key, name in UNIVARIATE_SIGNALS:
        scores = np.array([feats_by_key[k].get(key, np.nan) for k in keys_ordered])
        if np.isnan(scores).any() or len(np.unique(scores)) < 2:
            continue
        boot = bootstrap_auc(scores, needs_retrieval, n_iter=args.bootstrap_iters)
        univariate_results[key] = {
            "name": name,
            **boot,
        }

    # Multivariate arms: LOO-LR + bootstrap on OOF probs.
    arm_results = {}
    feat_list = [feats_by_key[k] for k in keys_ordered]
    for arm_name, fkeys in ARMS.items():
        scores = loo_lr_oof_scores(feat_list, fkeys, needs_retrieval)
        boot = bootstrap_auc(scores, needs_retrieval, n_iter=args.bootstrap_iters)
        arm_results[arm_name] = {
            "n_features": len(fkeys),
            **boot,
            "_oof_scores": scores.tolist(),
        }

    # ── Phase 8: end-to-end gated retrieval curves ───────
    print(f"\n[phase 8] gated-retrieval cost/accuracy curves", flush=True)
    curves = {}
    # Univariate signals.
    for key, name in UNIVARIATE_SIGNALS:
        scores = np.array([feats_by_key[k].get(key, np.nan) for k in keys_ordered])
        if np.isnan(scores).any():
            continue
        curves[f"univariate__{key}"] = gated_retrieval_curve(
            scores, init_correct, always_correct_filt)
    # Multivariate arms: use the LOO-LR OOF scores stashed above.
    for arm_name in ARMS:
        scores = np.array(arm_results[arm_name]["_oof_scores"])
        curves[f"arm__{arm_name}"] = gated_retrieval_curve(
            scores, init_correct, always_correct_filt)

    # Strip the OOF scores from the saved arm summary to keep file small.
    for arm in arm_results.values():
        arm.pop("_oof_scores", None)

    # ── Save results ─────────────────────────────────────
    overall = {
        "never_retrieve_accuracy": float(np.mean(
            [phase1[t["key"]]["correct"] for t in subset if t["key"] in phase1])),
        "always_retrieve_accuracy": float(np.mean(
            [phase2[t["key"]]["correct"] for t in subset if t["key"] in phase2])),
    }
    overall["gap_pp"] = (overall["always_retrieve_accuracy"]
                          - overall["never_retrieve_accuracy"]) * 100

    out = {
        "version": "v5",
        "model": args.model,
        "model_slug": slug,
        "benchmark": f"tianyang/repobench_python_v1.1 :{args.split}",
        "n_total": n_total,
        "n_filtered": n_filtered,
        "filter_retention_rate": n_filtered / max(n_total, 1),
        "overall": overall,
        "filtered_subset": {
            "n": n_filtered,
            "n_positives": n_pos,
            "positive_rate": n_pos / max(n_filtered, 1),
            "never_retrieve_accuracy": float(init_correct.mean()),
        },
        "univariate_auc": univariate_results,
        "arm_auc": arm_results,
        "gated_retrieval_curves": curves,
        "config": {
            "n_tasks": args.n_tasks,
            "split": args.split,
            "max_new_tokens": args.max_new_tokens,
            "n_samples_se": args.n_samples_se,
            "se_temperature": args.se_temperature,
            "max_ctx_chars": args.max_ctx_chars,
            "bootstrap_iters": args.bootstrap_iters,
            "seed": args.seed,
        },
    }
    save_cache(RESULTS, out)
    print(f"\n[done] wrote {RESULTS}", flush=True)

    # ── Print summary ────────────────────────────────────
    print(f"\n=== summary: {args.model} ===")
    print(f"  overall: never={overall['never_retrieve_accuracy']:.3f}, "
          f"always={overall['always_retrieve_accuracy']:.3f}, "
          f"gap={overall['gap_pp']:+.1f}pp")
    print(f"  filtered subset: n={n_filtered}, positives={n_pos} "
          f"({n_pos/max(n_filtered,1):.1%})")
    print(f"\n  Univariate AUC (95% CI):")
    for key, r in univariate_results.items():
        print(f"    {r['name']:<20} {r['auc']:.3f} [{r['ci_lo']:.3f}, "
              f"{r['ci_hi']:.3f}]")
    print(f"\n  Arm AUC (LOO-LR, 95% CI):")
    for arm, r in arm_results.items():
        print(f"    {arm:<22} {r['auc']:.3f} [{r['ci_lo']:.3f}, "
              f"{r['ci_hi']:.3f}]  ({r['n_features']} features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
