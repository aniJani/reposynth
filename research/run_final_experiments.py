#!/usr/bin/env python3
"""CCE final experiment runner. See research/paper/PROTOCOL.md §9.

Usage:
    python research/run_final_experiments.py \\
        --benchmark research/benchmarks/benchmark_v2_final.json \\
        --out       research/paper/paper_results.json \\
        --grid      headline \\
        --seeds     42,1337,2024

Modes:
    --estimate    Print projected wall-clock + judge-API cost; exit.
    --dry-run     Run pipeline against a tiny subset, no LLM judge calls (heuristic only).
    (default)     Real run. Resumes from any existing checkpoint.

Always:
  - Atomic per-tuple checkpoints (resumable across Colab disconnects).
  - Strict determinism per seed.
  - LLM-as-judge cache by content hash (judge each unique answer once, ever).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.paper.runner import env as env_mod
from research.paper.runner import grid as grid_mod
from research.paper.runner.checkpoint import Checkpoint, make_key
from research.paper.runner.judge import Judge, prompt_sha
from research.paper.runner.methods import METHODS, MethodContext, Question


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True, help="path to benchmark_vN.json")
    p.add_argument("--out", default="research/paper/paper_results.json")
    p.add_argument("--protocol", default="research/paper/PROTOCOL.md")
    p.add_argument("--grid", choices=["headline", "sensitivity", "all"], default="headline")
    p.add_argument("--methods", default="all", help="comma-list or 'all'")
    p.add_argument("--seeds", default="42,1337,2024")
    p.add_argument("--max-questions", type=int, default=None, help="debug: cap questions per repo")
    p.add_argument("--factorial", action="store_true",
                   help="use full factorial sensitivity sweep instead of cross-sections (much more compute)")
    p.add_argument("--estimate", action="store_true", help="print cost estimate and exit")
    p.add_argument("--dry-run", action="store_true", help="2 questions/repo, heuristic judge")
    p.add_argument("--judge-model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--judge-cache", default="research/paper/judge_cache.json")
    p.add_argument("--per-question-timeout", type=float, default=120.0)
    p.add_argument("--flush-every-seconds", type=float, default=30.0)
    p.add_argument(
        "--no-load-model",
        action="store_true",
        help="skip LLM loading (for grid validation). Generations will be placeholders.",
    )
    return p.parse_args()


def load_benchmark(path: str) -> list[Question]:
    with open(path) as f:
        data = json.load(f)
    raw_examples = data.get("examples", data) if isinstance(data, dict) else data
    out = []
    for e in raw_examples:
        if "repo" not in e:
            for repo_hint in ("flask", "fastapi", "requests", "httpx"):
                if repo_hint in (e.get("id") or "").lower() or repo_hint in (e.get("query") or "").lower():
                    e = {**e, "repo": repo_hint}
                    break
            else:
                e = {**e, "repo": "unknown"}
        out.append(Question.from_dict(e))
    return out


def filter_questions(qs: list[Question], max_per_repo: int | None) -> list[Question]:
    if max_per_repo is None:
        return qs
    by_repo: dict[str, list[Question]] = {}
    for q in qs:
        by_repo.setdefault(q.repo, []).append(q)
    out = []
    for repo, lst in by_repo.items():
        out.extend(lst[:max_per_repo])
    return out


def build_context(args, repo_files: dict[str, str], no_judge_call: bool) -> MethodContext:
    """Load model, embeddings, BM25 — once per run."""
    if args.no_load_model or args.dry_run:
        return MethodContext(
            model=None,
            tokenizer=None,
            embedder=None,
            repo_files=repo_files,
            embeddings_index=_StubIndex(repo_files),
            bm25_index=_StubBM25(repo_files),
            token_classifier=None,
        )

    print("[setup] loading CodeLlama-7B-Instruct (4-bit NF4)...", flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
    model = AutoModelForCausalLM.from_pretrained(
        "codellama/CodeLlama-7b-Instruct-hf",
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    print("[setup] building BM25 + embedding indices...", flush=True)
    bm25 = _build_bm25(repo_files)
    emb = _build_embeddings(repo_files)

    return MethodContext(
        model=model,
        tokenizer=tok,
        embedder=None,
        repo_files=repo_files,
        embeddings_index=emb,
        bm25_index=bm25,
        token_classifier=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def load_repo_files(repos: list[str]) -> dict[str, str]:
    """Stub: in production, clone+read each repo; return {file_path: content}.
    For now, returns empty so dry-run / grid validation can proceed."""
    return {}


def main() -> int:
    args = parse_args()

    methods = list(METHODS) if args.methods == "all" else [m.strip() for m in args.methods.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"[main] loading benchmark: {args.benchmark}", flush=True)
    questions = load_benchmark(args.benchmark)
    if args.dry_run:
        questions = filter_questions(questions, max_per_repo=2)
    elif args.max_questions is not None:
        questions = filter_questions(questions, max_per_repo=args.max_questions)
    print(f"[main] {len(questions)} questions across {len(set(q.repo for q in questions))} repos", flush=True)

    configs = grid_mod.expand(args.grid, methods, seeds, factorial=args.factorial)
    cost = grid_mod.estimate_cost(configs, n_questions=len(questions))
    print(f"[main] grid={args.grid}  configs={cost['n_configs']}  tuples={cost['n_tuples_total']}  "
          f"~{cost['wall_hours_est']}h  ~${cost['judge_usd_est']} judge", flush=True)

    if args.estimate:
        print(json.dumps(cost, indent=2))
        return 0

    cp = Checkpoint(args.out)
    repo_set = sorted({q.repo for q in questions})
    repo_files = load_repo_files(repo_set)

    cp.set_metadata(
        protocol_sha=env_mod.git_sha(args.protocol) or env_mod.file_sha256(args.protocol),
        benchmark_sha=env_mod.git_sha(args.benchmark) or env_mod.file_sha256(args.benchmark),
        code_sha=env_mod.git_sha(__file__),
        env=env_mod.record_environment(),
        model={
            "name": "codellama/CodeLlama-7b-Instruct-hf",
            "quantization": "nf4-bnb-bf16",
            "judge_model": args.judge_model,
            "judge_prompt_sha": prompt_sha(),
        },
        grid_mode=args.grid,
        seeds=seeds,
        methods=methods,
        n_questions=len(questions),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        dry_run=args.dry_run,
    )

    judge = Judge(model=args.judge_model, cache_path=args.judge_cache)
    ctx = build_context(args, repo_files, no_judge_call=args.dry_run)

    n_total = len(configs) * len(questions)
    n_done_at_start = cp.n_results()
    print(f"[main] {n_done_at_start}/{n_total} tuples already done — resuming.", flush=True)

    last_heartbeat = 0.0
    n_processed = 0
    n_failed = 0
    n_skipped = 0

    try:
        for cfg in configs:
            env_mod.setup_determinism(cfg.seed)
            method_cls = METHODS[cfg.method]
            cfg_dict = {**cfg.to_dict(), "seed": cfg.seed}
            method = method_cls(ctx, cfg_dict)

            for q in questions:
                key = make_key({**cfg.to_dict(), "qid": q.qid})
                if cp.has(key):
                    n_skipped += 1
                    continue

                try:
                    result = _run_one(method, q, judge, args)
                except Exception as e:
                    traceback.print_exc()
                    result = method._empty_result(q.qid, cfg.seed)
                    result.status = "failed"
                    result.error = f"{type(e).__name__}: {e}"
                    n_failed += 1

                cp.add(result.to_row())
                n_processed += 1

                if time.time() - last_heartbeat > args.flush_every_seconds:
                    last_heartbeat = time.time()
                    print(
                        f"[hb] method={cfg.method} seed={cfg.seed} done={n_processed} "
                        f"skipped={n_skipped} failed={n_failed} cached_judges={judge.n_cached()}",
                        flush=True,
                    )
                    cp.maybe_flush(every_seconds=0)
                    judge.maybe_flush(every_seconds=0)
    finally:
        cp.set_metadata(
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            n_processed_this_session=n_processed,
            n_skipped_this_session=n_skipped,
            n_failed_this_session=n_failed,
        )
        cp.flush()
        judge.close()

    print(
        f"[main] DONE  processed={n_processed}  skipped={n_skipped}  failed={n_failed}  "
        f"total_in_file={cp.n_results()}/{n_total}",
        flush=True,
    )
    return 0 if n_failed == 0 else 2


def _run_one(method, q: Question, judge: Judge, args) -> "Result":
    t0 = time.time()
    result = method.run(q)
    if result.status != "ok":
        return result

    judged = judge.score(
        question=q.query,
        generated=result.generated_text or "",
        gold=q.gold_answer or "",
        offline=args.dry_run,
    )
    result.judge_score = judged["score"]
    result.judge_explanation = judged["explanation"]
    result.judge_cached = bool(judged.get("cached", False))
    result.correctness = result.judge_score

    elapsed = (time.time() - t0) * 1000
    if args.per_question_timeout and elapsed > args.per_question_timeout * 1000:
        result.status = "timeout"
        result.error = f"exceeded {args.per_question_timeout}s"
    return result


# ---------------------------------------------------------------- stubs


class _StubIndex:
    """Used in --dry-run / --no-load-model. Returns deterministic top-k."""

    def __init__(self, repo_files: dict[str, str]):
        self.paths = list(repo_files)

    def top_k(self, query: str, k: int = 5):
        return [(p, 1.0 / (i + 1)) for i, p in enumerate(self.paths[:k])]


class _StubBM25(_StubIndex):
    pass


def _build_bm25(repo_files: dict[str, str]):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return _StubBM25(repo_files)

    class _BM25:
        def __init__(self, files):
            self.paths = list(files)
            self.corpus = [files[p].lower().split() for p in self.paths]
            self.bm = BM25Okapi(self.corpus)

        def top_k(self, query: str, k: int = 5):
            scores = self.bm.get_scores(query.lower().split())
            ranked = sorted(zip(self.paths, scores), key=lambda x: -x[1])
            return ranked[:k]

    return _BM25(repo_files)


def _build_embeddings(repo_files: dict[str, str]):
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        return _StubIndex(repo_files)

    class _Emb:
        def __init__(self, files):
            self.paths = list(files)
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.vecs = self.model.encode(
                [files[p][:2000] for p in self.paths],
                show_progress_bar=False,
                normalize_embeddings=True,
            )

        def top_k(self, query: str, k: int = 5):
            q = self.model.encode([query], normalize_embeddings=True)[0]
            sims = self.vecs @ q
            order = sims.argsort()[::-1][:k]
            return [(self.paths[i], float(sims[i])) for i in order]

    return _Emb(repo_files)


if __name__ == "__main__":
    raise SystemExit(main())
