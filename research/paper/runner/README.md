# CCE Final-Paper Runner

Encodes `research/paper/PROTOCOL.md` as runnable code. Produces
`research/paper/paper_results.json` — the canonical, reviewer-reproducible
results file. Every paper figure and table is generated from this file.

## Quick reference

```bash
# 1. Estimate cost. Always run first.
python research/run_final_experiments.py \
    --benchmark research/benchmarks/benchmark_v2_final.json \
    --grid all --estimate

# 2. Dry-run. Mandatory before the real run. Heuristic judge, no GPU needed.
python research/run_final_experiments.py \
    --benchmark research/benchmarks/benchmark_v2_final.json \
    --grid headline --dry-run --no-load-model

# 3. Real run. Resumable across Colab disconnects.
python research/run_final_experiments.py \
    --benchmark research/benchmarks/benchmark_v2_final.json \
    --out research/paper/paper_results.json \
    --grid all
```

## Files

| File | Purpose |
|------|---------|
| `research/run_final_experiments.py` | Entrypoint. ArgParse, main loop, orchestration. |
| `research/paper/runner/env.py` | Determinism + environment fingerprinting + git SHA. |
| `research/paper/runner/checkpoint.py` | Atomic, resumable JSON checkpoint store. |
| `research/paper/runner/judge.py` | LLM-as-judge with content-hash cache. |
| `research/paper/runner/grid.py` | Config grid expansion (headline / cross-section sensitivity / factorial). |
| `research/paper/runner/methods/base.py` | `Question`, `Result`, `MethodContext`, `BaseMethod` ABC. |
| `research/paper/runner/methods/baselines.py` | 6 working baselines (no_context, random, bm25, embedding, full_context, reposynth_base). |
| `research/paper/runner/methods/cce_methods.py` | uncert_cot / cce_adaptive / cce_learned — stubs awaiting wiring. |

## What's implemented vs what needs wiring

**Working** (verified end-to-end via `--dry-run`):
- Argument parsing & cost estimation
- Atomic checkpoint with resume-on-restart (every result row survives a Colab disconnect)
- Determinism setup across `random`, `numpy`, `torch`, `transformers`
- LLM-as-judge cache keyed by `sha256(model, prompt, question, generated, gold)` → judges each unique answer at most once across all methods/seeds/configs
- Grid expansion: 27 headline configs + 42 cross-section sensitivity configs (vs 324 if factorial)
- Failure handling: a method that crashes records a `status="failed"` row, run continues
- Heartbeat logging + periodic flush

**Stubs requiring lift from existing notebooks**:
- `methods/cce_methods.py::CCEAdaptiveMethod` → lift from `Week15_V6_Balanced_Tasks.ipynb`. The canonical functions are `generate_with_features` (cell 11) and `generate_simple` (cell 12); plus `retrieve` (cell 9) and `create_chunks` (cell 8). V6 is the version that produced the 0.95 headline result, so it's the version that must reproduce. **Verification step:** before declaring the migration done, re-run the migrated method on V6's exact 20-task set and confirm accuracy reproduces to within seed noise. **Audit step:** V6 doesn't literally name `H_code` / `H_lang` variables — confirm the actual entropy formula matches the paper's claim of "H_code − H_lang" before relying on it. If V6 computes a different signal under a different name, the paper claim has to be reworded.
- `methods/cce_methods.py::CCELearnedMethod` → lift `LearnedQueryPooler` from `Week9_V3_Learned_Query_Integration.ipynb` cell 13; train per PROTOCOL §5.1
- `methods/cce_methods.py::UncertCoTMethod` → lift line-boundary trigger from `add_cce_adaptive_features.py`
- `run_final_experiments.py::load_repo_files` — currently returns empty dict; wire to the existing repo cloner / file walker
- `methods/baselines.py::ReposynthBaseMethod._hybrid_retrieve` — currently returns BM25∪embedding; wire to `packages/python-orchestrator/orchestrator/retriever.py` if you want the original hybrid

Each stub has a clear pointer to where the working code already lives.

## Cost envelope (with cross-section sensitivity, n=100, 3 seeds)

| Item | Value |
|------|-------|
| Headline configs | 27 |
| Sensitivity configs | 42 |
| **Total configs** | **69** |
| Total tuples (× 100 questions) | 6,900 |
| Wall-clock @ 10s/gen | ~19 hours (fits in one Colab Pro session with checkpoint safety) |
| Judge API cost (gpt-4o-mini, ~40% cache miss rate) | ~$0.40 |

Use `--factorial` to opt into the full Cartesian sensitivity sweep (324 configs,
~97h wall-clock). Not recommended.

## Output schema

See PROTOCOL §9.3. Roughly: every row has `(method, seed, tau, partition, layers,
max_hops, qid, status, correctness, retrievals, context_p/r/f1, tokens_used,
tokens_generated, tokens_retrieved, hallucination, latency_ms, peak_vram_mb,
cce_trace, retrieved_files, generated_text, judge_score, judge_explanation,
judge_cached)`.

Aggregation (means, CIs, p-values) is **post-hoc** from raw rows — never written
into the JSON. This is by design: re-aggregation is always reproducible without
re-running experiments. Use `research/paper/make_figures.py` (TODO) and
`research/paper/run_significance_tests.py` (TODO) for downstream artifacts.

## Pre-flight before the real Colab run

Per PROTOCOL §9.5:

- [ ] PROTOCOL.md locked at v1.0 (committed)
- [ ] `benchmark_v2_final.json` exists and is committed
- [ ] All three CCE method stubs replaced with working implementations
- [ ] `load_repo_files` wired up
- [ ] `pytest research/integration_tests/` passes
- [ ] `--estimate` reviewed
- [ ] `--dry-run --no-load-model` passes (full grid, all methods, no failures)
- [ ] `--dry-run` (with model loaded) passes on 2 questions/repo
- [ ] Drive mounted on Colab for `--out` and `--judge-cache` (so checkpoint survives disconnect)
- [ ] `OPENAI_API_KEY` in env
- [ ] HuggingFace token loaded for CodeLlama gated access
