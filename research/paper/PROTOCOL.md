# CCE: Final Experimental Protocol

**Status:** DRAFT — locked once Q1–Q5 are resolved.
**Owner:** _your name_
**Last updated:** 2026-04-25

This document is the single source of truth for the final paper experiments.
Once committed, **no changes** without an explicit revision and version bump.
Every number in the paper must be reproducible from this protocol + the
benchmark + `run_final_experiments.py`.

---

## 1. Paper Identity

- **System name (paper-facing):** **CCE — Contrastive Code Entropy** *(confirmed)*
- **One-sentence claim:**
  > CCE — the difference between code-token entropy and language-token
  > entropy — is a training-free signal that detects when an LLM lacks code
  > context, and using it to trigger adaptive mid-generation retrieval
  > preserves answer quality while cutting retrieval cost.
- **Target venue:** **deliberately deferred.** Venue choice does not affect any experimental design decision in this document (model, baselines, metrics, dataset size, statistical protocol). Decision is post-results: strong results → submit higher; mixed → submit to a fit-aware venue. Locking it now imports fake constraints.
- **Target submission date:** **none yet.** Pace is set by the work, not a fictional deadline.
- **Codebase identifier:** `RepoSynth` (engineering substrate; not the paper's contribution)

---

## 2. Research Questions & Hypotheses

Carried over from `research/research_questions.md` (lines 16–235). Locked.

| ID  | Question / Hypothesis | Success criterion |
|-----|----------------------|-------------------|
| RQ1 | Can entropy-based uncertainty detect missing code context? | F1 ≥ 0.65 (P ≥ 0.70, R ≥ 0.60) |
| RQ2 | Does CCE outperform raw entropy? | F1(CCE) − F1(raw) ≥ 0.10 |
| RQ3 | Does adaptive retrieval preserve quality with fewer tokens? | accuracy ≥ 0.95 × always-retrieve **and** retrievals ≤ 0.70 × always-retrieve |
| RQ4 | Where in the token stream should entropy be measured? | F1 ≥ 0.90 × every-token, measurements < 0.30 × total |
| H1–H4 | (operationalized versions of RQ1–RQ4) | as above |

**Headline narrative:** *"CCE produces an efficiency-quality Pareto frontier that
dominates static-pre-retrieval baselines at tight retrieval budgets."* Every
figure and table in the paper must support or contextualize this claim.

---

## 3. Experimental Setup

### 3.1 Models

- **Generation + entropy model (CCE-eligible):** `codellama/CodeLlama-7b-Instruct-hf`
  - Loaded via `transformers.AutoModelForCausalLM.from_pretrained(...)` with **4-bit quantization** via `BitsAndBytesConfig` (NF4, double-quant, bf16 compute).
  - Architecture: 32 transformer layers, hidden_dim = 4096.
  - Same model is used for both generation and entropy probes — no decoupling. This is intentional: CCE requires logits from the *generating* model, otherwise the entropy signal is decoupled from what is actually generated.
  - HuggingFace revision SHA → record in `paper_results.json` (do not rely on `main`).
- **LLM-as-judge (for answer-correctness scoring):** `gpt-4o-mini` *(decision, see §4.1)*. Reason: open-weights judge would entangle the CCE-model with the judge; a well-calibrated commercial judge avoids that confound. Pin a specific snapshot date.
- **Embedding model (for `embedding` baseline + retrieval):** `sentence-transformers/all-MiniLM-L6-v2` (already in `retriever.py`).
- **Probe layers (CCE token-classifier inputs):** `[16, 24, 31]` (per `week8_results.json` config — these correspond to mid, upper-mid, and final transformer layer of CodeLlama-7B's 32-layer stack).
- **Generation hyperparameters:** `max_new_tokens=150, temperature=0.0, do_sample=False` (greedy decoding; reproducible). Document any deviation per-experiment.

### 3.2 Repositories

- **Primary corpus:** `httpx`, `Flask`, `FastAPI`, `Requests` (all already curated; see `research/scripts/curate_benchmark.py`)
- **Pinned commits:** record SHA in `benchmark_v2_final.json`
- **Why these:** widely-used Python libraries; `curate_benchmark.py` already
  generates Q&A from them; balances HTTP-client / web-framework diversity.

### 3.3 Benchmark

- **Filename:** `research/benchmarks/benchmark_v2_final.json`
- **Size target:** ≥ 100 questions, balanced across repos and difficulty
  (≥ 25 per repo, ≥ 30 easy, ≥ 40 medium, ≥ 30 hard)
- **Schema (per question):** carried over from `benchmark_v1.json` —
  `{id, query, ground_truth_answer, ground_truth_files, ground_truth_keywords,
  ground_truth_missing_positions, difficulty, category, hop_count}`
- **Construction:** `curate_benchmark.py` extended to 100; manual review of every
  question for ground-truth correctness; freeze and version.

### 3.4 Hardware

- **Compute environment:** **Colab Pro** (assume A100 40GB or V100 16GB; confirm at runtime and record in `paper_results.json`).
- **Memory budget:** CodeLlama-7B at 4-bit ≈ 4–5 GB VRAM, leaves ample headroom for KV-cache + retrieval embeddings on either GPU.
- **Wall-clock budget estimate** (back-of-envelope):
  - 100 questions × 9 methods × 3 seeds × (3 thresholds × 2 partitions × 3 layer-sets × 3 max-hops grid restricted to `cce_*` only) ≈ **~3,000 generations for headline + ~5,000 for sensitivity**.
  - At ~10s/generation on A100 → ~22 hours. On V100 → ~35 hours.
  - **Plan for resumability:** Colab Pro disconnects after 24h. Runner must checkpoint and resume (see §9).
- **Determinism:** `torch.use_deterministic_algorithms(True)`, `torch.manual_seed(seed)`, `transformers.set_seed(seed)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Record `torch.__version__`, `transformers.__version__`, CUDA, cuDNN, GPU model in every results JSON.

---

## 4. Headline Metric & Pareto Framing

The paper's lead figure is an **efficiency-quality Pareto curve**, *not* a
single-number leaderboard. Each method produces a point (or a curve, if it has
a tunable knob); we report the Pareto frontier.

### 4.1 Primary axes

- **Quality (y-axis):** *Answer Correctness* — LLM-as-judge score in [0, 1] using
  `[NEEDS Q1, judge model]` against `ground_truth_answer`. Aggregate = mean over
  benchmark.
- **Cost (x-axis):** *Retrieval count per question* — total number of retrieval
  invocations during generation (1 for static-pre-retrieval methods; variable for
  CCE-adaptive).

### 4.2 Secondary metrics (reported in tables, not on the lead figure)

- **Context Precision / Recall / F1** — retrieved files vs `ground_truth_files`
- **Token efficiency** — `1 − tokens_used / always_retrieve_tokens`
- **Hallucination rate** — proportion of generated claims unsupported by retrieved context (LLM-judge)
- **Latency overhead** — wall-clock seconds vs `no_context`
- **CCE-detection metrics (RQ1, RQ2 only):** P / R / F1 of spike-position alignment with `ground_truth_missing_positions`

### 4.3 What we explicitly do NOT use

- The "Composite" score from week8_results.json. Composite scores hide the trade-off; the trade-off *is* the contribution.

---

## 5. Baselines (Locked)

**Nine** methods, no additions, no removals after this commit.

| # | Method | Description | Retrieval count |
|---|--------|-------------|-----------------|
| 1 | `no_context` | Generate with query only, no retrieval | 0 |
| 2 | `random` | Retrieve k random files at start | 1 |
| 3 | `bm25` | BM25 over file content, top-k upfront | 1 |
| 4 | `embedding` | FAISS over MiniLM embeddings, top-k upfront | 1 |
| 5 | `full_context` | Concatenate every file under model context limit | 1 |
| 6 | `reposynth_base` | Hybrid keyword+semantic upfront retrieval | 1 |
| 7 | `uncert_cot` | Line-boundary uncertainty, raw entropy trigger (UnCert-CoT-style) | variable |
| 8 | **`cce_adaptive`** | **CCE = H_code − H_lang trigger; multi-hop on confused tokens; heuristic file scorer** | **variable** |
| 9 | `cce_learned` | Same trigger as `cce_adaptive`, but file scoring uses the learned cross-attention pooler (24M params) | variable |

Top-k for upfront methods: **k = 5** (locked).

**Q4 resolution: include `cce_learned` and re-run cleanly.** The Week 9 V3 vs Week 11
contradiction is a deliberate experimental question — does the learned scorer help or
hurt? The new run on the v2 benchmark, with identical training/eval splits and 3 seeds,
produces a definitive answer. No corner-cutting; it gets the same statistical treatment
as every other baseline.

### 5.1 Learned scorer training protocol (locked, since `cce_learned` is in)

- **Architecture:** unchanged from `Week9_V3_Learned_Query_Integration.ipynb` (cell 13) — frozen MiniLM encoder, 4 learnable query vectors, MultiheadAttention, bilinear scorer.
- **Training data:** `research/data/training_samples_large.json` (frozen at current SHA).
- **Train/val/test split:** by **repo**, not by question (so test-time generalization is across-repo). Train on httpx + Flask, validate on FastAPI, test on Requests. Re-shuffle for 3 seeds.
- **Optimizer:** AdamW, lr=1e-4, batch=16, epochs=10, early-stop on val composite.
- **Selection:** best epoch by val composite, **no test-set peeking**.
- **Reported number:** mean over 3 retrained checkpoints (one per outer seed).

---

## 6. Sensitivity Grid

Reported as ablation in §5 of the paper, not the headline.

| Axis | Values | Why |
|------|--------|-----|
| CCE threshold τ | `{0.3, 0.5, 0.7}` | covers the operating regime observed in `httpx_tuned_data.json` (CCE range 3.0–7.7 → normalize first) |
| Token partition | `{strict, lenient}` | tests robustness to the CODE/LANGUAGE/UNKNOWN classifier |
| Probe layer set | `{[31], [16,24,31], all}` | answers RQ4 |
| Max retrieval hops | `{1, 3, 5}` | tests multi-hop necessity |

The lead Pareto figure uses **τ = 0.5, lenient partition, layers [16,24,31], max hops = 3**.

---

## 7. Statistical Protocol

- **Seeds:** 3 (`{42, 1337, 2024}`) — every run repeated, results aggregated.
- **Confidence intervals:** Bootstrap 95% CI via `research/add_bootstrap_ci.py`, **n_bootstrap = 1000**, on every cell of every results table.
- **Significance tests:**
  - **Paired t-test** of CCE-adaptive vs `embedding` (strongest non-trivial baseline) on per-question correctness. Report p-values; α = 0.05.
  - **Wilcoxon signed-rank** as a non-parametric robustness check.
  - **Chi-squared** for spike-detection contingency (RQ1, per `research_questions.md:312`).
- **Multiple-comparison correction:** Bonferroni across the four RQs.
- **Reporting:** every number in the paper is `mean [CI_low, CI_high]`. No bare numbers.

---

## 8. Reproducibility

- **Pin everything:**
  - Code commit SHA → embedded in `paper_results.json`
  - Benchmark version → embedded
  - Model SHA / version (HuggingFace revision) → embedded
  - `requirements.lock` checked in
  - Tree-sitter grammar SHAs → embedded
  - `torch.__version__`, `cuda`, `cudnn` → embedded
- **Single-command reproduction:** `python research/run_final_experiments.py --protocol research/paper/PROTOCOL.md`
- **Released artifacts (with paper):**
  - `benchmark_v2_final.json`
  - `paper_results.json`
  - All notebooks frozen as PDF + checked-in `.ipynb`
  - Docker image with pinned dependencies

---

## 9. The Final Runner — `run_final_experiments.py` and the Colab Notebook

Two artifacts:

1. **`research/run_final_experiments.py`** — the canonical script. Runs end-to-end. Reproduces every paper number.
2. **`research/paper/Final_Paper_Experiments.ipynb`** — a thin Colab wrapper that clones the repo, sets up the environment, and calls the script. The notebook is for *running on Colab Pro*, not for *containing* the experiment logic.

The script is the source of truth. The notebook is the launcher.

### 9.1 Non-negotiable runner requirements (Q4 — no corner cutting)

The runner must satisfy ALL of the following. A reviewer-ready experiment cannot have any of these missing.

#### Resumability
- **Checkpoint after every (method, seed, config, question) tuple.** Atomic write: write to `paper_results.json.tmp`, fsync, rename. Never lose a completed result to a Colab disconnect.
- **Resume on restart:** runner reads existing `paper_results.json` and skips any tuple already present. Idempotent.
- **Per-tuple key:** `(method, seed, tau, partition, layers, max_hops, qid)`.

#### Determinism & seeding
- Set ALL of: `torch.manual_seed`, `np.random.seed`, `random.seed`, `transformers.set_seed`, `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic = True`, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Every random choice (file shuffling, batch order, etc.) goes through a seeded `numpy.random.Generator` passed explicitly — no hidden global state.

#### GPU memory management
- Free the model between method-runs only when method-class changes (CCE-adaptive vs static-pre-retrieval) — keep loaded otherwise.
- After every 50 questions: `torch.cuda.empty_cache()`, `gc.collect()`. Log peak VRAM to per-tuple metadata.
- Detect OOM and fail loudly; do NOT silently fall back to CPU.

#### Logging
- Every per-question result contains: `correctness`, `retrievals`, `context_p`, `context_r`, `context_f1`, `tokens_used`, `tokens_generated`, `tokens_retrieved`, `hallucination`, `latency_ms`, `peak_vram_mb`, `cce_trace` (token-level entropy values for `cce_*` methods only), `retrieved_files` (list), `generated_text`, `judge_score`, `judge_explanation`.
- Stdout log mirrored to `research/paper/run.log` with `tee`. Log lines are JSON, one per event.
- Heartbeat every 60 seconds: "method=X seed=Y q=Z/100 vram=N% elapsed=T".

#### LLM-as-judge integrity
- Judge calls are **cached by content hash** of (question, generated_answer, ground_truth). Cache survives restarts. Each question is judged once, ever, across all methods. Re-running a method does NOT re-call the judge unless the generated answer changed.
- Judge prompt is checked into `research/paper/judge_prompt.md` and hashed into results.
- Judge model SHA / snapshot date pinned.

#### Dry-run mode
- `--dry-run` flag: runs the entire grid against 2 questions per repo, no judge calls (uses heuristic correctness), to verify the pipeline before committing 30 hours of compute. **Mandatory pre-flight before any real run.**

#### Cost estimation
- `--estimate` flag: prints estimated wall-clock and judge-API cost for the full grid. Run this first; do not start the real run if estimates exceed budget.

#### Failure handling
- A failed tuple (OOM, timeout, generation error) writes a result row with `status="failed"` and the exception traceback. The run does NOT abort. Failed tuples are retried in a final pass at the end.
- Hard timeout per question: 120 seconds. Default: 60 seconds.

### 9.2 Inputs

- `--protocol research/paper/PROTOCOL.md`
- `--benchmark research/benchmarks/benchmark_v2_final.json`
- `--out research/paper/paper_results.json`
- `--seeds 42,1337,2024`
- `--methods all` (or comma-list to subset)
- `--config-grid headline` (only headline cell) | `sensitivity` (full grid) | `all`
- `--dry-run`, `--estimate`, `--resume` (default true), `--max-questions N` (debug)

### 9.3 Output schema (`paper_results.json`)

```json
{
  "protocol_sha": "...",                     // git SHA of PROTOCOL.md
  "benchmark_sha": "...",                    // git SHA of benchmark_v2_final.json
  "code_sha": "...",                         // git SHA of run_final_experiments.py
  "started_at": "ISO8601",
  "completed_at": "ISO8601 or null",
  "env": {
    "torch": "...", "transformers": "...", "cuda": "...", "cudnn": "...",
    "gpu_model": "A100-40GB", "python": "3.11.x"
  },
  "model": {
    "name": "codellama/CodeLlama-7b-Instruct-hf",
    "revision": "<HF SHA>",
    "quantization": "nf4-bnb-bf16",
    "judge_model": "gpt-4o-mini-2024-07-18",
    "judge_prompt_sha": "..."
  },
  "results": [
    {
      "method": "cce_adaptive",
      "seed": 42,
      "tau": 0.5,
      "partition": "lenient",
      "layers": [16,24,31],
      "max_hops": 3,
      "qid": "flask_001",
      "status": "ok",
      "correctness": 0.83,
      "retrievals": 2,
      "context_p": 0.6, "context_r": 0.8, "context_f1": 0.69,
      "tokens_used": 4321, "tokens_generated": 142, "tokens_retrieved": 4179,
      "hallucination": 0.0,
      "latency_ms": 1234,
      "peak_vram_mb": 6800,
      "cce_trace": [{"pos":12,"cce":0.61,"triggered":true}, "..."],
      "retrieved_files": ["flask/app.py"],
      "generated_text": "...",
      "judge_score": 0.85, "judge_explanation": "..."
    }
  ]
}
```

Aggregation (means, CIs, p-values) is computed **post-hoc** by `make_figures.py` from this raw `results` list — never written into the JSON, so re-aggregation is always reproducible.

### 9.4 Downstream artifacts (separate scripts, separate concerns)

- `research/paper/make_figures.py` — reads `paper_results.json`, produces Pareto figure + every paper table as PDF/CSV.
- `research/paper/run_significance_tests.py` — bootstrap CI + paired t-test + Wilcoxon + chi-squared, output as `significance.json`.
- `research/paper/Final_Paper_Experiments.ipynb` — Colab launcher: clones repo, installs deps, mounts Drive for checkpoint persistence, runs `--estimate`, then `--dry-run`, then real run with `--resume`. The notebook contains zero experiment logic.

### 9.5 Pre-flight checklist (before clicking "run all" on Colab)

- [ ] Protocol locked (Q1–Q5 resolved, committed)
- [ ] Benchmark v2 frozen and committed
- [ ] Runner passes `pytest research/integration_tests/`
- [ ] `--estimate` output reviewed (wall-clock + judge cost)
- [ ] `--dry-run` passes on 2 questions per repo, all 9 methods
- [ ] Drive mounted for checkpoint persistence (Colab disconnects mid-run otherwise lose state)
- [ ] Judge API key present in env, rate-limit headroom checked
- [ ] HuggingFace token loaded for CodeLlama gated access

---

## 10. Out of Scope for This Paper

Explicitly cut, to keep the contribution focused:

- The graph-knapsack context optimizer (`context_optimizer.py`) — unused in CCE evaluation; mention in related work, not as a contribution.
- The TOON format — out of scope; one-sentence mention if anywhere.
- The Vibe Station UI / web app — system paper material, not this paper.
- Multi-language support beyond Python — corpus is Python-only.
- Security scanner — irrelevant.
- Token estimator API — irrelevant.

---

## 11. Open Items (Q1–Q5 from review)

| ID  | Question | Status | Resolution |
|-----|----------|--------|-----------|
| Q1  | LLM choice for CCE/generation | **RESOLVED** | `codellama/CodeLlama-7b-Instruct-hf`, 4-bit (NF4 + bf16), single model for both generation and CCE |
| Q2  | Target venue + deadline | **DEFERRED** | Decoupled from protocol. Pick post-results. Same experiments serve any plausible venue |
| Q3  | Compute environment | **RESOLVED** | Colab Pro (A100/V100); checkpoint to Drive; ~22–35h projected |
| Q4  | Heuristic vs learned scorer | **RESOLVED** | Include `cce_learned` as 9th baseline, retrain cleanly per §5.1, no corner-cutting |
| Q5  | System name | **RESOLVED** | CCE |

**All five questions resolved.** Q2 is deferred by design — venue is downstream of results. Protocol is ready to lock at v1.0 once you've read it end-to-end.

---

## Revision history

- v0.1 (2026-04-25) — initial draft, awaiting Q1–Q5.
- v0.2 (2026-04-25) — Q1 (CodeLlama-7B), Q3 (Colab Pro), Q4 (include cce_learned, clean retrain), Q5 (CCE) resolved. Q2 set to tentative target EMNLP Findings 2026. §3.1 specified model + judge. §3.4 specified Colab Pro + wall-clock estimate. §5 expanded to 9 baselines with §5.1 learned-scorer training protocol. §9 expanded with non-negotiable resumability/determinism/logging/judge-cache/dry-run/cost-estimate requirements and pre-flight checklist.
- v0.3 (2026-04-25) — Q2 deferred (venue decoupled from protocol; pick post-results). Removed deadline placeholder. All five questions now resolved.
