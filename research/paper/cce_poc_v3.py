"""V3 PoC: multi-model, semantic-entropy ablation.

What v3 adds over v2:
- ``--model`` argument: switch between CodeLlama / Qwen2.5-Coder /
  DeepSeek-Coder. Output paths are namespaced per model so runs don't
  clobber each other.
- Chat-template-aware prompting via ``tokenizer.apply_chat_template`` -
  each model gets prompted in its own native format.
- Phase 1.5: sample N generations per task at temperature > 0, cluster by
  sentence-embedding cosine similarity, compute semantic entropy
  (Kuhn 2023 / Farquhar 2024). Adds 4 SE features to every task.
- Three new ablation arms expose semantic entropy:
    - ``semantic_entropy_only``  (the strong baseline we compare against)
    - ``flare_plus_se``          (cheap features + semantic entropy)
    - ``all_features``           (everything we have)

Usage (Colab Pro, ~2h per model on A100):

    !python research/paper/cce_poc_v3.py \\
        --model "Qwen/Qwen2.5-Coder-7B-Instruct" \\
        --repos-dir /content \\
        --out-dir   /content/drive/MyDrive/cce_poc_v3
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.paper.runner.cce_features import (
    ABLATION_ARMS, ALL_FEATURES_PLUS_SE, V6_GENERIC_FEATURES,
    CCEFeatureExtractor, build_partition_indices,
)
from research.paper.runner.semantic_entropy import semantic_entropy
from research.paper.v2_tasks import TASKS, REPOS


# Model-specific tweaks. Most behavior is auto-detected from the tokenizer's
# chat_template; this just documents what we expect / configure.
SUPPORTED_MODELS = {
    "codellama/CodeLlama-7b-Instruct-hf":   {"hidden_size": 4096, "n_layers": 32, "probe_layers": [16, 24, 31]},
    "Qwen/Qwen2.5-Coder-7B-Instruct":       {"hidden_size": 3584, "n_layers": 28, "probe_layers": [14, 21, 27]},
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5": {"hidden_size": 4096, "n_layers": 30, "probe_layers": [15, 22, 29]},
}


def slugify(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="HuggingFace model id (must be in SUPPORTED_MODELS or override)")
    p.add_argument("--repos-dir", default="/content",
                   help="parent dir under which repos are cloned")
    p.add_argument("--out-dir", default="research/paper/v3_runs",
                   help="output dir; results+caches are namespaced per model under this")
    p.add_argument("--max-new-tokens", type=int, default=250)
    p.add_argument("--max-new-tokens-sample", type=int, default=120,
                   help="shorter for the SE sampling phase to control compute")
    p.add_argument("--top-k-chunks", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples-se", type=int, default=5,
                   help="how many sampled generations per task for semantic entropy")
    p.add_argument("--se-temperature", type=float, default=0.7)
    p.add_argument("--se-sim-threshold", type=float, default=0.85)
    p.add_argument("--skip-generation", action="store_true",
                   help="skip Phase 1 if cached")
    p.add_argument("--skip-se-collection", action="store_true",
                   help="skip Phase 1.5 if cached")
    p.add_argument("--skip-phase3", action="store_true",
                   help="skip Phase 3 entirely")
    p.add_argument("--repos", default="all", help="comma-list or 'all'")
    return p.parse_args()


# ----------------------------------------------------------------- repo & retrieval


def clone_missing_repos(repos_dir: str, which: list[str]) -> None:
    repos_dir = Path(repos_dir)
    repos_dir.mkdir(parents=True, exist_ok=True)
    for r in which:
        meta = REPOS[r]
        target = repos_dir / r
        if target.exists() and any(target.iterdir()):
            continue
        depth = f"--depth {meta['depth']}" if meta.get("depth") else ""
        cmd = f"git clone {depth} {meta['clone_url']} {target}"
        print(f"[setup] cloning {r}: {cmd}", flush=True)
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError(f"git clone failed for {r}: rc={ret}")


def load_source_files_multi(repos_dir: str, which: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for r in which:
        meta = REPOS[r]
        repo_root = Path(repos_dir) / r
        if not repo_root.exists():
            print(f"[warn] {repo_root} missing, skipping {r}", flush=True)
            continue
        n_before = len(out)
        for root, _, filenames in os.walk(repo_root):
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                fpath = Path(root) / fname
                rel = str(fpath.relative_to(repo_root))
                if not rel.startswith(meta["source_root"]):
                    continue
                try:
                    out[rel] = fpath.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
        print(f"[setup] {r}: {len(out) - n_before} files", flush=True)
    return out


def setup_model_and_embedder(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from sentence_transformers import SentenceTransformer

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"[setup] hidden_size={model.config.hidden_size} "
          f"n_layers={getattr(model.config, 'num_hidden_layers', 'unknown')}",
          flush=True)
    return model, tokenizer, embedder


def probe_layers_for(model_name: str, n_layers: int | None = None) -> list[int]:
    if model_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name]["probe_layers"]
    if n_layers is None:
        return [-1]
    return [n_layers // 2, int(n_layers * 0.75), n_layers - 1]


def build_chunks_and_index(source_files, embedder, chunk_size: int = 80):
    chunks = []
    for filepath, content in source_files.items():
        lines = content.split("\n")
        for i in range(0, len(lines), chunk_size // 2):
            chunk_lines = lines[i:i + chunk_size]
            if len(chunk_lines) < 10:
                continue
            chunks.append({"filepath": filepath, "start_line": i,
                            "content": "\n".join(chunk_lines)})
    texts = [f"{c['filepath']}:\n{c['content']}" for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False,
                                  convert_to_tensor=True, batch_size=64)
    print(f"[setup] {len(chunks)} chunks indexed", flush=True)
    return chunks, embeddings


def retrieve(query, chunks, embeddings, embedder, top_k=3, target_file=None):
    import torch, torch.nn.functional as F
    q = embedder.encode(query, convert_to_tensor=True)
    sims = F.cosine_similarity(q.unsqueeze(0), embeddings)
    if target_file:
        for i, c in enumerate(chunks):
            if target_file in c["filepath"]:
                sims[i] += 0.3
    top = torch.topk(sims, k=min(top_k, len(chunks))).indices
    parts = []
    for idx in top:
        c = chunks[idx.item()]
        parts.append(f"# From {c['filepath']} (line {c['start_line']}):\n{c['content']}")
    return "\n\n".join(parts)


# ----------------------------------------------------------------- prompting


def build_chat_prompt(tokenizer, repo: str, question: str, context: str | None) -> str:
    """Format a prompt for the model using its native chat template."""
    if context:
        user_msg = (f"Use this source code to answer the question:\n\n{context}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer based ONLY on the source code above. Be specific and precise.")
    else:
        user_msg = (f"Answer this question about the {repo} Python library:\n\n{question}\n\n"
                    f"Be specific and precise.")

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
    return f"[INST] {user_msg} [/INST]"


# ----------------------------------------------------------------- generation


def generate_with_features(model, tokenizer, prompt: str, cce_extractor,
                            max_new_tokens: int = 250) -> tuple[str, dict]:
    import numpy as np, torch, torch.nn.functional as F
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    current_mask = inputs.attention_mask.clone()
    generated_ids = input_ids.clone()

    all_hs, all_attn, all_entropy = [], [], []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids=generated_ids, attention_mask=current_mask,
                        output_hidden_states=True, output_attentions=True)
        logits = out.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        all_entropy.append(entropy.item())

        last_hs = out.hidden_states[-1][:, -1, :]
        all_hs.append(last_hs.cpu().float().numpy())

        attn_probs = out.attentions[-1][:, :, -1, :]
        attn_entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-10), dim=-1).mean()
        all_attn.append(attn_entropy.item())

        cce_extractor.observe(logits[0])

        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        current_mask = torch.cat([current_mask, torch.ones((1, 1),
                                  device=model.device, dtype=current_mask.dtype)], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    response = tokenizer.decode(generated_ids[0][input_ids.shape[1]:],
                                  skip_special_tokens=True)
    hs = np.vstack(all_hs)
    return response.strip(), {
        "hs_mean_norm": float(np.mean(np.linalg.norm(hs, axis=1))),
        "hs_std_norm":  float(np.std(np.linalg.norm(hs, axis=1))),
        "hs_max_norm":  float(np.max(np.linalg.norm(hs, axis=1))),
        "cce_mean":     float(np.mean(all_entropy)),
        "cce_max":      float(np.max(all_entropy)),
        "cce_std":      float(np.std(all_entropy)),
        "cce_spikes":   int(sum(1 for e in all_entropy
                                 if e > np.mean(all_entropy) + 2 * np.std(all_entropy))),
        "attn_mean":    float(np.mean(all_attn)),
        "attn_max":     float(np.max(all_attn)),
        "attn_std":     float(np.std(all_attn)),
        "response_length": len(all_entropy),
        **cce_extractor.aggregate(),
    }


def generate_simple(model, tokenizer, prompt: str, max_new_tokens: int = 250,
                    do_sample: bool = False, temperature: float = 1.0) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        kwargs = dict(max_new_tokens=max_new_tokens,
                       pad_token_id=tokenizer.eos_token_id)
        if do_sample:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kwargs.update(do_sample=False)
        out = model.generate(**inputs, **kwargs)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                              skip_special_tokens=True).strip()


def sample_generations(model, tokenizer, prompt: str, n: int,
                        max_new_tokens: int, temperature: float) -> list[str]:
    out = []
    for _ in range(n):
        out.append(generate_simple(model, tokenizer, prompt,
                                     max_new_tokens=max_new_tokens,
                                     do_sample=True, temperature=temperature))
    return out


# ----------------------------------------------------------------- LOO ablation


def run_loo_ablation(features_per_task, outcomes, ablation_arms, seed=42):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    y = np.array([0 if c else 1 for c in outcomes])
    n = len(y)
    out: dict[str, dict] = {}

    for arm_name, feature_names in ablation_arms.items():
        try:
            X = np.array([[f.get(name, 0.0) for name in feature_names]
                            for f in features_per_task])
        except Exception as e:
            print(f"[warn] arm {arm_name} skipped: {e}", flush=True)
            continue
        preds = np.zeros(n, dtype=int)
        probs = np.zeros(n, dtype=float)
        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            X_tr, y_tr = X[mask], y[mask]
            if len(np.unique(y_tr)) < 2:
                continue
            clf = LogisticRegression(random_state=seed, max_iter=2000,
                                       class_weight="balanced")
            clf.fit(X_tr, y_tr)
            preds[i] = int(clf.predict(X[i:i + 1])[0])
            probs[i] = float(clf.predict_proba(X[i:i + 1])[0][1])

        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        acc = (tp + tn) / n if n else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        out[arm_name] = dict(
            n_features=len(feature_names),
            features=list(feature_names),
            accuracy=acc, precision=prec, recall=rec, f1=f1,
            tp=tp, fp=fp, tn=tn, fn=fn,
            predictions=preds.tolist(), probs=probs.tolist(),
        )
    return out


def per_repo_breakdown(arm_results, tasks, initial_correct):
    out = {}
    for arm_name, arm in arm_results.items():
        preds = arm["predictions"]
        by_repo: dict[str, dict] = {}
        for i, t in enumerate(tasks):
            r = t["repo"]
            d = by_repo.setdefault(r, {"n": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0})
            actual_err = 0 if initial_correct[i] else 1
            d["n"] += 1
            if preds[i] == 1 and actual_err == 1: d["tp"] += 1
            elif preds[i] == 1 and actual_err == 0: d["fp"] += 1
            elif preds[i] == 0 and actual_err == 0: d["tn"] += 1
            else: d["fn"] += 1
        for r, d in by_repo.items():
            d["accuracy"] = (d["tp"] + d["tn"]) / d["n"] if d["n"] else 0.0
            denom_p = d["tp"] + d["fp"]; denom_r = d["tp"] + d["fn"]
            p = d["tp"] / denom_p if denom_p else 0.0
            rec = d["tp"] / denom_r if denom_r else 0.0
            d["precision"] = p; d["recall"] = rec
            d["f1"] = 2 * p * rec / (p + rec) if (p + rec) else 0.0
        out[arm_name] = by_repo
    return out


# ----------------------------------------------------------------- phase 3


def run_phase3(arm_results, tasks, initial_correct, model, tokenizer,
               chunks, embeddings, embedder, top_k, max_new_tokens,
               phase3_cache_path: Path) -> dict:
    import numpy as np

    cache: dict[str, dict] = {}
    if phase3_cache_path.exists():
        with open(phase3_cache_path) as f:
            cache = json.load(f)

    def safe_verify(t, ans):
        try:
            return bool(t["verify"](ans))
        except Exception as e:
            print(f"  [warn] verify error on task {t['id']}: {type(e).__name__}: {e}",
                  flush=True)
            return False

    def regen(t):
        key = f"task_{t['id']}_retr"
        if key in cache:
            return cache[key]["correct"]
        ctx = retrieve(t["question"], chunks, embeddings, embedder,
                        top_k=top_k, target_file=t["relevant_file"])
        ans = generate_simple(model, tokenizer,
                                build_chat_prompt(tokenizer, t["repo"], t["question"], ctx),
                                max_new_tokens=max_new_tokens)
        ok = safe_verify(t, ans)
        cache[key] = {"answer": ans, "correct": ok}
        with open(phase3_cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        return ok

    print("[phase3] always-retrieve baseline...", flush=True)
    always = [regen(t) for t in tasks]
    print(f"[phase3] always-retrieve accuracy: {sum(always)/len(tasks):.3f}", flush=True)

    out = {}
    n = len(tasks)
    for arm_name, arm in arm_results.items():
        preds = arm["predictions"]
        final, used = [], []
        for i, t in enumerate(tasks):
            if preds[i] == 1:
                final.append(regen(t)); used.append(True)
            else:
                final.append(initial_correct[i]); used.append(False)
        out[arm_name] = dict(
            initial_accuracy=float(np.mean(initial_correct)),
            final_accuracy=float(np.mean(final)),
            always_retrieve_accuracy=float(np.mean(always)),
            n_retrievals_used=int(sum(used)),
            n_retrievals_saved=int(sum(1 for u in used if not u)),
            retrieval_save_rate=float(1 - sum(used) / n),
        )
    return out


# ----------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    repos = list(REPOS) if args.repos == "all" else [r.strip() for r in args.repos.split(",")]
    tasks = [t for t in TASKS if t["repo"] in repos]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(args.model)
    features_cache = out_dir / f"features__{slug}.json"
    se_cache       = out_dir / f"se_samples__{slug}.json"
    phase3_cache   = out_dir / f"phase3__{slug}.json"
    results_path   = out_dir / f"results__{slug}.json"

    print(f"[main] model={args.model} slug={slug}", flush=True)
    print(f"[main] {len(tasks)} tasks across {len(repos)} repos: {repos}", flush=True)

    need_model = not (args.skip_generation and features_cache.exists()
                       and (args.skip_se_collection and se_cache.exists())
                       and args.skip_phase3)

    model = tokenizer = embedder = chunks = embeddings = None
    if need_model:
        clone_missing_repos(args.repos_dir, repos)
        source_files = load_source_files_multi(args.repos_dir, repos)
        print(f"[setup] total {len(source_files)} files", flush=True)
        print(f"[phase1] loading model {args.model}", flush=True)
        model, tokenizer, embedder = setup_model_and_embedder(args.model)
        chunks, embeddings = build_chunks_and_index(source_files, embedder)

    # Phase 1: greedy generation + per-token features
    if args.skip_generation and features_cache.exists():
        print(f"[phase1] loading cached features from {features_cache}", flush=True)
        with open(features_cache) as f:
            blob = json.load(f)
        features_per_task = blob["features"]
        initial_answers = blob["initial_answers"]
        initial_correct = blob["initial_correct"]
    else:
        code_ids, lang_ids = build_partition_indices(tokenizer, partition="lenient")
        print(f"[phase1] code_ids={len(code_ids)} lang_ids={len(lang_ids)}", flush=True)

        features_per_task, initial_answers, initial_correct = [], [], []
        print(f"[phase1] generating features for {len(tasks)} tasks...", flush=True)
        t0 = time.time()
        for t in tasks:
            extractor = CCEFeatureExtractor(code_ids=code_ids, lang_ids=lang_ids)
            ans, feats = generate_with_features(
                model, tokenizer,
                build_chat_prompt(tokenizer, t["repo"], t["question"], None),
                cce_extractor=extractor, max_new_tokens=args.max_new_tokens,
            )
            initial_answers.append(ans)
            features_per_task.append(feats)
            try:
                ok = bool(t["verify"](ans))
            except Exception as e:
                print(f"  [warn] verify error on task {t['id']}: {e}", flush=True)
                ok = False
            initial_correct.append(ok)
            print(f"  task {t['id']:3d} [{t['repo']:8s} {t['difficulty']:4s}] "
                  f"{'OK' if ok else 'WRONG'}  cce={feats.get('real_cce_mean', 0):+.3f}",
                  flush=True)
        print(f"[phase1] done in {time.time() - t0:.1f}s", flush=True)
        with open(features_cache, "w") as f:
            json.dump({"features": features_per_task,
                       "initial_answers": initial_answers,
                       "initial_correct": initial_correct,
                       "qids": [t["id"] for t in tasks]}, f, indent=2)

    # Phase 1.5: semantic entropy via sampled generations
    if args.skip_se_collection and se_cache.exists():
        print(f"[phase1.5] loading cached SE from {se_cache}", flush=True)
        with open(se_cache) as f:
            se_blob = json.load(f)
        se_per_task = se_blob["se_features"]
        se_samples_per_task = se_blob.get("samples", [])
    else:
        print(f"[phase1.5] sampling {args.n_samples_se} generations per task @ T={args.se_temperature}",
              flush=True)
        t0 = time.time()
        se_per_task: list[dict] = []
        se_samples_per_task: list[list[str]] = []
        for t in tasks:
            prompt = build_chat_prompt(tokenizer, t["repo"], t["question"], None)
            samples = sample_generations(
                model, tokenizer, prompt,
                n=args.n_samples_se,
                max_new_tokens=args.max_new_tokens_sample,
                temperature=args.se_temperature,
            )
            se = semantic_entropy(samples, embedder,
                                    sim_threshold=args.se_sim_threshold)
            se_per_task.append(se)
            se_samples_per_task.append(samples)
            print(f"  task {t['id']:3d} se={se['semantic_entropy']:.3f} "
                  f"clusters={se['n_clusters']} mode={se['largest_cluster_frac']:.2f}",
                  flush=True)
        print(f"[phase1.5] done in {time.time() - t0:.1f}s", flush=True)
        with open(se_cache, "w") as f:
            json.dump({"se_features": se_per_task,
                       "samples": se_samples_per_task,
                       "qids": [t["id"] for t in tasks]}, f, indent=2)

    # Merge SE features into per-task features dict
    assert len(se_per_task) == len(features_per_task)
    for f, se in zip(features_per_task, se_per_task):
        f.update({k: se[k] for k in ("semantic_entropy", "semantic_entropy_norm",
                                       "n_clusters", "largest_cluster_frac")})

    # Phase 2
    print(f"\n[phase2] LOO ablation across {len(ABLATION_ARMS)} arms (n={len(tasks)})",
          flush=True)
    arm_results = run_loo_ablation(features_per_task, initial_correct, ABLATION_ARMS,
                                     seed=args.seed)
    print()
    print(f"{'ARM':<22} {'#feat':>5} {'acc':>5} {'prec':>5} {'rec':>5} {'f1':>5}")
    for name, r in arm_results.items():
        print(f"{name:<22} {r['n_features']:>5d} {r['accuracy']:>5.3f} "
              f"{r['precision']:>5.3f} {r['recall']:>5.3f} {r['f1']:>5.3f}")

    repo_breakdown = per_repo_breakdown(arm_results, tasks, initial_correct)

    phase3 = {}
    if not args.skip_phase3 and model is not None:
        print(f"\n[phase3] end-to-end gating", flush=True)
        phase3 = run_phase3(arm_results, tasks, initial_correct, model, tokenizer,
                              chunks, embeddings, embedder,
                              top_k=args.top_k_chunks, max_new_tokens=args.max_new_tokens,
                              phase3_cache_path=phase3_cache)
        print()
        print(f"{'ARM':<22} {'init':>5} {'final':>5} {'always':>5} {'used':>5} {'saved':>5} {'save%':>6}")
        for name, r in phase3.items():
            print(f"{name:<22} {r['initial_accuracy']:>5.3f} {r['final_accuracy']:>5.3f} "
                  f"{r['always_retrieve_accuracy']:>5.3f} {r['n_retrievals_used']:>5d} "
                  f"{r['n_retrievals_saved']:>5d} {r['retrieval_save_rate']*100:>5.1f}%")

    out = {
        "version": "cce_poc_v3",
        "model": args.model,
        "model_slug": slug,
        "n_tasks": len(tasks),
        "repos": repos,
        "seed": args.seed,
        "se_config": {
            "n_samples": args.n_samples_se,
            "temperature": args.se_temperature,
            "sim_threshold": args.se_sim_threshold,
        },
        "phase2_loo_ablation": arm_results,
        "phase2_per_repo": repo_breakdown,
        "phase3_end_to_end": phase3,
        "feature_definitions": {
            "all_features_plus_se": list(ALL_FEATURES_PLUS_SE),
            "ablation_arms": {k: list(v) for k, v in ABLATION_ARMS.items()},
        },
    }
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[done] results at {results_path}", flush=True)

    try:
        del model; gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
