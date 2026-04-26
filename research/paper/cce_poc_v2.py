"""V2 PoC: 80-task, 4-repo CCE ablation.

Differences from v1 (research/paper/cce_poc.py):
- Multi-repo: loads source files from httpx, flask, fastapi, requests into a
  single combined retrieval index; tasks specify their `repo` and the
  retrieval target_file boost still works.
- Phase-3 cache: for each task, retrieved-context regeneration is computed
  AT MOST ONCE across all arms, even if multiple arms predict error.
  (Greedy decoding makes regeneration deterministic; this saves ~6x time.)
- Per-repo breakdown in the output JSON.
- Reads tasks from research.paper.v2_tasks.

Usage (Colab Pro):
    !python research/paper/cce_poc_v2.py \\
        --repos-dir /content \\
        --out research/paper/cce_poc_v2_results.json \\
        --features-cache research/paper/cce_poc_v2_features.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.paper.runner.cce_features import (
    ABLATION_ARMS, ALL_FEATURES, V6_GENERIC_FEATURES,
    CCEFeatureExtractor, build_partition_indices,
)
from research.paper.v2_tasks import TASKS, REPOS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repos-dir", default="/content",
                   help="parent dir under which each repo is cloned: "
                        "<repos-dir>/{httpx,flask,fastapi,requests}")
    p.add_argument("--model", default="codellama/CodeLlama-7b-Instruct-hf")
    p.add_argument("--out", default="research/paper/cce_poc_v2_results.json")
    p.add_argument("--features-cache", default="research/paper/cce_poc_v2_features.json")
    p.add_argument("--phase3-cache", default="research/paper/cce_poc_v2_phase3.json")
    p.add_argument("--max-new-tokens", type=int, default=250)
    p.add_argument("--top-k-chunks", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-generation", action="store_true",
                   help="skip Phase 1 if features-cache exists; only redo ablation")
    p.add_argument("--repos", default="all",
                   help="comma-list of repos to include, or 'all'")
    return p.parse_args()


# ----------------------------------------------------------------- repo loading


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
    """Load .py files from each repo. Returns dict keyed by relative path
    matching the `relevant_file` field used in tasks (e.g. 'httpx/_api.py',
    'src/flask/app.py', 'fastapi/applications.py', 'src/requests/api.py').
    """
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
        print(f"[setup] {r}: {len(out) - n_before} files loaded "
              f"(filter: {meta['source_root']}/)", flush=True)
    return out


# ----------------------------------------------------------------- model & retrieval


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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="eager",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return model, tokenizer, embedder


def build_chunks_and_index(source_files: dict[str, str], embedder, chunk_size: int = 80):
    chunks = []
    for filepath, content in source_files.items():
        lines = content.split("\n")
        for i in range(0, len(lines), chunk_size // 2):
            chunk_lines = lines[i:i + chunk_size]
            if len(chunk_lines) < 10:
                continue
            chunks.append({
                "filepath": filepath,
                "start_line": i,
                "content": "\n".join(chunk_lines),
            })
    texts = [f"{c['filepath']}:\n{c['content']}" for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False,
                                  convert_to_tensor=True, batch_size=64)
    print(f"[setup] {len(chunks)} chunks indexed", flush=True)
    return chunks, embeddings


def retrieve(query: str, chunks, embeddings, embedder, top_k: int = 3,
             target_file: str | None = None) -> str:
    import torch
    import torch.nn.functional as F
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


# ----------------------------------------------------------------- generation


def build_prompt(repo: str, question: str, context: str | None) -> str:
    if context:
        return (f"[INST] Use this source code to answer the question:\n\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer based ONLY on the source code above. Be specific and precise. [/INST]")
    return (f"[INST] Answer this question about the {repo} Python library:\n\n{question}\n\n"
            f"Be specific and precise. [/INST]")


def generate_with_features(model, tokenizer, prompt: str, cce_extractor,
                            max_new_tokens: int = 250) -> tuple[str, dict[str, float]]:
    import numpy as np
    import torch
    import torch.nn.functional as F

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    current_mask = inputs.attention_mask.clone()
    generated_ids = input_ids.clone()

    all_hs: list[Any] = []
    all_attn: list[float] = []
    all_entropy: list[float] = []

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
        current_mask = torch.cat([current_mask, torch.ones((1, 1), device=model.device,
                                                            dtype=current_mask.dtype)], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    response = tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    hs_array = np.vstack(all_hs)

    features = {
        "hs_mean_norm": float(np.mean(np.linalg.norm(hs_array, axis=1))),
        "hs_std_norm":  float(np.std(np.linalg.norm(hs_array, axis=1))),
        "hs_max_norm":  float(np.max(np.linalg.norm(hs_array, axis=1))),
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
    return response.strip(), features


def generate_simple(model, tokenizer, prompt: str, max_new_tokens: int = 250) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


# ----------------------------------------------------------------- ablation


def run_loo_ablation(features_per_task: list[dict], outcomes: list[bool],
                     ablation_arms: dict, seed: int = 42) -> dict[str, dict]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    y = np.array([0 if c else 1 for c in outcomes])
    n = len(y)
    out: dict[str, dict] = {}

    for arm_name, feature_names in ablation_arms.items():
        X = np.array([[f[name] for name in feature_names] for f in features_per_task])
        preds = np.zeros(n, dtype=int)
        probs = np.zeros(n, dtype=float)
        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            X_tr, y_tr = X[mask], y[mask]
            if len(np.unique(y_tr)) < 2:
                continue
            clf = LogisticRegression(random_state=seed, max_iter=1000,
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


# ----------------------------------------------------------------- phase 3 with cache


def run_phase3(arm_results, tasks, initial_correct, model, tokenizer,
               chunks, embeddings, embedder, top_k: int, max_new_tokens: int,
               phase3_cache_path: Path) -> dict:
    """End-to-end gating with retrieved-answer cache shared across arms."""
    import numpy as np

    cache: dict[str, dict] = {}
    if phase3_cache_path.exists():
        with open(phase3_cache_path) as f:
            cache = json.load(f)

    def safe_verify(t, ans: str) -> bool:
        try:
            return bool(t["verify"](ans))
        except Exception as e:
            print(f"  [warn] verify error on task {t['id']}: {type(e).__name__}: {e}",
                  flush=True)
            return False

    def regen_with_retrieval(t) -> bool:
        key = f"task_{t['id']}_retr"
        if key in cache:
            return cache[key]["correct"]
        ctx = retrieve(t["question"], chunks, embeddings, embedder,
                       top_k=top_k, target_file=t["relevant_file"])
        ans = generate_simple(model, tokenizer,
                              build_prompt(t["repo"], t["question"], ctx),
                              max_new_tokens=max_new_tokens)
        ok = safe_verify(t, ans)
        cache[key] = {"answer": ans, "correct": ok}
        with open(phase3_cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        return ok

    print("[phase3] computing always-retrieve baseline (cached, shared)...", flush=True)
    always_correct = []
    for t in tasks:
        always_correct.append(regen_with_retrieval(t))
    print(f"[phase3] always-retrieve accuracy: {sum(always_correct)/len(tasks):.3f}", flush=True)

    out = {}
    n = len(tasks)
    for arm_name, arm in arm_results.items():
        preds = arm["predictions"]
        final_correct, used_retrieval = [], []
        for i, t in enumerate(tasks):
            if preds[i] == 1:
                final_correct.append(regen_with_retrieval(t))
                used_retrieval.append(True)
            else:
                final_correct.append(initial_correct[i])
                used_retrieval.append(False)
        out[arm_name] = dict(
            initial_accuracy=float(np.mean(initial_correct)),
            final_accuracy=float(np.mean(final_correct)),
            always_retrieve_accuracy=float(np.mean(always_correct)),
            n_retrievals_used=int(sum(used_retrieval)),
            n_retrievals_saved=int(sum(1 for u in used_retrieval if not u)),
            retrieval_save_rate=float(1 - sum(used_retrieval) / n),
        )
    return out


def per_repo_breakdown(arm_results, tasks, initial_correct) -> dict:
    """Per-arm × per-repo classifier accuracy."""
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


# ----------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.features_cache); cache_path.parent.mkdir(parents=True, exist_ok=True)
    phase3_cache_path = Path(args.phase3_cache); phase3_cache_path.parent.mkdir(parents=True, exist_ok=True)

    repos = list(REPOS) if args.repos == "all" else [r.strip() for r in args.repos.split(",")]
    tasks = [t for t in TASKS if t["repo"] in repos]
    print(f"[main] {len(tasks)} tasks across {len(repos)} repos: {repos}", flush=True)

    # Phase 1: features (cached)
    if args.skip_generation and cache_path.exists():
        print(f"[phase1] loading cached features from {cache_path}", flush=True)
        with open(cache_path) as f:
            cache = json.load(f)
        features_per_task = cache["features"]
        initial_answers = cache["initial_answers"]
        initial_correct = cache["initial_correct"]
        # task list for phase3 needs to come from v2_tasks (lambdas not serialized)
        cached_qids = cache.get("qids") or [t["id"] for t in tasks]
        if cached_qids != [t["id"] for t in tasks]:
            print("[warn] cached qids != current tasks; consider re-running w/o --skip-generation",
                  flush=True)
    else:
        clone_missing_repos(args.repos_dir, repos)
        source_files = load_source_files_multi(args.repos_dir, repos)
        print(f"[setup] total {len(source_files)} files across {len(repos)} repos", flush=True)

        print(f"[phase1] loading model {args.model}", flush=True)
        model, tokenizer, embedder = setup_model_and_embedder(args.model)

        print("[phase1] building partition indices for CCE", flush=True)
        code_ids, lang_ids = build_partition_indices(tokenizer, partition="lenient")
        print(f"[phase1] code_ids={len(code_ids)} lang_ids={len(lang_ids)}", flush=True)

        chunks, embeddings = build_chunks_and_index(source_files, embedder)

        features_per_task: list[dict] = []
        initial_answers: list[str] = []
        initial_correct: list[bool] = []
        print(f"[phase1] generating features for {len(tasks)} tasks...", flush=True)
        t0 = time.time()
        for t in tasks:
            extractor = CCEFeatureExtractor(code_ids=code_ids, lang_ids=lang_ids)
            ans, feats = generate_with_features(
                model, tokenizer,
                build_prompt(t["repo"], t["question"], None),
                cce_extractor=extractor, max_new_tokens=args.max_new_tokens,
            )
            initial_answers.append(ans)
            features_per_task.append(feats)
            try:
                ok = bool(t["verify"](ans))
            except Exception as e:
                print(f"  [warn] verify error on task {t['id']}: {type(e).__name__}: {e}",
                      flush=True)
                ok = False
            initial_correct.append(ok)
            print(f"  task {t['id']:3d} [{t['repo']:8s} {t['difficulty']:4s}] "
                  f"{'OK' if ok else 'WRONG'}  cce={feats.get('real_cce_mean', 0):+.3f}",
                  flush=True)
        print(f"[phase1] done in {time.time() - t0:.1f}s", flush=True)

        with open(cache_path, "w") as f:
            json.dump({
                "features": features_per_task,
                "initial_answers": initial_answers,
                "initial_correct": initial_correct,
                "qids": [t["id"] for t in tasks],
            }, f, indent=2)

    # Phase 2
    print(f"\n[phase2] LOO ablation across {len(ABLATION_ARMS)} arms (n={len(tasks)})", flush=True)
    arm_results = run_loo_ablation(features_per_task, initial_correct, ABLATION_ARMS, seed=args.seed)
    print()
    print(f"{'ARM':<20} {'#feat':>5} {'acc':>5} {'prec':>5} {'rec':>5} {'f1':>5}")
    for name, r in arm_results.items():
        print(f"{name:<20} {r['n_features']:>5d} {r['accuracy']:>5.3f} "
              f"{r['precision']:>5.3f} {r['recall']:>5.3f} {r['f1']:>5.3f}")

    repo_breakdown = per_repo_breakdown(arm_results, tasks, initial_correct)

    # Phase 3 (only if model present)
    phase3 = {}
    if not args.skip_generation:
        print(f"\n[phase3] end-to-end gating with shared retrieval cache", flush=True)
        phase3 = run_phase3(arm_results, tasks, initial_correct, model, tokenizer,
                             chunks, embeddings, embedder,
                             top_k=args.top_k_chunks, max_new_tokens=args.max_new_tokens,
                             phase3_cache_path=phase3_cache_path)
        print()
        print(f"{'ARM':<20} {'init':>5} {'final':>5} {'always':>5} {'used':>5} {'saved':>5} {'save%':>6}")
        for name, r in phase3.items():
            print(f"{name:<20} {r['initial_accuracy']:>5.3f} {r['final_accuracy']:>5.3f} "
                  f"{r['always_retrieve_accuracy']:>5.3f} {r['n_retrievals_used']:>5d} "
                  f"{r['n_retrievals_saved']:>5d} {r['retrieval_save_rate']*100:>5.1f}%")

    out = {
        "version": "cce_poc_v2",
        "model": args.model,
        "n_tasks": len(tasks),
        "repos": repos,
        "seed": args.seed,
        "phase2_loo_ablation": arm_results,
        "phase2_per_repo": repo_breakdown,
        "phase3_end_to_end": phase3,
        "feature_definitions": {
            "v6_generic": list(V6_GENERIC_FEATURES),
            "all": list(ALL_FEATURES),
            "ablation_arms": {k: list(v) for k, v in ABLATION_ARMS.items()},
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[done] results at {out_path}", flush=True)

    try:
        del model
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
