"""V6 + CCE-as-feature: 6-arm ablation proof-of-concept.

WHAT THIS DOES
==============
1. Loads CodeLlama-7B-Instruct-hf (4-bit, eager attention).
2. Loads V6's 20 balanced tasks (10 easy, 10 hard) on httpx.
3. Phase 1: for each task, generates an answer with no retrieval and collects
   per-token features — V6's original 11 features PLUS 8 new contrastive-code-
   entropy features (H_code, H_lang, CCE = H_code − H_lang aggregates).
4. Phase 2: trains six LogisticRegression error-detection classifiers, one per
   ablation arm (different feature subsets), each via leave-one-out cross-
   validation.
5. Phase 3: for each (arm, held-out task), if the LOO classifier predicts the
   answer is wrong, retrieves relevant chunks and regenerates with context.
6. Reports the ablation table: classifier F1 + end-to-end accuracy +
   retrievals saved per arm.

WHY THIS MATTERS
================
The V6 paper-novelty question reduces to: do contrastive-code-entropy
features (H_code − H_lang) carry signal that V6's existing hidden-state +
plain-Shannon-entropy features don't already capture? If the `v6_plus_cce`
arm beats `v6_full` by a meaningful margin (and `cce_only` is non-trivial),
the CCE work is paying off — the paper has a defensible novelty story
against SAPLMA / FLARE / DRAGIN. If the arms are within CIs, V6 alone is
not differentiated enough for a top venue.

USAGE (Colab Pro)
=================
    !pip install -q torch transformers accelerate bitsandbytes \\
                    sentence-transformers scikit-learn
    !git clone https://github.com/aniJani/reposynth.git
    %cd reposynth
    !git checkout Research
    !git clone --depth 1 https://github.com/encode/httpx.git /content/httpx
    !python research/paper/cce_poc.py --httpx-dir /content/httpx \\
                                       --out research/paper/cce_poc_results.json
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
from research.paper.v6_tasks import TASKS


# ----------------------------------------------------------------- args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--httpx-dir", default="/content/httpx",
                   help="path to a cloned httpx repo")
    p.add_argument("--model", default="codellama/CodeLlama-7b-Instruct-hf")
    p.add_argument("--out", default="research/paper/cce_poc_results.json")
    p.add_argument("--max-new-tokens", type=int, default=250)
    p.add_argument("--top-k-chunks", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--features-cache", default="research/paper/cce_poc_features.json",
                   help="cache features so reruns of just the ablation are instant")
    p.add_argument("--skip-generation", action="store_true",
                   help="skip Phase 1 if features-cache exists; only redo ablation")
    return p.parse_args()


# ----------------------------------------------------------------- setup


def load_source_files(repo_dir: str) -> dict[str, str]:
    files: dict[str, str] = {}
    for root, _, filenames in os.walk(os.path.join(repo_dir, 'httpx')):
        for fname in filenames:
            if fname.endswith('.py'):
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, repo_dir)
                with open(fpath, encoding='utf-8') as f:
                    files[rel] = f.read()
    return files


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
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return model, tokenizer, embedder


def build_chunks_and_index(source_files: dict[str, str], embedder, chunk_size: int = 80):
    import torch  # noqa: F401  (used downstream)

    chunks = []
    for filepath, content in source_files.items():
        lines = content.split('\n')
        for i in range(0, len(lines), chunk_size // 2):
            chunk_lines = lines[i:i + chunk_size]
            if len(chunk_lines) < 10:
                continue
            chunks.append({
                'filepath': filepath,
                'start_line': i,
                'content': '\n'.join(chunk_lines),
            })
    texts = [f"{c['filepath']}:\n{c['content']}" for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_tensor=True)
    return chunks, embeddings


def retrieve(query: str, chunks, embeddings, embedder, top_k: int = 3,
             target_file: str | None = None) -> str:
    import torch
    import torch.nn.functional as F
    q = embedder.encode(query, convert_to_tensor=True)
    sims = F.cosine_similarity(q.unsqueeze(0), embeddings)
    if target_file:
        for i, c in enumerate(chunks):
            if target_file in c['filepath']:
                sims[i] += 0.3
    top = torch.topk(sims, k=min(top_k, len(chunks))).indices
    parts = []
    for idx in top:
        c = chunks[idx.item()]
        parts.append(f"# From {c['filepath']} (line {c['start_line']}):\n{c['content']}")
    return "\n\n".join(parts)


# ----------------------------------------------------------------- generation + features


def build_prompt(question: str, context: str | None) -> str:
    if context:
        return (f"[INST] Use this source code to answer the question:\n\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer based ONLY on the source code above. Be specific and precise. [/INST]")
    return (f"[INST] Answer this question about the httpx Python library:\n\n{question}\n\n"
            f"Be specific and precise. [/INST]")


def generate_with_features(model, tokenizer, prompt: str, cce_extractor,
                            max_new_tokens: int = 250) -> tuple[str, dict[str, float]]:
    """V6's generate_with_features, augmented with CCE feature extraction."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_ids = inputs.input_ids
    current_mask = inputs.attention_mask.clone()
    generated_ids = input_ids.clone()

    all_hs: list[Any] = []
    all_attn: list[float] = []
    all_entropy: list[float] = []
    # CCE extractor accumulates internally; caller creates a fresh one per task.

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
        'hs_mean_norm': float(np.mean(np.linalg.norm(hs_array, axis=1))),
        'hs_std_norm':  float(np.std(np.linalg.norm(hs_array, axis=1))),
        'hs_max_norm':  float(np.max(np.linalg.norm(hs_array, axis=1))),
        'cce_mean':     float(np.mean(all_entropy)),  # plain Shannon, kept for V6 compat
        'cce_max':      float(np.max(all_entropy)),
        'cce_std':      float(np.std(all_entropy)),
        'cce_spikes':   int(sum(1 for e in all_entropy
                                 if e > np.mean(all_entropy) + 2 * np.std(all_entropy))),
        'attn_mean':    float(np.mean(all_attn)),
        'attn_max':     float(np.max(all_attn)),
        'attn_std':     float(np.std(all_attn)),
        'response_length': len(all_entropy),
        **cce_extractor.aggregate(),
    }
    return response.strip(), features


def generate_simple(model, tokenizer, prompt: str, max_new_tokens: int = 250) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


# ----------------------------------------------------------------- ablation


def run_loo_ablation(features_per_task: list[dict], task_outcomes: list[bool],
                     ablation_arms: dict, seed: int = 42) -> dict[str, dict]:
    """For each arm (feature subset), do leave-one-out CV with LogisticRegression.

    Returns per-arm: {accuracy, precision, recall, f1, predictions[], probs[]}
    where predictions[i] is the LOO prediction for held-out task i.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    y = np.array([0 if c else 1 for c in task_outcomes])  # 1 = error
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
                preds[i] = 0; probs[i] = 0.0
                continue
            clf = LogisticRegression(random_state=seed, max_iter=1000,
                                     class_weight='balanced')
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
            predictions=preds.tolist(),
            probs=probs.tolist(),
        )
    return out


def run_phase3_per_arm(arm_results: dict, tasks: list[dict], features_per_task,
                       initial_answers: list[str], model, tokenizer, chunks,
                       embeddings, embedder, top_k: int, max_new_tokens: int) -> dict:
    """For each arm: for each task, if predicted_error → retrieve & regenerate.
    Compute end-to-end accuracy + retrievals_used + savings vs always-retrieve."""
    import numpy as np

    n = len(tasks)
    initial_correct = [tasks[i]['verify'](initial_answers[i]) for i in range(n)]

    # always-retrieve baseline (one pass, shared by all arms)
    always_correct = []
    print("[phase3] computing always-retrieve baseline once...", flush=True)
    for t in tasks:
        ctx = retrieve(t['question'], chunks, embeddings, embedder,
                       top_k=top_k, target_file=t['relevant_file'])
        ans = generate_simple(model, tokenizer, build_prompt(t['question'], ctx),
                              max_new_tokens=max_new_tokens)
        always_correct.append(t['verify'](ans))

    out = {}
    for arm_name, arm in arm_results.items():
        preds = arm['predictions']
        final_correct, used_retrieval = [], []
        for i, t in enumerate(tasks):
            if preds[i] == 1:  # predicted wrong → retrieve
                ctx = retrieve(t['question'], chunks, embeddings, embedder,
                               top_k=top_k, target_file=t['relevant_file'])
                ans = generate_simple(model, tokenizer, build_prompt(t['question'], ctx),
                                      max_new_tokens=max_new_tokens)
                final_correct.append(t['verify'](ans))
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


# ----------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.features_cache); cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Phase 1: features (cached)
    if args.skip_generation and cache_path.exists():
        print(f"[phase1] loading cached features from {cache_path}", flush=True)
        with open(cache_path) as f:
            cache = json.load(f)
        features_per_task = cache['features']
        initial_answers = cache['initial_answers']
        initial_correct = cache['initial_correct']
        # tasks reload (verify funcs come from v6_tasks)
        tasks = TASKS
    else:
        print(f"[phase1] loading httpx from {args.httpx_dir}", flush=True)
        source_files = load_source_files(args.httpx_dir)
        print(f"[phase1] {len(source_files)} httpx files loaded", flush=True)

        print(f"[phase1] loading model {args.model}", flush=True)
        model, tokenizer, embedder = setup_model_and_embedder(args.model)

        print("[phase1] building partition indices for CCE", flush=True)
        code_ids, lang_ids = build_partition_indices(tokenizer, partition="lenient")
        print(f"[phase1] code_ids={len(code_ids)} lang_ids={len(lang_ids)}", flush=True)

        print("[phase1] building chunks + retrieval index", flush=True)
        chunks, embeddings = build_chunks_and_index(source_files, embedder)

        tasks = TASKS
        features_per_task: list[dict] = []
        initial_answers: list[str] = []
        initial_correct: list[bool] = []
        print(f"[phase1] generating features for {len(tasks)} tasks...", flush=True)
        t0 = time.time()
        for t in tasks:
            extractor = CCEFeatureExtractor(code_ids=code_ids, lang_ids=lang_ids)
            ans, feats = generate_with_features(
                model, tokenizer, build_prompt(t['question'], None),
                cce_extractor=extractor, max_new_tokens=args.max_new_tokens,
            )
            initial_answers.append(ans)
            features_per_task.append(feats)
            ok = t['verify'](ans)
            initial_correct.append(ok)
            print(f"  task {t['id']:2d} [{t['difficulty']:4s}] {'OK' if ok else 'WRONG'}  "
                  f"cce={feats.get('real_cce_mean', 0):.3f}  hcode={feats.get('h_code_mean', 0):.3f}  "
                  f"hlang={feats.get('h_lang_mean', 0):.3f}",
                  flush=True)
        print(f"[phase1] done in {time.time() - t0:.1f}s", flush=True)

        with open(cache_path, "w") as f:
            json.dump({
                "features": features_per_task,
                "initial_answers": initial_answers,
                "initial_correct": initial_correct,
            }, f, indent=2)

    # Phase 2: LOO ablation
    print(f"\n[phase2] LOO ablation across {len(ABLATION_ARMS)} arms", flush=True)
    arm_results = run_loo_ablation(features_per_task, initial_correct, ABLATION_ARMS, seed=args.seed)
    print()
    print(f"{'ARM':<20} {'#feat':>5} {'acc':>5} {'prec':>5} {'rec':>5} {'f1':>5}")
    for name, r in arm_results.items():
        print(f"{name:<20} {r['n_features']:>5d} {r['accuracy']:>5.3f} "
              f"{r['precision']:>5.3f} {r['recall']:>5.3f} {r['f1']:>5.3f}")

    # Phase 3 (only if model loaded — i.e. not in --skip-generation mode)
    phase3 = {}
    if not args.skip_generation:
        print(f"\n[phase3] end-to-end with retrieval gating", flush=True)
        phase3 = run_phase3_per_arm(
            arm_results, tasks, features_per_task, initial_answers,
            model, tokenizer, chunks, embeddings, embedder,
            top_k=args.top_k_chunks, max_new_tokens=args.max_new_tokens,
        )
        print()
        print(f"{'ARM':<20} {'init':>5} {'final':>5} {'always':>5} {'used':>5} {'saved':>5} {'save%':>6}")
        for name, r in phase3.items():
            print(f"{name:<20} {r['initial_accuracy']:>5.3f} {r['final_accuracy']:>5.3f} "
                  f"{r['always_retrieve_accuracy']:>5.3f} {r['n_retrievals_used']:>5d} "
                  f"{r['n_retrievals_saved']:>5d} {r['retrieval_save_rate']*100:>5.1f}%")

    # Save
    out = {
        "version": "cce_poc_v1",
        "model": args.model,
        "n_tasks": len(tasks),
        "seed": args.seed,
        "phase2_loo_ablation": arm_results,
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

    # Free GPU
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
