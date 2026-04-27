"""Diagnostic: can each gating signal predict needs_retrieval per task?

For each model:
- needs_retrieval[i] = always_correct[i] AND NOT initial_correct[i]
  (i.e., retrieval would have changed the outcome from wrong to right)
- Score every gating signal (univariate + multivariate arms) as a binary
  classifier predicting needs_retrieval. Report AUC and best-F1-threshold
  precision/recall.

This is the question the paper actually wants to answer: when retrieval
would help, can the LLM-internal signal tell?

Reads research/paper/v3_results/{features_with_nli__,phase3__}{slug}.json.
Writes research/paper/v3_results/needs_retrieval_diagnostic.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut


V3 = Path("research/paper/v3_results")

MODELS = [
    ("Qwen/Qwen2.5-Coder-7B-Instruct",
     "Qwen_Qwen2_5_Coder_7B_Instruct"),
    ("deepseek-ai/deepseek-coder-7b-instruct-v1.5",
     "deepseek_ai_deepseek_coder_7b_instruct_v1_5"),
    ("codellama/CodeLlama-7b-Instruct-hf",
     "codellama_CodeLlama_7b_Instruct_hf"),
]

# Univariate signals: (feature key, paper-display name).
UNIVARIATE = [
    ("nli_se",            "NLI-SE"),
    ("nli_n_clusters",    "NLI clusters"),
    ("semantic_entropy",  "embedding-SE"),
    ("n_clusters",        "emb clusters"),
    ("cce_max",           "FLARE max-ent"),
    ("cce_mean",          "FLARE mean-ent"),
    ("cce_spikes",        "FLARE spikes"),
    ("hs_max_norm",       "SAPLMA hs_max"),
    ("hs_mean_norm",      "SAPLMA hs_mean"),
    ("real_cce_max",      "CCE max"),
    ("h_code_max",        "H_code max"),
    ("h_lang_max",        "H_lang max"),
    ("response_length",   "length (control)"),
]

# Multivariate arms — feature subsets matching cce_features.py::ABLATION_ARMS,
# evaluated here with the new needs_retrieval label.
ARMS = {
    "saplma_only": [
        "hs_mean_norm", "hs_std_norm", "hs_max_norm",
    ],
    "flare_only": [
        "cce_mean", "cce_max", "cce_std", "cce_spikes",
    ],
    "cce_only": [
        "real_cce_mean", "real_cce_max", "real_cce_std", "real_cce_spikes",
        "h_code_mean", "h_code_max", "h_lang_mean", "h_lang_max",
    ],
    "embedding_se_only": [
        "semantic_entropy", "semantic_entropy_norm",
        "n_clusters", "largest_cluster_frac",
    ],
    "nli_se_only": [
        "nli_se", "nli_se_norm",
        "nli_n_clusters", "nli_largest_cluster_frac",
    ],
    "flare_plus_emb_se": [
        "cce_mean", "cce_max", "cce_std", "cce_spikes",
        "semantic_entropy", "semantic_entropy_norm",
        "n_clusters", "largest_cluster_frac",
    ],
    "flare_plus_nli_se": [
        "cce_mean", "cce_max", "cce_std", "cce_spikes",
        "nli_se", "nli_se_norm",
        "nli_n_clusters", "nli_largest_cluster_frac",
    ],
    "v6_full": [
        "hs_mean_norm", "hs_std_norm", "hs_max_norm",
        "cce_mean", "cce_max", "cce_std", "cce_spikes",
        "attn_mean", "attn_max", "attn_std", "response_length",
    ],
}


def load_model(slug: str):
    with open(V3 / f"features_with_nli__{slug}.json") as f:
        feat = json.load(f)
    with open(V3 / f"phase3__{slug}.json") as f:
        phase3 = json.load(f)
    qids = feat["qids"]
    initial_correct = np.array(feat["initial_correct"], dtype=bool)
    features = feat["features"]
    always_correct = []
    for qid in qids:
        key = f"task_{qid}_retr"
        always_correct.append(bool(phase3[key]["correct"]) if key in phase3 else False)
    return qids, initial_correct, np.array(always_correct, dtype=bool), features


def best_f1_at_threshold(scores: np.ndarray, labels: np.ndarray):
    """Find threshold that maximizes F1; return (threshold, F1, P, R)."""
    if labels.sum() == 0:
        return float("nan"), 0.0, 0.0, 0.0
    p, r, t = precision_recall_curve(labels, scores)
    denom = p + r
    f1s = np.where(denom > 0, 2 * p * r / np.where(denom > 0, denom, 1), 0.0)
    i_best = int(np.argmax(f1s))
    if i_best >= len(t):
        thr = float(t[-1]) if len(t) else float("nan")
    else:
        thr = float(t[i_best])
    pred = (scores >= thr).astype(int)
    return (
        thr,
        float(f1_score(labels, pred)),
        float(precision_score(labels, pred, zero_division=0)),
        float(recall_score(labels, pred, zero_division=0)),
    )


def loo_lr(features: list[dict], feat_keys: list[str], labels: np.ndarray):
    """Leave-one-out logistic regression. Returns out-of-fold scores + preds."""
    X = np.array([[f.get(k, 0.0) for k in feat_keys] for f in features], dtype=float)
    n = len(X)
    scores = np.zeros(n)
    preds = np.zeros(n, dtype=int)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        y_train = labels[train_idx]
        if len(np.unique(y_train)) < 2:
            # Degenerate fold: only one class. Predict that class.
            scores[test_idx[0]] = float(y_train[0])
            preds[test_idx[0]] = int(y_train[0])
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X[train_idx], y_train)
        scores[test_idx[0]] = float(clf.predict_proba(X[test_idx])[0, 1])
        preds[test_idx[0]] = int(clf.predict(X[test_idx])[0])
    return scores, preds


def analyze_model(model_name: str, slug: str) -> dict:
    qids, initial_correct, always_correct, features = load_model(slug)
    n = len(qids)

    # Task category counts (the four corners).
    helps = int((always_correct & ~initial_correct).sum())
    knew = int((initial_correct & always_correct).sum())
    both_fail = int((~initial_correct & ~always_correct).sum())
    hurts = int((initial_correct & ~always_correct).sum())
    needs_retrieval = (always_correct & ~initial_correct).astype(int)

    print(f"\n=== {model_name} ===")
    print(f"  n={n}")
    print(f"  helps (needs_retrieval=1):   {helps:>3}  ({helps / n:.1%})")
    print(f"  already-knew:                {knew:>3}  ({knew / n:.1%})")
    print(f"  both-fail:                   {both_fail:>3}  ({both_fail / n:.1%})")
    print(f"  retrieval-hurts:             {hurts:>3}  ({hurts / n:.1%})")
    print(f"  → base rate of positive class: {helps / n:.3f}")

    # Univariate signals (no training, just AUC/threshold-search).
    print(f"\n  Univariate signals predicting needs_retrieval:")
    print(f"    {'signal':<20} {'AUC':>6}  {'F1*':>5}  {'P*':>5}  {'R*':>5}")
    uni = {}
    for key, name in UNIVARIATE:
        scores = np.array([f.get(key, np.nan) for f in features], dtype=float)
        if np.isnan(scores).any() or len(np.unique(scores)) < 2:
            continue
        try:
            auc = float(roc_auc_score(needs_retrieval, scores))
        except ValueError:
            auc = float("nan")
        thr, f1, p, r = best_f1_at_threshold(scores, needs_retrieval)
        # For signals where lower-is-better predicts positive (e.g., higher entropy -> more uncertain ->
        # should retrieve), AUC < 0.5 means we should flip the sign. Report abs(0.5 - auc) + 0.5 as
        # "discriminative power" alongside, but keep raw AUC honest.
        flipped = max(auc, 1 - auc) if not np.isnan(auc) else float("nan")
        print(f"    {name:<20} {auc:>6.3f}  {f1:>5.3f}  {p:>5.3f}  {r:>5.3f}    "
              f"(|disc|={flipped:.3f})")
        uni[key] = {"name": name, "auc": auc, "f1_at_best": f1,
                    "precision_at_best": p, "recall_at_best": r,
                    "discriminative_power": flipped}

    # Multivariate arms: leave-one-out logistic regression with the new label.
    print(f"\n  Multivariate arms (LOO-LR predicting needs_retrieval):")
    print(f"    {'arm':<22} {'AUC':>6}  {'F1':>5}  {'P':>5}  {'R':>5}  {'#fired':>6}")
    arms = {}
    for arm_name, feat_keys in ARMS.items():
        scores, preds = loo_lr(features, feat_keys, needs_retrieval)
        if len(np.unique(scores)) >= 2 and helps > 0:
            try:
                auc = float(roc_auc_score(needs_retrieval, scores))
            except ValueError:
                auc = float("nan")
        else:
            auc = float("nan")
        f1 = float(f1_score(needs_retrieval, preds, zero_division=0))
        p = float(precision_score(needs_retrieval, preds, zero_division=0))
        r = float(recall_score(needs_retrieval, preds, zero_division=0))
        n_fired = int(preds.sum())
        print(f"    {arm_name:<22} {auc:>6.3f}  {f1:>5.3f}  {p:>5.3f}  {r:>5.3f}  "
              f"{n_fired:>6}")
        arms[arm_name] = {"auc": auc, "f1": f1, "precision": p,
                          "recall": r, "n_fired": n_fired,
                          "n_features": len(feat_keys)}

    return {
        "model": model_name,
        "n_tasks": n,
        "categories": {
            "helps": helps, "already_knew": knew,
            "both_fail": both_fail, "retrieval_hurts": hurts,
        },
        "base_rate": helps / n if n else 0.0,
        "univariate": uni,
        "arms": arms,
    }


def main() -> int:
    out: dict = {}
    for model_name, slug in MODELS:
        out[model_name] = analyze_model(model_name, slug)
    out_path = V3 / "needs_retrieval_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
