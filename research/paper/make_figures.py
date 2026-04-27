"""Generate the Pareto-frontier figure for the v3 paper.

Reads research/paper/v3_results/v3_final_with_nli_and_ci.json and writes
research/paper/figures/pareto_v3.{pdf,png}: a 1-row 3-panel figure plotting
accuracy vs. retrieval-save-rate per model with bootstrap 95% CIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_DISPLAY = {
    "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen2.5-Coder-7B",
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5": "DeepSeek-Coder-7B-v1.5",
    "codellama/CodeLlama-7b-Instruct-hf": "CodeLlama-7B",
}

# Plot order — keep paper-narrative groups together (anchors first, then SE families).
ARM_ORDER = [
    "saplma_only",
    "cce_only",
    "v6_full",
    "embedding_se_only",
    "flare_plus_emb_se",
    "flare_only",
    "nli_se_only",
    "flare_plus_nli_se",
]

ARM_DISPLAY = {
    "saplma_only": "SAPLMA",
    "cce_only": "CCE",
    "v6_full": "V6 (all)",
    "embedding_se_only": "emb-SE",
    "flare_plus_emb_se": "FLARE+emb-SE",
    "flare_only": "FLARE",
    "nli_se_only": "NLI-SE",
    "flare_plus_nli_se": "FLARE+NLI-SE",
}

# Color groups: hidden-state probes (cool grays), entropy/SE families (warm/blue/green).
ARM_STYLE = {
    "saplma_only":       dict(color="#888888", marker="o"),
    "cce_only":          dict(color="#aa6666", marker="o"),
    "v6_full":           dict(color="#6666aa", marker="o"),
    "embedding_se_only": dict(color="#e08214", marker="s"),
    "flare_plus_emb_se": dict(color="#fdb863", marker="s"),
    "flare_only":        dict(color="#1f77b4", marker="D"),
    "nli_se_only":       dict(color="#2ca02c", marker="^"),
    "flare_plus_nli_se": dict(color="#67c267", marker="^"),
}


def pareto_frontier(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Return points on the upper-right Pareto frontier (max save, max accuracy).

    A point (s_i, a_i) is dominated if some other (s_j, a_j) has s_j >= s_i and
    a_j >= a_i with at least one strict inequality.
    """
    front = []
    for i, (si, ai, label) in enumerate(points):
        dominated = False
        for j, (sj, aj, _) in enumerate(points):
            if i == j:
                continue
            if sj >= si and aj >= ai and (sj > si or aj > ai):
                dominated = True
                break
        if not dominated:
            front.append((si, ai, label))
    return sorted(front, key=lambda p: p[0])


def plot_panel(ax, model_key: str, model_data: dict) -> None:
    arms = model_data["arms"]

    # Anchors.
    always_acc = model_data["always_retrieve_accuracy"]
    always_lo, always_hi = model_data["always_retrieve_accuracy_ci"]
    init_acc = model_data["initial_accuracy"]
    init_lo, init_hi = model_data["initial_accuracy_ci"]

    # Plot the always-retrieve ceiling band as a horizontal reference.
    ax.axhspan(always_lo, always_hi, color="#cccccc", alpha=0.35, zorder=0)
    ax.axhline(always_acc, color="#444444", linestyle="--", linewidth=0.8, zorder=1,
               label=f"always-retrieve ({always_acc:.3f})")

    # Anchor: never-retrieve (= no_signal_baseline = initial_accuracy at save=1.0).
    ax.errorbar(
        1.0, init_acc,
        yerr=[[init_acc - init_lo], [init_hi - init_acc]],
        fmt="X", color="#222222", markersize=8, capsize=3,
        label=f"never-retrieve ({init_acc:.3f})",
        zorder=4,
    )

    # Per-arm points + CIs.
    pareto_pts = [(0.0, always_acc, "always-retrieve"),
                  (1.0, init_acc, "never-retrieve")]
    for arm in ARM_ORDER:
        if arm not in arms:
            continue
        r = arms[arm]
        s = r["retrieval_save_rate"]
        a = r["final_accuracy"]
        lo, hi = r["final_accuracy_ci"]
        style = ARM_STYLE[arm]
        ax.errorbar(
            s, a,
            yerr=[[a - lo], [hi - a]],
            fmt=style["marker"], color=style["color"],
            markersize=7, capsize=2.5, linewidth=1.0,
            label=ARM_DISPLAY[arm],
            zorder=3,
        )
        pareto_pts.append((s, a, arm))

    # Pareto frontier (thin connecting line).
    front = pareto_frontier(pareto_pts)
    if len(front) >= 2:
        xs = [p[0] for p in front]
        ys = [p[1] for p in front]
        ax.plot(xs, ys, color="#888888", linestyle=":", linewidth=1.0,
                alpha=0.7, zorder=2)

    ax.set_title(MODEL_DISPLAY.get(model_key, model_key), fontsize=10)
    ax.set_xlabel("retrieval save rate")
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in-file",
        default="research/paper/v3_results/v3_final_with_nli_and_ci.json",
    )
    p.add_argument(
        "--out-dir",
        default="research/paper/figures",
    )
    args = p.parse_args()

    with open(args.in_file) as f:
        data = json.load(f)

    # Order panels: Qwen, DeepSeek, CodeLlama (matches paper §5 anchor order).
    panel_order = [
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "codellama/CodeLlama-7b-Instruct-hf",
    ]
    panel_keys = [k for k in panel_order if k in data]

    fig, axes = plt.subplots(1, len(panel_keys), figsize=(13, 5.0), sharey=True)
    if len(panel_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, panel_keys):
        plot_panel(ax, key, data[key])

    axes[0].set_ylabel("final accuracy")
    # Tight, common y-range across panels (data span: 0.725..0.963).
    axes[0].set_ylim(0.68, 1.005)

    # Single shared legend below the panels: strip per-panel numerics from anchor
    # labels first, then dedupe by the cleaned name (keeps insertion order).
    handles_labels = []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            base = l.split(" (")[0] if l.startswith(("always-retrieve", "never-retrieve")) else l
            if base in seen:
                continue
            seen.add(base)
            handles_labels.append((h, base))
    handles, labels = zip(*handles_labels)

    fig.suptitle(
        "Pareto frontier: accuracy vs. retrieval-save rate (n=80, bootstrap 95% CI)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.96))
    fig.legend(
        handles, labels,
        loc="lower center", ncol=min(len(labels), 5),
        frameon=False, fontsize=8.5,
        bbox_to_anchor=(0.5, 0.0),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "pareto_v3.pdf"
    png_path = out_dir / "pareto_v3.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"[done] {pdf_path}")
    print(f"[done] {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
