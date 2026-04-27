"""Generate the headline figure for the v3 paper.

Reads research/paper/v3_results/v3_final_with_nli_and_ci.json and writes
research/paper/figures/headline_v3.{pdf,png}: a 1-row 3-panel forest plot
showing, per model, the three accuracies that carry the paper's claim:

    always-retrieve  (the ceiling, 100% retrieval cost)
    headline adaptive method (the win, X% retrieval cost)
    never-retrieve  (the floor, 0% retrieval cost)

Each row shows the accuracy point estimate as a circle and the bootstrap
95% CI as a thick horizontal segment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines


MODEL_DISPLAY = {
    "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen2.5-Coder-7B",
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5": "DeepSeek-Coder-7B-v1.5",
    "codellama/CodeLlama-7b-Instruct-hf": "CodeLlama-7B",
}

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

HEADLINE_ARM = {
    "Qwen/Qwen2.5-Coder-7B-Instruct": "nli_se_only",
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5": "nli_se_only",
    "codellama/CodeLlama-7b-Instruct-hf": "flare_plus_emb_se",
}

C_ALWAYS = "#1f4e79"      # deep blue — ceiling anchor
C_HEADLINE = "#2ca02c"    # green — the adaptive win
C_NEVER = "#888888"       # gray — floor


def panel_rows(model_data: dict, headline_arm: str) -> list[dict]:
    arms = model_data["arms"]
    headline = arms[headline_arm]
    cost_pct = int(round((1 - headline["retrieval_save_rate"]) * 100))
    return [
        {  # row 0 = top
            "label": "always-retrieve",
            "cost_label": "100% retrieval cost",
            "acc": model_data["always_retrieve_accuracy"],
            "ci": model_data["always_retrieve_accuracy_ci"],
            "color": C_ALWAYS,
            "weight": "normal",
        },
        {
            "label": ARM_DISPLAY[headline_arm],
            "cost_label": f"{cost_pct}% retrieval cost (adaptive)",
            "acc": headline["final_accuracy"],
            "ci": headline["final_accuracy_ci"],
            "color": C_HEADLINE,
            "weight": "bold",
        },
        {
            "label": "never-retrieve",
            "cost_label": "0% retrieval cost",
            "acc": model_data["initial_accuracy"],
            "ci": model_data["initial_accuracy_ci"],
            "color": C_NEVER,
            "weight": "normal",
        },
    ]


def plot_panel(ax, model_key: str, model_data: dict) -> None:
    rows = panel_rows(model_data, HEADLINE_ARM[model_key])
    y_positions = [2, 1, 0]  # top -> bottom: always, headline, never

    # Faint vertical reference at the always-retrieve point estimate so the
    # eye can drop a line and see how close the adaptive bar comes.
    always_acc = rows[0]["acc"]
    ax.axvline(always_acc, color=C_ALWAYS, linestyle=":", linewidth=0.9,
               alpha=0.45, zorder=1)

    for y, row in zip(y_positions, rows):
        lo, hi = row["ci"]
        # CI segment.
        ax.plot([lo, hi], [y, y],
                color=row["color"], linewidth=5.0, alpha=0.55,
                solid_capstyle="round", zorder=3)
        # Point estimate.
        ax.plot(row["acc"], y, "o",
                color=row["color"], markersize=11,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        # Numeric annotation directly above the point estimate.
        ax.text(row["acc"], y + 0.18,
                f'{row["acc"]:.3f}  [{lo:.3f}, {hi:.3f}]',
                ha="center", va="bottom",
                fontsize=8.5, color=row["color"],
                fontweight=row["weight"], zorder=6)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [f'{r["label"]}\n{r["cost_label"]}' for r in rows],
        fontsize=9,
    )
    # Bold the headline row's y-tick label.
    for tick, row in zip(ax.get_yticklabels(), rows):
        if row["weight"] == "bold":
            tick.set_color(row["color"])
            tick.set_fontweight("bold")

    ax.set_ylim(-0.6, 2.7)
    ax.set_xlim(0.70, 1.005)
    ax.set_xticks([0.70, 0.80, 0.90, 1.00])
    ax.set_xlabel("accuracy")
    ax.set_title(MODEL_DISPLAY.get(model_key, model_key), fontsize=11)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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

    panel_order = [
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "codellama/CodeLlama-7b-Instruct-hf",
    ]
    panel_keys = [k for k in panel_order if k in data]

    fig, axes = plt.subplots(1, len(panel_keys), figsize=(14, 3.6))
    if len(panel_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, panel_keys):
        plot_panel(ax, key, data[key])

    fig.suptitle(
        "Adaptive retrieval matches always-retrieve at a fraction of the cost "
        "(n=80, points = bootstrap mean, segments = 95% CI)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "headline_v3.pdf"
    png_path = out_dir / "headline_v3.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"[done] {pdf_path}")
    print(f"[done] {png_path}")

    # Clean up the obsolete pareto figure if it lingered.
    for stale in (out_dir / "pareto_v3.pdf", out_dir / "pareto_v3.png"):
        if stale.exists():
            stale.unlink()
            print(f"[clean] removed stale {stale.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
