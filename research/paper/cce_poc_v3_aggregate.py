"""Aggregate per-model v3 results into a unified cross-model table.

Reads results__<slug>.json from a directory and produces a combined
JSON + a printed markdown table of (arm, model, F1) and a Pareto summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True,
                   help="dir with results__*.json files (one per model)")
    p.add_argument("--out", default="research/paper/v3_combined_results.json")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    files = sorted(in_dir.glob("results__*.json"))
    if not files:
        print(f"no results__*.json found in {in_dir}")
        return 1

    per_model = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        per_model[d["model"]] = d

    # Print Phase-2 F1 cross-model
    arms = sorted({arm for d in per_model.values()
                    for arm in d.get("phase2_loo_ablation", {})})
    print("\n## Phase-2 F1 across models\n")
    header = ["arm"] + list(per_model.keys())
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))
    for arm in arms:
        row = [arm]
        for m in per_model:
            f1 = per_model[m].get("phase2_loo_ablation", {}).get(arm, {}).get("f1", float("nan"))
            row.append(f"{f1:.3f}" if f1 == f1 else "—")
        print(" | ".join(row))

    # Print Phase-3 final accuracy cross-model
    print("\n## Phase-3 final accuracy across models\n")
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))
    for arm in arms:
        row = [arm]
        for m in per_model:
            v = per_model[m].get("phase3_end_to_end", {}).get(arm, {}).get("final_accuracy")
            row.append(f"{v:.3f}" if v is not None else "—")
        print(" | ".join(row))

    # Phase-3 save rates
    print("\n## Phase-3 retrieval save rate across models\n")
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))
    for arm in arms:
        row = [arm]
        for m in per_model:
            v = per_model[m].get("phase3_end_to_end", {}).get(arm, {}).get("retrieval_save_rate")
            row.append(f"{v:.2f}" if v is not None else "—")
        print(" | ".join(row))

    # Save combined
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"per_model": per_model}, f, indent=2)
    print(f"\n[done] combined results at {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
