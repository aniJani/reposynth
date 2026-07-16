"""Legibility benchmark: agent-alone vs agent-with-tools over seeded cases.

Scoring and case-loading are pure (unit-tested). The model comparison in main()
is gated on ANTHROPIC_API_KEY. All cases use the fixture connector — offline,
deterministic, and doubling as the connector regression harness.
"""
import json
import os
import sys
from pathlib import Path

_MCP = Path(__file__).resolve().parents[1] / "packages" / "mcp-server"
sys.path.insert(0, str(_MCP))
import mcp_server._paths  # noqa: E402,F401
from mcp_server import tools  # noqa: E402


def load_cases(cases_dir):
    out = []
    for case in sorted(Path(cases_dir).iterdir()):
        if not case.is_dir():
            continue
        truth = json.loads((case / "truth.json").read_text())
        out.append({"name": case.name, "question": truth["question"], "answer": truth["answer"],
                    "repo_dir": str(case / "repo"), "state_path": str(case / "state.json")})
    return out


def score(rows):
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    cw = sum(1 for r in rows if r["confident"] and not r["correct"])
    return {"n": n, "correct": correct, "correct_rate": correct / n if n else 0.0,
            "confidently_wrong": cw, "confidently_wrong_rate": cw / n if n else 0.0}


def tool_answer(case):
    """Deterministic tool-derived answer for the missing-table question type."""
    os.environ["REPOSYNTH_PROJECT_DIR"] = str(Path(case["repo_dir"]).parent)
    # point a fixture target at this case's state.json
    reposynth = Path(case["repo_dir"]).parent / ".reposynth"
    reposynth.mkdir(exist_ok=True)
    (reposynth / "targets.json").write_text(json.dumps({
        "targets": {"dev": {"connector": "fixture", "statePath": case["state_path"], "risk": "dev"}}}))
    res = tools.deployment_check("dev", repo_path=case["repo_dir"])
    fails = [e for e in res.get("expectations", []) if e["result"] == "fail"]
    if fails:
        missing = ", ".join(sorted(e["assertion"].get("table", "?") for e in fails))
        return {"answer": f"yes: {missing}", "confidence": "high"}
    return {"answer": "no", "confidence": "high"}


def main():
    cases = load_cases(str(Path(__file__).parent / "cases"))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY unset — running tool-derived answers only (no model comparison).")
        for c in cases:
            print(f"  {c['name']}: tool -> {tool_answer(c)}  (truth: {c['answer']})")
        return
    # With a key: ask the model {answer, confidence} alone vs with-tools, then score().
    # (Model wiring intentionally minimal; extend as the benchmark grows.)
    print("Model comparison path: implement model calls, collect rows, call score().")


if __name__ == "__main__":
    main()
