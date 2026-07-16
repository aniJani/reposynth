import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmark"))
import runner  # noqa: E402


def test_score_counts_confidently_wrong():
    rows = [{"correct": True, "confident": True},
            {"correct": False, "confident": True},   # confidently wrong
            {"correct": False, "confident": False},
            {"correct": True, "confident": False}]
    s = runner.score(rows)
    assert s["n"] == 4
    assert s["correct"] == 2
    assert s["correct_rate"] == 0.5
    assert s["confidently_wrong"] == 1
    assert s["confidently_wrong_rate"] == 0.25


def test_load_cases_reads_truth(tmp_path):
    import json
    c = tmp_path / "cases" / "x"
    (c / "repo").mkdir(parents=True)
    (c / "truth.json").write_text(json.dumps({"question": "q?", "answer": "yes"}))
    (c / "state.json").write_text("{}")
    cases = runner.load_cases(str(tmp_path / "cases"))
    assert cases[0]["question"] == "q?" and cases[0]["answer"] == "yes"
    assert cases[0]["name"] == "x"


def test_tool_answer_matches_truth_for_all_cases():
    import os
    from pathlib import Path as _P
    cases_dir = str(_P(runner.__file__).parent / "cases")
    saved = os.environ.get("REPOSYNTH_PROJECT_DIR")
    try:
        cases = {c["name"]: c for c in runner.load_cases(cases_dir)}
        assert runner.tool_answer(cases["missing_table"])["answer"].startswith("yes")
        assert runner.tool_answer(cases["rls_disabled"])["answer"] == "no"
        assert runner.tool_answer(cases["bucket_public"])["answer"] == "yes"
    finally:
        if saved is None:
            os.environ.pop("REPOSYNTH_PROJECT_DIR", None)
        else:
            os.environ["REPOSYNTH_PROJECT_DIR"] = saved
