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
