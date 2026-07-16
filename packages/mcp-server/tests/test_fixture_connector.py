import json
import mcp_server._paths  # noqa: F401
from orchestrator.infra.connectors.base import get_connector
from orchestrator.infra.connectors import fixture  # noqa: F401  (register)


def test_fixture_serves_recorded_doc(tmp_path, monkeypatch):
    doc = {"connector": "fixture", "target": "dev",
           "sections": {"schema": {"hash": "x", "tables": [{"name": "orders"}]}}}
    (tmp_path / "state.json").write_text(json.dumps(doc))
    monkeypatch.setenv("REPOSYNTH_PROJECT_DIR", str(tmp_path))
    conn = get_connector("fixture")
    got = conn.fetch_state({"name": "dev", "statePath": "state.json"})
    assert got == doc


def test_fixture_absolute_path(tmp_path):
    doc = {"connector": "fixture", "target": "dev", "sections": {}}
    p = tmp_path / "abs.json"
    p.write_text(json.dumps(doc))
    conn = get_connector("fixture")
    assert conn.fetch_state({"name": "dev", "statePath": str(p)}) == doc
