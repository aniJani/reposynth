import json

import pytest

import mcp_server._paths  # noqa: F401
from mcp_server import tools
from orchestrator.infra.state_doc import make_state_doc

DOC_SECTIONS = {
    "schema": {"tables": [{"name": "users", "columns": [], "indexes": [], "foreignKeys": []}]},
    "rls": {"tables": [{"table": "users", "enabled": True, "policies": []}]},
}


class FakeConnector:
    id = "postgres"

    def detect(self, project_dir):
        return {"detected": False}

    def fetch_state(self, target):
        return make_state_doc("postgres", target["name"], DOC_SECTIONS)


@pytest.fixture
def project(tmp_path, monkeypatch):
    d = tmp_path / ".reposynth"
    d.mkdir()
    (d / "targets.json").write_text(json.dumps({"targets": {
        "prod": {"connector": "postgres", "urlEnv": "PROD_URL", "risk": "prod"}}}))
    monkeypatch.setenv("REPOSYNTH_PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(tools, "get_connector", lambda _id: FakeConnector())
    return tmp_path


def test_infra_state_carries_risk(project):
    out = tools.infra_state("prod")
    assert out["target"] == "prod"
    assert out["risk"] == "prod"
    assert "schema" in out["state"]["sections"]


def test_infra_state_single_section(project):
    out = tools.infra_state("prod", section="rls")
    assert out["state"]["tables"][0]["table"] == "users"


def test_infra_verify_and_impact(project):
    v = tools.infra_verify("prod", [{"type": "table_exists", "table": "users"}])
    assert v["risk"] == "prod" and v["summary"]["pass"] == 1
    i = tools.infra_impact("prod", {"op": "drop_table", "table": "users"})
    assert i["risk"] == "prod" and i["result"] == "analyzed"


def test_snapshot_then_drift_live(project):
    snap = tools.infra_snapshot("prod", label="base")
    assert snap["risk"] == "prod"
    drift = tools.infra_drift(snap["snapshot"]["id"], "live:prod")
    assert drift["risk"] == "prod"
    assert drift["diff"]["unchanged"]  # identical docs → sections unchanged


def test_unknown_target_returns_error(project):
    out = tools.infra_state("staging")
    assert "error" in out


def test_drift_traversal_ref_returns_error(project):
    out = tools.infra_drift("../../../etc/passwd", "live:prod")
    assert "error" in out and "Invalid snapshot id" in out["error"]


class ExplodingConnector(FakeConnector):
    def fetch_state(self, target):
        raise ValueError("connection refused: db host unreachable")


def test_connector_runtime_error_returns_error(project, monkeypatch):
    monkeypatch.setattr(tools, "get_connector", lambda _id: ExplodingConnector())
    out = tools.infra_state("prod")
    assert "error" in out and "unreachable" in out["error"]
