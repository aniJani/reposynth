import json
from pathlib import Path

import pytest

import mcp_server._paths  # noqa: F401
from orchestrator.infra.targets import get_target, load_targets, resolve_env

VALID = {
    "targets": {
        "dev": {"connector": "supabase", "projectRef": "abc123",
                "tokenEnv": "SUPABASE_ACCESS_TOKEN", "risk": "dev"},
        "prod": {"connector": "postgres", "urlEnv": "PROD_PG_URL_READONLY", "risk": "prod"},
    }
}


def write_targets(tmp_path: Path, data: dict) -> Path:
    d = tmp_path / ".reposynth"
    d.mkdir()
    (d / "targets.json").write_text(json.dumps(data))
    return tmp_path


def test_load_targets_injects_name_and_keeps_risk(tmp_path):
    project = write_targets(tmp_path, VALID)
    targets = load_targets(project)
    assert targets["prod"]["name"] == "prod"
    assert targets["prod"]["risk"] == "prod"
    assert targets["dev"]["connector"] == "supabase"


def test_load_targets_rejects_bad_risk(tmp_path):
    bad = {"targets": {"x": {"connector": "postgres", "urlEnv": "U", "risk": "yolo"}}}
    project = write_targets(tmp_path, bad)
    with pytest.raises(ValueError):
        load_targets(project)


def test_get_target_miss_names_available(tmp_path):
    project = write_targets(tmp_path, VALID)
    with pytest.raises(KeyError) as exc:
        get_target("staging", project)
    assert "dev" in str(exc.value) and "prod" in str(exc.value)


def test_resolve_env_reads_and_errors(tmp_path, monkeypatch):
    project = write_targets(tmp_path, VALID)
    target = get_target("prod", project)
    monkeypatch.setenv("PROD_PG_URL_READONLY", "postgres://ro@host/db")
    assert resolve_env(target, "urlEnv") == "postgres://ro@host/db"
    monkeypatch.delenv("PROD_PG_URL_READONLY")
    with pytest.raises(RuntimeError) as exc:
        resolve_env(target, "urlEnv")
    assert "PROD_PG_URL_READONLY" in str(exc.value)
