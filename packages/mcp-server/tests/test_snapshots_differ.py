import mcp_server._paths  # noqa: F401
from orchestrator.infra import differ, snapshots
from orchestrator.infra.state_doc import make_state_doc


def make_doc(tables, buckets=None, env=None):
    sections = {"schema": {"tables": tables}}
    if buckets is not None:
        sections["storage"] = {"buckets": buckets}
    if env is not None:
        sections["config"] = env
    return make_state_doc("postgres", "prod", sections)


T_USERS = {"name": "users", "columns": [], "indexes": [], "foreignKeys": []}
T_ORDERS = {"name": "orders", "columns": [], "indexes": [], "foreignKeys": []}
T_ORDERS_V2 = {"name": "orders",
               "columns": [{"name": "total", "type": "numeric", "nullable": True}],
               "indexes": [], "foreignKeys": []}


def test_save_load_roundtrip(tmp_path):
    doc = make_doc([T_USERS])
    meta = snapshots.save_snapshot(doc, project=tmp_path, label="baseline")
    assert "baseline" in meta["id"]
    loaded = snapshots.load_snapshot(meta["id"], project=tmp_path)
    assert loaded == doc
    assert snapshots.list_snapshots(project=tmp_path)[0]["id"] == meta["id"]


def test_load_missing_lists_available(tmp_path):
    snapshots.save_snapshot(make_doc([T_USERS]), project=tmp_path)
    try:
        snapshots.load_snapshot("nope", project=tmp_path)
        assert False, "expected KeyError"
    except KeyError as exc:
        assert "available" in str(exc).lower()


def test_diff_detects_added_removed_changed():
    a = make_doc([T_USERS, T_ORDERS])
    b = make_doc([T_ORDERS_V2])
    out = differ.diff(a, b)
    schema = out["sections"]["schema"]
    assert schema["removed"] == ["users"]
    assert schema["changed"] == ["orders"]
    assert schema["added"] == []


def test_diff_unchanged_section_listed():
    a = make_doc([T_USERS], buckets=[{"name": "avatars", "public": False}])
    b = make_doc([T_ORDERS], buckets=[{"name": "avatars", "public": False}])
    out = differ.diff(a, b)
    assert "storage" in out["unchanged"]


def test_diff_config_rotation():
    a = make_doc([T_USERS], env={"envNames": ["KEY"], "valueHashes": {"KEY": "sha256:aaa"}})
    b = make_doc([T_USERS], env={"envNames": ["KEY"], "valueHashes": {"KEY": "sha256:bbb"}})
    out = differ.diff(a, b)
    assert out["sections"]["config"]["rotated"] == ["KEY"]
