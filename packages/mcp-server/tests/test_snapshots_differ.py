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


def test_save_snapshot_never_clobbers_on_id_collision(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshots, "_utc_ts", lambda: "20260101T000000Z")
    monkeypatch.setattr(snapshots, "section_hash", lambda _doc: "sha256:" + "0" * 64)
    doc_a = make_doc([T_USERS])
    doc_b = make_doc([T_ORDERS])
    meta_a = snapshots.save_snapshot(doc_a, project=tmp_path)
    meta_b = snapshots.save_snapshot(doc_b, project=tmp_path)
    assert meta_a["id"] != meta_b["id"]
    assert snapshots.load_snapshot(meta_a["id"], project=tmp_path) == doc_a
    assert snapshots.load_snapshot(meta_b["id"], project=tmp_path) == doc_b
    again = snapshots.save_snapshot(doc_a, project=tmp_path)
    assert again["id"] == meta_a["id"]


def test_diff_section_added_and_removed():
    a = make_doc([T_USERS])
    b = make_doc([T_USERS], buckets=[{"name": "avatars", "public": False}])
    out = differ.diff(a, b)
    assert out["sections"]["storage"] == {"status": "added"}
    back = differ.diff(b, a)
    assert back["sections"]["storage"] == {"status": "removed"}


def test_diff_rls_functions_and_auth_branches():
    a = make_state_doc("supabase", "dev", {
        "rls": {"tables": [{"table": "users", "enabled": True, "policies": []}]},
        "functions": {"list": [{"name": "resize", "status": "ACTIVE"}]},
        "auth": {"providers": ["github"], "settings": {}},
    })
    b = make_state_doc("supabase", "dev", {
        "rls": {"tables": [{"table": "users", "enabled": False, "policies": []}]},
        "functions": {"list": [{"name": "resize", "status": "ACTIVE"},
                               {"name": "mail", "status": "ACTIVE"}]},
        "auth": {"providers": ["github", "google"], "settings": {}},
    })
    out = differ.diff(a, b)
    assert out["sections"]["rls"]["changed"] == ["users"]
    assert out["sections"]["functions"]["added"] == ["mail"]
    assert out["sections"]["auth"]["added"] == ["google"]


def test_label_sanitized_and_truncated(tmp_path):
    doc = make_doc([T_USERS])
    meta = snapshots.save_snapshot(doc, project=tmp_path, label="my label!/" + "x" * 40)
    label_part = meta["id"].split("-", 2)[2]
    assert len(label_part) == 32
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    assert set(label_part) <= allowed
