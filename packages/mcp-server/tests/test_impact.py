import mcp_server._paths  # noqa: F401
from orchestrator.infra.impact import impact
from orchestrator.infra.state_doc import make_state_doc

DOC = make_state_doc("supabase", "prod", {
    "schema": {"tables": [
        {"name": "users", "columns": [], "indexes": [], "foreignKeys": []},
        {"name": "orders", "columns": [], "indexes": [],
         "foreignKeys": [{"name": "orders_user_fk", "toTable": "users"}]}]},
    "rls": {"tables": [
        {"table": "users", "enabled": True, "policies": [
            {"name": "sel", "cmd": "SELECT", "roles": ["authenticated"],
             "using": "t", "withCheck": None}]},
        {"table": "orders", "enabled": True, "policies": []}]},
    "storage": {"buckets": [{"name": "avatars", "public": True}]},
    "functions": {"list": [{"name": "resize", "status": "ACTIVE"}]},
})


def test_drop_table_reports_inbound_fks_and_risk():
    out = impact(DOC, {"op": "drop_table", "table": "users"}, risk="prod")
    assert out["risk"] == "prod"
    assert out["result"] == "analyzed"
    assert any("orders_user_fk" in str(f) for f in out["findings"])


def test_drop_policy_reports_remaining_state():
    out = impact(DOC, {"op": "drop_policy", "table": "users", "policy": "sel"}, risk="prod")
    joined = str(out["findings"])
    assert "0 remaining" in joined or "no policies remain" in joined


def test_drop_role_finds_referencing_policies():
    out = impact(DOC, {"op": "drop_role", "role": "authenticated"}, risk="dev")
    assert any("sel" in str(f) for f in out["findings"])


def test_unknown_op_is_unknown_not_empty():
    out = impact(DOC, {"op": "vaporize_cluster"}, risk="prod")
    assert out["result"] == "unknown"


def test_delete_bucket_reports_visibility_and_not_found():
    out = impact(DOC, {"op": "delete_bucket", "bucket": "avatars"}, risk="prod")
    assert out["result"] == "analyzed" and any("PUBLIC" in str(f) for f in out["findings"])
    missing = impact(DOC, {"op": "delete_bucket", "bucket": "nope"}, risk="prod")
    assert any("not found" in str(f) for f in missing["findings"])


def test_delete_function_reports_status_and_not_found():
    out = impact(DOC, {"op": "delete_function", "function": "resize"}, risk="dev")
    assert any("404" in str(f) for f in out["findings"])
    missing = impact(DOC, {"op": "delete_function", "function": "nope"}, risk="dev")
    assert any("not found" in str(f) for f in missing["findings"])


def test_missing_required_section_is_unknown():
    bare = make_state_doc("postgres", "prod", {"schema": {"tables": []}})
    out = impact(bare, {"op": "delete_bucket", "bucket": "avatars"}, risk="prod")
    assert out["result"] == "unknown"
    assert any("storage" in str(f) for f in out["findings"])


def test_drop_policy_nonexistent_policy_is_flagged():
    out = impact(DOC, {"op": "drop_policy", "table": "users", "policy": "ghost"}, risk="prod")
    assert any("not found" in str(f) for f in out["findings"])
