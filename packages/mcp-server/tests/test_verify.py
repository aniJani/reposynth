import mcp_server._paths  # noqa: F401
from orchestrator.infra.state_doc import make_state_doc
from orchestrator.infra.verify import verify

DOC = make_state_doc("supabase", "dev", {
    "schema": {"tables": [{
        "name": "orders",
        "columns": [{"name": "id", "type": "uuid", "nullable": False}],
        "indexes": ["orders_pkey"], "foreignKeys": []}]},
    "rls": {"tables": [{"table": "orders", "enabled": True, "policies": [
        {"name": "sel", "cmd": "SELECT", "roles": ["authenticated"],
         "using": "true", "withCheck": None}]}]},
    "auth": {"providers": ["github"], "settings": {}},
    "storage": {"buckets": [{"name": "avatars", "public": False}]},
    "functions": {"list": [{"name": "resize", "status": "ACTIVE"}]},
    "config": {"envNames": ["STRIPE_KEY"], "valueHashes": {"STRIPE_KEY": "sha256:x"}},
})


def run_one(assertion):
    return verify(DOC, [assertion])["results"][0]


def test_passing_assertions():
    for a in [
        {"type": "table_exists", "table": "orders"},
        {"type": "column_matches", "table": "orders", "column": "id", "col_type": "uuid", "nullable": False},
        {"type": "rls_enabled", "table": "orders"},
        {"type": "policy_exists", "table": "orders", "cmd": "SELECT", "role": "authenticated"},
        {"type": "index_exists", "table": "orders", "index": "orders_pkey"},
        {"type": "auth_provider_enabled", "provider": "github"},
        {"type": "bucket_exists", "bucket": "avatars", "public": False},
        {"type": "function_deployed", "function": "resize"},
        {"type": "env_name_present", "env": "STRIPE_KEY"},
    ]:
        assert run_one(a)["result"] == "pass", a


def test_failing_assertion_reports_actual():
    r = run_one({"type": "table_exists", "table": "invoices"})
    assert r["result"] == "fail"
    assert "orders" in str(r["actual"])


def test_bucket_visibility_mismatch_fails():
    r = run_one({"type": "bucket_exists", "bucket": "avatars", "public": True})
    assert r["result"] == "fail"


def test_unknown_type_is_unsupported():
    r = run_one({"type": "quantum_check"})
    assert r["result"] == "unsupported"


def test_missing_section_is_unsupported():
    bare = make_state_doc("postgres", "prod", {"schema": {"tables": []}})
    r = verify(bare, [{"type": "env_name_present", "env": "X"}])["results"][0]
    assert r["result"] == "unsupported"


def test_summary_counts():
    out = verify(DOC, [
        {"type": "table_exists", "table": "orders"},
        {"type": "table_exists", "table": "nope"},
        {"type": "quantum_check"},
    ])
    assert out["summary"] == {"pass": 1, "fail": 1, "unsupported": 1}
