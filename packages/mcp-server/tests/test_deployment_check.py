import json
import mcp_server._paths  # noqa: F401
from mcp_server import tools


def _setup(tmp_path, monkeypatch, state_sections, files):
    doc = {"connector": "fixture", "target": "dev", "sections": state_sections}
    (tmp_path / "state.json").write_text(json.dumps(doc))
    (tmp_path / ".reposynth").mkdir()
    (tmp_path / ".reposynth" / "targets.json").write_text(json.dumps({
        "targets": {"dev": {"connector": "fixture", "statePath": "state.json", "risk": "dev"}}}))
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    monkeypatch.setenv("REPOSYNTH_PROJECT_DIR", str(tmp_path))


def test_missing_table_reports_fail_with_provenance(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch,
           {"schema": {"hash": "x", "tables": [{"name": "orders", "columns": [], "indexes": [], "foreignKeys": []}]}},
           {"src/billing.ts": """
import { createClient } from '@supabase/supabase-js'
const supabase = createClient(1, 2)
supabase.from('invoices')
"""})
    out = tools.deployment_check("dev")
    inv = [e for e in out["expectations"] if e["assertion"].get("table") == "invoices"]
    assert inv and inv[0]["result"] == "fail"
    assert inv[0]["sites"][0]["file"] == "src/billing.ts"
    assert out["risk"] == "dev"
    assert out["summary"]["fail"] >= 1


def test_present_table_passes(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch,
           {"schema": {"hash": "x", "tables": [{"name": "orders", "columns": [], "indexes": [], "foreignKeys": []}]}},
           {"src/db.ts": """
import { createClient } from '@supabase/supabase-js'
const supabase = createClient(1, 2)
supabase.from('orders')
"""})
    out = tools.deployment_check("dev")
    orders = [e for e in out["expectations"] if e["assertion"].get("table") == "orders"]
    assert orders and orders[0]["result"] == "pass"


def test_dynamic_ref_appears_in_skipped(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, {"schema": {"hash": "x", "tables": []}},
           {"src/db.ts": """
import { createClient } from '@supabase/supabase-js'
const supabase = createClient(1, 2)
supabase.from(name)
"""})
    out = tools.deployment_check("dev")
    assert out["summary"]["skipped"] >= 1


def test_env_on_target_without_config_is_unsupported(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, {"schema": {"hash": "x", "tables": []}},
           {"supabase/functions/sync/index.ts": "const k = Deno.env.get('SVC')\n"})
    out = tools.deployment_check("dev")
    envs = [e for e in out["expectations"] if e["assertion"]["type"] == "env_name_present"]
    assert envs and envs[0]["result"] == "unsupported"


def test_error_on_unknown_target(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, {"schema": {"hash": "x", "tables": []}}, {})
    out = tools.deployment_check("nope")
    assert "error" in out
