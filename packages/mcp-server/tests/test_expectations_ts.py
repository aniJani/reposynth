import mcp_server._paths  # noqa: F401
from orchestrator.infra.expectations import collect_clients, extract_ts

APP = '''
import { createClient } from '@supabase/supabase-js'
const supabase = createClient(process.env.URL, process.env.KEY)
const a = await supabase.from('orders').select('*')
const b = supabase.storage.from('avatars').list()
await supabase.functions.invoke('sync-job')
await supabase.auth.signInWithOAuth({ provider: 'google' })
const arr = Array.from('abc')
const dyn = supabase.from(tableName)
'''

FN = '''
const key = Deno.env.get('SERVICE_ROLE')
'''

DRIZZLE = '''
import { pgTable, text } from 'drizzle-orm/pg-core'
export const users = pgTable('users', { id: text('id') })
'''


def test_collect_clients_finds_binding():
    assert collect_clients([("app.ts", APP)]) == {"supabase"}


def test_client_calls_map_to_assertions():
    clients = collect_clients([("app.ts", APP)])
    out = extract_ts(APP, "src/app.ts", clients)
    got = {(f["assertion"]["type"], f["assertion"].get("table") or f["assertion"].get("bucket")
            or f["assertion"].get("function") or f["assertion"].get("provider"))
           for f in out["findings"]}
    assert ("table_exists", "orders") in got
    assert ("bucket_exists", "avatars") in got
    assert ("function_deployed", "sync-job") in got
    assert ("auth_provider_enabled", "google") in got


def test_array_from_is_ignored_not_skipped():
    clients = collect_clients([("app.ts", APP)])
    out = extract_ts(APP, "src/app.ts", clients)
    # 'abc' must not appear anywhere
    blob = str(out)
    assert "abc" not in blob


def test_dynamic_from_is_skipped():
    clients = collect_clients([("app.ts", APP)])
    out = extract_ts(APP, "src/app.ts", clients)
    assert any(s["reason"] == "dynamic argument" for s in out["skipped"])


def test_process_env_in_app_code_is_app_env():
    out = extract_ts(APP, "src/app.ts", set())
    envs = sorted(e["env"] for e in out["app_env"])
    assert envs == ["KEY", "URL"]


def test_deno_env_under_functions_dir_is_emitted():
    out = extract_ts(FN, "supabase/functions/sync/index.ts", set())
    types = [f["assertion"]["type"] for f in out["findings"]]
    assert "env_name_present" in types
    assert out["findings"][0]["assertion"]["env"] == "SERVICE_ROLE"


def test_pgtable_becomes_table_exists():
    out = extract_ts(DRIZZLE, "schema.ts", set())
    assert any(f["assertion"] == {"type": "table_exists", "table": "users"}
               for f in out["findings"])


def test_dynamic_call_shapes_are_skipped():
    src = """
import { createClient } from '@supabase/supabase-js'
const supabase = createClient(1, 2)
supabase.from(getName())
supabase.from(`t_${x}`)
supabase.from(TABLES.orders)
"""
    clients = collect_clients([("a.ts", src)])
    out = extract_ts(src, "a.ts", clients)
    assert len(out["skipped"]) == 3
    assert all(s["reason"] == "dynamic argument" for s in out["skipped"])
    assert out["findings"] == []


def test_dynamic_oauth_provider_is_skipped():
    src = """
const supabase = createClient(1, 2)
supabase.auth.signInWithOAuth({ provider: providerVar })
"""
    out = extract_ts(src, "a.ts", {"supabase"})
    assert any(s["reason"] == "dynamic argument" for s in out["skipped"])
    assert not any(f["assertion"]["type"] == "auth_provider_enabled" for f in out["findings"])


def test_dynamic_pgtable_name_is_skipped():
    src = """
import { pgTable } from 'drizzle-orm/pg-core'
export const t = pgTable(tableNameVar, { id: 1 })
"""
    out = extract_ts(src, "schema.ts", set())
    assert any(s["reason"] == "dynamic argument" for s in out["skipped"])
    assert not any(f["assertion"].get("table") for f in out["findings"])
