import mcp_server._paths  # noqa: F401
from orchestrator.infra.expectations import extract


def _write(root, rel, content):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def test_extract_dedupes_and_provenances(tmp_path):
    _write(tmp_path, "src/db.ts", """
import { createClient } from '@supabase/supabase-js'
const supabase = createClient(process.env.URL, process.env.KEY)
export const q = () => supabase.from('orders')
""")
    _write(tmp_path, "src/api.ts", """
import { supabase } from './db'
const r = supabase.from('orders').select('*')
""")
    # supabase in api.ts isn't bound to createClient there, but the repo-wide
    # pre-pass binds it from db.ts -> both sites count.
    out = extract(str(tmp_path))
    orders = [a for a in out["assertions"] if a["assertion"] == {"type": "table_exists", "table": "orders"}]
    assert len(orders) == 1
    assert len(orders[0]["sites"]) == 2


def test_extract_always_notes_rls(tmp_path):
    _write(tmp_path, "m.py", 'class O(Base):\n    __tablename__ = "orders"\n')
    out = extract(str(tmp_path))
    assert any("RLS" in n for n in out["notes"])


def test_extract_notes_app_env_count(tmp_path):
    _write(tmp_path, "app.py", 'import os\nX = os.environ["STRIPE_KEY"]\nY = os.getenv("DB_URL")\n')
    out = extract(str(tmp_path))
    assert any("app" in n.lower() and "2" in n for n in out["notes"])


def test_extract_skips_node_modules(tmp_path):
    _write(tmp_path, "node_modules/pkg/index.ts",
           "const supabase = createClient(1,2)\nsupabase.from('secret')\n")
    out = extract(str(tmp_path))
    assert all(a["assertion"].get("table") != "secret" for a in out["assertions"])
