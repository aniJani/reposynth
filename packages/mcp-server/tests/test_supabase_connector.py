import json

import mcp_server._paths  # noqa: F401
from orchestrator.infra.connectors import postgres
from orchestrator.infra.connectors.base import get_connector
from orchestrator.infra.connectors.supabase import SupabaseConnector

SQL_ROWS = {
    postgres.TABLES_SQL: [{"name": "users", "rls_enabled": True}],
    postgres.COLUMNS_SQL: [
        {"table_name": "users", "column_name": "id", "data_type": "uuid", "is_nullable": "NO"}],
    postgres.INDEXES_SQL: [],
    postgres.FOREIGN_KEYS_SQL: [],
    postgres.POLICIES_SQL: [],
}


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    def get(self, url, headers=None, timeout=None):
        if url.endswith("/config/auth"):
            return FakeResponse({"site_url": "https://x.dev",
                                 "external_github_enabled": True,
                                 "external_google_enabled": False})
        if url.endswith("/storage/buckets"):
            return FakeResponse([{"name": "avatars", "public": False}])
        if url.endswith("/functions"):
            return FakeResponse([{"slug": "resize", "name": "resize", "status": "ACTIVE"}])
        if url.endswith("/secrets"):
            return FakeResponse([{"name": "STRIPE_KEY", "value": "sk_live_LEAKME"}])
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url, json=None, headers=None, timeout=None):
        assert url.endswith("/database/query")
        return FakeResponse(SQL_ROWS[json["query"]])


def make_target():
    return {"name": "dev", "connector": "supabase", "projectRef": "abc123",
            "tokenEnv": "SUPABASE_ACCESS_TOKEN", "risk": "dev"}


def test_fetch_state_builds_all_sections(monkeypatch):
    monkeypatch.setenv("SUPABASE_ACCESS_TOKEN", "sbp_test")
    doc = SupabaseConnector(session=FakeSession()).fetch_state(make_target())
    s = doc["sections"]
    assert s["auth"]["providers"] == ["github"]
    assert s["storage"]["buckets"] == [{"name": "avatars", "public": False}]
    assert s["functions"]["list"] == [{"name": "resize", "status": "ACTIVE"}]
    assert s["rls"]["tables"][0] == {"table": "users", "enabled": True, "policies": []}
    assert s["config"]["envNames"] == ["STRIPE_KEY"]


def test_secret_values_never_in_doc(monkeypatch):
    monkeypatch.setenv("SUPABASE_ACCESS_TOKEN", "sbp_test")
    doc = SupabaseConnector(session=FakeSession()).fetch_state(make_target())
    assert "sk_live_LEAKME" not in json.dumps(doc)


def test_supabase_registered():
    assert get_connector("supabase").id == "supabase"
