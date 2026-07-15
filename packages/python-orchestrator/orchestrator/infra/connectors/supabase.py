"""Supabase connector: Management API for platform state, SQL endpoint for schema/RLS.

One credential (personal access token, read-scoped recommended). Secret values
from /secrets are hashed immediately via redact_values — they never enter the doc.
"""
from pathlib import Path
from typing import Optional

from ..state_doc import make_state_doc, redact_values
from ..targets import resolve_env
from . import base, postgres

API_BASE = "https://api.supabase.com"
TIMEOUT = 30


class SupabaseConnector:
    id = "supabase"

    def __init__(self, session=None):
        self._session = session  # injectable for tests; lazy requests.Session otherwise

    def detect(self, project_dir: str) -> dict:
        found = (Path(project_dir) / "supabase" / "config.toml").exists()
        return {"detected": found, "connector": self.id}

    def fetch_state(self, target: dict) -> dict:
        session = self._session
        if session is None:
            import requests
            session = requests.Session()

        ref = target["projectRef"]
        headers = {"Authorization": f"Bearer {resolve_env(target, 'tokenEnv')}"}

        def get(path):
            resp = session.get(f"{API_BASE}{path}", headers=headers, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()

        def run_sql(sql):
            resp = session.post(f"{API_BASE}/v1/projects/{ref}/database/query",
                                json={"query": sql}, headers=headers, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()

        sections = postgres.introspect(run_sql)

        auth_cfg = get(f"/v1/projects/{ref}/config/auth")
        providers = sorted(
            k[len("external_"):-len("_enabled")]
            for k, v in auth_cfg.items()
            if k.startswith("external_") and k.endswith("_enabled") and v
        )
        sections["auth"] = {"providers": providers,
                            "settings": {"site_url": auth_cfg.get("site_url")}}

        buckets = get(f"/v1/projects/{ref}/storage/buckets")
        sections["storage"] = {"buckets": [
            {"name": b.get("name"), "public": bool(b.get("public"))} for b in buckets]}

        functions = get(f"/v1/projects/{ref}/functions")
        sections["functions"] = {"list": [
            {"name": f.get("slug") or f.get("name"), "status": f.get("status")}
            for f in functions]}

        secrets = get(f"/v1/projects/{ref}/secrets")
        sections["config"] = redact_values(
            {s["name"]: s.get("value", "") for s in secrets})

        return make_state_doc("supabase", target["name"], sections)


base.register(SupabaseConnector())
