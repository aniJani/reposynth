"""Env-gated integration test against a real free-tier Supabase project.

Run in CI with SUPABASE_IT_TOKEN + SUPABASE_IT_PROJECT_REF set; skipped otherwise.
"""
import os

import pytest

import mcp_server._paths  # noqa: F401
from orchestrator.infra.connectors.supabase import SupabaseConnector

pytestmark = pytest.mark.skipif(
    not (os.environ.get("SUPABASE_IT_TOKEN") and os.environ.get("SUPABASE_IT_PROJECT_REF")),
    reason="integration env not configured",
)


def test_live_fetch_state(monkeypatch):
    monkeypatch.setenv("SUPABASE_ACCESS_TOKEN", os.environ["SUPABASE_IT_TOKEN"])
    target = {"name": "it", "connector": "supabase", "risk": "dev",
              "projectRef": os.environ["SUPABASE_IT_PROJECT_REF"],
              "tokenEnv": "SUPABASE_ACCESS_TOKEN"}
    doc = SupabaseConnector().fetch_state(target)
    assert set(doc["sections"]) >= {"schema", "rls", "auth", "storage", "functions", "config"}
