import os
import mcp_server._paths  # noqa: F401
import pytest
from orchestrator.infra.connectors import firebase  # noqa: F401
from orchestrator.infra.connectors.base import get_connector

_PROJECT = os.environ.get("FIREBASE_IT_PROJECT")
_SA = os.environ.get("FIREBASE_IT_SA_PATH")

pytestmark = pytest.mark.skipif(
    not (_PROJECT and _SA),
    reason="set FIREBASE_IT_PROJECT and FIREBASE_IT_SA_PATH to run the live Firebase test")


def test_live_fetch_state_shapes():
    conn = get_connector("firebase")
    doc = conn.fetch_state({"name": "it", "projectId": _PROJECT, "credentialsEnv": "FIREBASE_IT_SA_PATH"})
    assert doc["connector"] == "firebase"
    for s in ("collections", "indexes", "rules", "auth", "storage", "functions"):
        assert s in doc["sections"]
    # verify the implementation-verify items resolved against the real API:
    assert isinstance(doc["sections"]["collections"]["list"], list)
    assert isinstance(doc["sections"]["functions"]["unreachable"], list)
