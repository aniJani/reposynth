import mcp_server._paths  # noqa: F401
import json
import pytest
from orchestrator.infra.connectors.base import get_connector
from orchestrator.infra.connectors import firebase  # noqa: F401  (register)


def _full_caller():
    def call(method, url, params=None, json_body=None):
        if "testIamPermissions" in url:
            return {}  # read-only
        if url.rstrip("/").endswith("/databases/(default)"):
            return {"type": "FIRESTORE_NATIVE"}
        if "listCollectionIds" in url:
            return {"collectionIds": ["users"]}
        if "/indexes" in url:
            return {"indexes": []}
        if "/fields" in url:
            return {"fields": []}
        if url.endswith("/releases"):
            return {"releases": []}
        if url.endswith("/config"):
            return {"subtype": "FIREBASE_AUTH", "signIn": {}}
        if url.endswith("/defaultSupportedIdpConfigs"):
            return {"defaultSupportedIdpConfigs": []}
        if "firebasestorage" in url:
            return {"buckets": []}
        if "/functions" in url:
            return {"functions": [], "unreachable": []}
        raise AssertionError(url)
    return call


def test_connector_registered():
    assert get_connector("firebase").id == "firebase"


def test_fetch_state_full_doc_with_injected_call():
    conn = get_connector("firebase")
    doc = conn.fetch_state({"name": "dev", "projectId": "p"}, call=_full_caller())
    assert doc["connector"] == "firebase"
    for s in ("collections", "indexes", "rules", "auth", "storage", "functions"):
        assert s in doc["sections"]
    assert doc["sections"]["collections"]["list"][0]["collectionId"] == "users"


def test_datastore_mode_raises():
    def call(method, url, params=None, json_body=None):
        if "testIamPermissions" in url:
            return {}
        if "/databases/(default)" in url:
            return {"type": "DATASTORE_MODE"}
        raise AssertionError(url)
    conn = get_connector("firebase")
    with pytest.raises(RuntimeError, match="Datastore mode"):
        conn.fetch_state({"name": "dev", "projectId": "p"}, call=call)


def test_probe_write_perm_blocks_fetch():
    def call(method, url, params=None, json_body=None):
        if "testIamPermissions" in url:
            return {"permissions": ["datastore.entities.delete"]}
        raise AssertionError("should not read past the probe")
    conn = get_connector("firebase")
    with pytest.raises(RuntimeError, match="write permissions"):
        conn.fetch_state({"name": "dev", "projectId": "p"}, call=call)
