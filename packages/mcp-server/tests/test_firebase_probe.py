import mcp_server._paths  # noqa: F401
import pytest
from orchestrator.infra.connectors.firebase import probe_readonly, WRITE_PERMS


def test_probe_raises_when_write_granted():
    def call(method, url, params=None, json_body=None):
        assert "testIamPermissions" in url
        return {"permissions": ["datastore.entities.delete"]}  # a write perm is held
    with pytest.raises(RuntimeError, match="write permissions"):
        probe_readonly(call, "proj")


def test_probe_passes_when_no_write_granted():
    def call(method, url, params=None, json_body=None):
        return {}  # none of the queried write perms granted
    probe_readonly(call, "proj")  # must not raise


def test_probe_proceeds_when_call_errors():
    def call(method, url, params=None, json_body=None):
        raise RuntimeError("API disabled")
    probe_readonly(call, "proj")  # warn-and-proceed: must not raise


def test_write_perms_are_all_mutating():
    assert all(any(v in p for v in ("create", "update", "delete")) for p in WRITE_PERMS)
