import mcp_server._paths  # noqa: F401
from orchestrator.infra.connectors.firebase import fetch_storage


def _storage_caller(bucket_meta, bucket_iam, buckets=("p.appspot.com",)):
    def call(method, url, params=None, json_body=None):
        if "firebasestorage" in url:
            return {"buckets": [{"name": f"projects/p/buckets/{b}"} for b in buckets]}
        if url.endswith("/iam"):
            return bucket_iam
        return bucket_meta  # storage.googleapis.com/.../b/{bucket}
    return call


def test_public_via_iam_allusers():
    call = _storage_caller(
        {"iamConfiguration": {"publicAccessPrevention": "inherited"}},
        {"bindings": [{"role": "roles/storage.objectViewer", "members": ["allUsers"]}]})
    b = fetch_storage(call, "p")["buckets"][0]
    assert b["publicViaIAM"] is True and b["public"] is True


def test_pap_enforced_is_definitely_private():
    call = _storage_caller(
        {"iamConfiguration": {"publicAccessPrevention": "enforced"}},
        {"bindings": [{"role": "roles/storage.objectViewer", "members": ["allUsers"]}]})
    b = fetch_storage(call, "p")["buckets"][0]
    assert b["public"] is False  # PAP wins


def test_private_iam_unknown_rules_omits_public():
    call = _storage_caller(
        {"iamConfiguration": {"publicAccessPrevention": "inherited"}},
        {"bindings": [{"role": "roles/storage.objectViewer", "members": ["user:a@b.com"]}]})
    b = fetch_storage(call, "p")["buckets"][0]
    assert b["publicViaIAM"] is False
    assert "public" not in b  # genuinely unknown, not falsely 'private'
