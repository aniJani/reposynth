import mcp_server._paths  # noqa: F401
from orchestrator.infra.state_doc import make_state_doc, SECTION_NAMES
from orchestrator.infra.verify import verify
from orchestrator.infra.differ import diff


def test_new_sections_accepted():
    for s in ("collections", "indexes", "rules"):
        assert s in SECTION_NAMES
    doc = make_state_doc("firebase", "dev", {
        "collections": {"list": [{"collectionId": "users"}], "rootComplete": True}})
    assert "collections" in doc["sections"]


def test_collection_exists_pass_and_fail_with_caveat():
    doc = make_state_doc("firebase", "dev",
        {"collections": {"list": [{"collectionId": "users"}], "rootComplete": True}})
    ok = verify(doc, [{"type": "collection_exists", "collection": "users"}])
    assert ok["results"][0]["result"] == "pass"
    bad = verify(doc, [{"type": "collection_exists", "collection": "orders"}])
    assert bad["results"][0]["result"] == "fail"
    assert "ephemeral" in str(bad["results"][0]["actual"]).lower()


def test_collection_exists_unsupported_without_section():
    doc = make_state_doc("postgres", "dev", {"schema": {"tables": []}})
    r = verify(doc, [{"type": "collection_exists", "collection": "x"}])
    assert r["results"][0]["result"] == "unsupported"


def test_differ_collections_added_removed():
    a = make_state_doc("firebase", "dev",
        {"collections": {"list": [{"collectionId": "users"}], "rootComplete": True}})
    b = make_state_doc("firebase", "dev",
        {"collections": {"list": [{"collectionId": "users"}, {"collectionId": "orders"}],
                         "rootComplete": True}})
    d = diff(a, b)
    assert d["sections"]["collections"]["added"] == ["orders"]


def test_differ_rules_hash_drift():
    a = make_state_doc("firebase", "dev", {"rules": {"services": [
        {"service": "cloud.firestore", "content": "A", "contentSha256": "sha256:a"}]}})
    b = make_state_doc("firebase", "dev", {"rules": {"services": [
        {"service": "cloud.firestore", "content": "B", "contentSha256": "sha256:b"}]}})
    d = diff(a, b)
    assert "cloud.firestore" in d["sections"]["rules"]["changed"]
