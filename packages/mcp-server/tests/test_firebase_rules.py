import mcp_server._paths  # noqa: F401
from orchestrator.infra.connectors.firebase import fetch_rules


def test_fetch_rules_resolves_release_to_source():
    def call(method, url, params=None, json_body=None):
        if url.endswith("/releases"):
            return {"releases": [
                {"name": "projects/p/releases/cloud.firestore",
                 "rulesetName": "projects/p/rulesets/r1"},
                {"name": "projects/p/releases/firebase.storage/p.appspot.com",
                 "rulesetName": "projects/p/rulesets/r2"}]}
        if url.endswith("/rulesets/r1"):
            return {"source": {"files": [{"name": "firestore.rules", "content": "FS_RULES"}]}}
        if url.endswith("/rulesets/r2"):
            return {"source": {"files": [{"name": "storage.rules", "content": "ST_RULES"}]}}
        raise AssertionError(url)
    out = fetch_rules(call, "p")
    by_svc = {s["service"]: s for s in out["services"]}
    assert by_svc["cloud.firestore"]["content"] == "FS_RULES"
    assert by_svc["firebase.storage"]["scope"] == "p.appspot.com"
    assert by_svc["cloud.firestore"]["contentSha256"].startswith("sha256:")


def test_fetch_rules_empty_when_no_matching_release():
    def call(method, url, params=None, json_body=None):
        return {"releases": [{"name": "projects/p/releases/other", "rulesetName": "x"}]}
    assert fetch_rules(call, "p")["services"] == []
