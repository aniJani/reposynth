import mcp_server._paths  # noqa: F401
from orchestrator.infra.connectors.firebase import fetch_functions


def test_fetch_functions_parses_name_region_status():
    def call(method, url, params=None, json_body=None):
        return {"functions": [
            {"name": "projects/p/locations/us-central1/functions/sendEmail",
             "state": "ACTIVE", "environment": "GEN_2",
             "serviceConfig": {"uri": "https://x"}}],
            "unreachable": ["europe-west1"]}
    out = fetch_functions(call, "p")
    fn = out["list"][0]
    assert fn["name"] == "sendEmail" and fn["region"] == "us-central1"
    assert fn["status"] == "ACTIVE" and fn["generation"] == "gen2"
    assert out["unreachable"] == ["europe-west1"]
