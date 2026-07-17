import mcp_server._paths  # noqa: F401
from orchestrator.infra.connectors.firebase import fetch_auth


def test_fetch_auth_builds_provider_list():
    def call(method, url, params=None, json_body=None):
        if url.endswith("/config"):
            return {"subtype": "FIREBASE_AUTH", "authorizedDomains": ["localhost"],
                    "mfa": {"state": "DISABLED"},
                    "signIn": {"email": {"enabled": True}, "phoneNumber": {"enabled": False},
                               "anonymous": {"enabled": True}}}
        if url.endswith("/defaultSupportedIdpConfigs"):
            return {"defaultSupportedIdpConfigs": [
                {"name": "projects/p/defaultSupportedIdpConfigs/google.com", "enabled": True},
                {"name": "projects/p/defaultSupportedIdpConfigs/github.com", "enabled": False}]}
        raise AssertionError(url)
    out = fetch_auth(call, "p")
    assert set(out["providers"]) == {"password", "anonymous", "google.com"}
    assert out["settings"]["tier"] == "FIREBASE_AUTH"
    assert out["settings"]["mfaState"] == "DISABLED"
