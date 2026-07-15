import json

import pytest

import mcp_server._paths  # noqa: F401
from orchestrator.infra.state_doc import make_state_doc, redact_values, section_hash


def test_section_hash_stable_under_key_order():
    assert section_hash({"a": 1, "b": 2}) == section_hash({"b": 2, "a": 1})
    assert section_hash({"a": 1}) != section_hash({"a": 2})
    assert section_hash({"a": 1}).startswith("sha256:")


def test_redact_values_never_leaks_raw_value():
    secret = "sk_live_SUPERSECRET123"
    payload = redact_values({"STRIPE_KEY": secret})
    assert secret not in json.dumps(payload)
    assert payload["envNames"] == ["STRIPE_KEY"]
    assert payload["valueHashes"]["STRIPE_KEY"].startswith("sha256:")


def test_redacted_hash_detects_rotation():
    a = redact_values({"KEY": "value-1"})
    b = redact_values({"KEY": "value-2"})
    assert a["valueHashes"]["KEY"] != b["valueHashes"]["KEY"]


def test_make_state_doc_shape_and_hashes():
    doc = make_state_doc("postgres", "prod", {"schema": {"tables": []}})
    assert doc["connector"] == "postgres"
    assert doc["target"] == "prod"
    assert doc["capturedAt"]
    assert doc["sections"]["schema"]["hash"].startswith("sha256:")
    assert doc["sections"]["schema"]["tables"] == []


def test_make_state_doc_rejects_unknown_section():
    with pytest.raises(ValueError):
        make_state_doc("postgres", "prod", {"bogus": {}})


def test_make_state_doc_rejects_non_dict_payload():
    with pytest.raises(ValueError):
        make_state_doc("postgres", "prod", {"schema": [("tables", [])]})
