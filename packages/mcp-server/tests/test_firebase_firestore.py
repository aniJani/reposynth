import mcp_server._paths  # noqa: F401
from orchestrator.infra.connectors.firebase import fetch_collections, fetch_indexes


def _caller(routes):
    def call(method, url, params=None, json_body=None):
        for frag, resp in routes.items():
            if frag in url:
                return resp
        raise AssertionError(f"unexpected url {url}")
    return call


def test_fetch_collections_root_only_paginated():
    routes = {"listCollectionIds": {"collectionIds": ["users", "orders"]}}  # no nextPageToken
    out = fetch_collections(_caller(routes), "proj", "(default)")
    assert [c["collectionId"] for c in out["list"]] == ["orders", "users"]  # sorted output
    assert out["rootComplete"] is True
    assert out["list"][0]["subcollections"] == []


def test_fetch_collections_marks_incomplete_on_pagetoken():
    calls = {"n": 0}
    def call(method, url, params=None, json_body=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"collectionIds": ["a"], "nextPageToken": "T"}
        return {"collectionIds": ["b"]}
    out = fetch_collections(call, "proj", "(default)")
    assert [c["collectionId"] for c in out["list"]] == ["a", "b"]
    assert out["rootComplete"] is True  # fully paginated


def test_fetch_indexes_shapes():
    routes = {
        "/indexes": {"indexes": [
            {"queryScope": "COLLECTION", "state": "READY",
             "fields": [{"fieldPath": "userId", "order": "ASCENDING"}]}]},
        "/fields": {"fields": [
            {"name": "projects/p/databases/d/collectionGroups/orders/fields/notes"}]},
    }
    out = fetch_indexes(_caller(routes), "proj", "(default)")
    assert out["composite"][0]["fields"][0]["fieldPath"] == "userId"
    assert out["singleFieldOverrides"][0]["fieldPath"] == "notes"
