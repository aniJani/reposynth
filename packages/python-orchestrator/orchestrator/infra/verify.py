"""Assertion engine over a StateDoc. Closed vocabulary; honest failure.

Unknown assertion types and assertions over a section the connector didn't
emit return 'unsupported' — never a silent pass. The vocabulary grows by
release, not by guessing.
"""


def _section(doc, name):
    return doc.get("sections", {}).get(name)


def _find(items, key, value):
    for item in items:
        if item.get(key) == value:
            return item
    return None


def _check(doc, a):
    kind = a.get("type")

    def unsupported(reason):
        return {"assertion": a, "result": "unsupported", "expected": None, "actual": reason}

    def outcome(passed, expected, actual):
        return {"assertion": a, "result": "pass" if passed else "fail",
                "expected": expected, "actual": actual}

    if kind == "table_exists":
        s = _section(doc, "schema")
        if s is None:
            return unsupported("no 'schema' section for this target")
        names = [t["name"] for t in s["tables"]]
        return outcome(a["table"] in names, a["table"], names)

    if kind == "column_matches":
        s = _section(doc, "schema")
        if s is None:
            return unsupported("no 'schema' section for this target")
        table = _find(s["tables"], "name", a["table"])
        if table is None:
            return outcome(False, a["table"], "table not found")
        col = _find(table["columns"], "name", a["column"])
        if col is None:
            return outcome(False, a["column"], [c["name"] for c in table["columns"]])
        ok = (("col_type" not in a or col["type"] == a["col_type"])
              and ("nullable" not in a or col["nullable"] == a["nullable"]))
        return outcome(ok, {k: a[k] for k in ("col_type", "nullable") if k in a}, col)

    if kind == "rls_enabled":
        s = _section(doc, "rls")
        if s is None:
            return unsupported("no 'rls' section for this target")
        table = _find(s["tables"], "table", a["table"])
        if table is None:
            return outcome(False, a["table"], "table not found")
        return outcome(bool(table["enabled"]), True, table["enabled"])

    if kind == "policy_exists":
        s = _section(doc, "rls")
        if s is None:
            return unsupported("no 'rls' section for this target")
        table = _find(s["tables"], "table", a["table"])
        if table is None:
            return outcome(False, a["table"], "table not found")
        matches = [p for p in table["policies"]
                   if ("cmd" not in a or p["cmd"] == a["cmd"])
                   and ("role" not in a or a["role"] in p["roles"])
                   and ("name" not in a or p["name"] == a["name"])]
        return outcome(bool(matches), a, table["policies"])

    if kind == "index_exists":
        s = _section(doc, "schema")
        if s is None:
            return unsupported("no 'schema' section for this target")
        table = _find(s["tables"], "name", a["table"])
        if table is None:
            return outcome(False, a["table"], "table not found")
        return outcome(a["index"] in table["indexes"], a["index"], table["indexes"])

    if kind == "auth_provider_enabled":
        s = _section(doc, "auth")
        if s is None:
            return unsupported("no 'auth' section for this target")
        return outcome(a["provider"] in s["providers"], a["provider"], s["providers"])

    if kind == "bucket_exists":
        s = _section(doc, "storage")
        if s is None:
            return unsupported("no 'storage' section for this target")
        bucket = _find(s["buckets"], "name", a["bucket"])
        if bucket is None:
            return outcome(False, a["bucket"], [b["name"] for b in s["buckets"]])
        if "public" in a:
            return outcome(bucket["public"] == a["public"], {"public": a["public"]}, bucket)
        return outcome(True, a["bucket"], bucket)

    if kind == "function_deployed":
        s = _section(doc, "functions")
        if s is None:
            return unsupported("no 'functions' section for this target")
        fn = _find(s["list"], "name", a["function"])
        if fn is None:
            return outcome(False, a["function"], [f["name"] for f in s["list"]])
        if "status" in a:
            return outcome(fn.get("status") == a["status"], {"status": a["status"]}, fn)
        return outcome(True, a["function"], fn)

    if kind == "env_name_present":
        s = _section(doc, "config")
        if s is None:
            return unsupported("no 'config' section for this target")
        return outcome(a["env"] in s["envNames"], a["env"], s["envNames"])

    return unsupported(f"unknown assertion type '{kind}'")


def verify(state_doc: dict, assertions: list) -> dict:
    results = [_check(state_doc, a) for a in assertions]
    summary = {"pass": 0, "fail": 0, "unsupported": 0}
    for r in results:
        summary[r["result"]] += 1
    return {"results": results, "summary": summary}
