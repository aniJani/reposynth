"""Blast-radius heuristic over a captured StateDoc.

ponytail: heuristic over captured state only — no live dependency graph, no
traffic analysis. Ops outside the closed set return result='unknown', never a
reassuring empty findings list. Extend the op set per release.
"""


def _section(doc, name):
    return doc.get("sections", {}).get(name)


def _analyzed(op, risk, findings):
    return {"op": op, "risk": risk, "result": "analyzed", "findings": findings}


_REQUIRED_SECTIONS = {
    "drop_table": ("schema", "rls"),
    "delete_bucket": ("storage",),
    "drop_policy": ("rls",),
    "drop_role": ("rls",),
    "delete_function": ("functions",),
}


def impact(state_doc: dict, op: dict, risk: str) -> dict:
    kind = op.get("op")
    if kind not in _REQUIRED_SECTIONS:
        return {"op": op, "risk": risk, "result": "unknown",
                "findings": [f"op '{kind}' is outside the analyzed set — no claim made"]}
    missing = [n for n in _REQUIRED_SECTIONS[kind] if _section(state_doc, n) is None]
    if missing:
        return {"op": op, "risk": risk, "result": "unknown",
                "findings": [f"section '{n}' not captured for this target — no claim made" for n in missing]}
    schema = _section(state_doc, "schema") or {"tables": []}
    rls = _section(state_doc, "rls") or {"tables": []}

    if kind == "drop_table":
        table = op["table"]
        findings = []
        for t in schema["tables"]:
            for fk in t.get("foreignKeys", []):
                if fk["toTable"] == table:
                    findings.append(
                        f"table '{t['name']}' references '{table}' via FK '{fk['name']}' — drop breaks it")
        for t in rls["tables"]:
            if t["table"] == table and t["policies"]:
                findings.append(f"{len(t['policies'])} RLS policies on '{table}' are lost")
        if not findings:
            findings.append(f"no inbound FKs or policies found for '{table}' in captured state")
        return _analyzed(op, risk, findings)

    if kind == "delete_bucket":
        storage = _section(state_doc, "storage") or {"buckets": []}
        bucket = next((b for b in storage["buckets"] if b["name"] == op["bucket"]), None)
        if bucket is None:
            return _analyzed(op, risk, [f"bucket '{op['bucket']}' not found in captured state"])
        vis = "PUBLIC" if bucket["public"] else "private"
        return _analyzed(op, risk, [f"bucket '{op['bucket']}' is {vis}; objects become unreachable"])

    if kind == "drop_policy":
        table = next((t for t in rls["tables"] if t["table"] == op["table"]), None)
        if table is None:
            return _analyzed(op, risk, [f"table '{op['table']}' not found in captured state"])
        if not any(p["name"] == op["policy"] for p in table["policies"]):
            return _analyzed(op, risk, [
                f"policy '{op['policy']}' not found on table '{op['table']}' in captured state — drop would be a no-op"])
        remaining = [p["name"] for p in table["policies"] if p["name"] != op["policy"]]
        findings = [f"{len(remaining)} remaining policies on '{op['table']}': {remaining}"]
        if table["enabled"] and not remaining:
            findings.append(
                f"'{op['table']}' has RLS enabled and no policies remain — non-owner roles lose ALL access")
        return _analyzed(op, risk, findings)

    if kind == "drop_role":
        findings = []
        for t in rls["tables"]:
            for p in t["policies"]:
                if op["role"] in p["roles"]:
                    findings.append(f"policy '{p['name']}' on '{t['table']}' references role '{op['role']}'")
        if not findings:
            findings.append(f"no policies reference role '{op['role']}' in captured state")
        return _analyzed(op, risk, findings)

    if kind == "delete_function":
        functions = _section(state_doc, "functions") or {"list": []}
        fn = next((f for f in functions["list"] if f["name"] == op["function"]), None)
        if fn is None:
            return _analyzed(op, risk, [f"function '{op['function']}' not found in captured state"])
        return _analyzed(op, risk, [f"function '{op['function']}' is {fn.get('status')}; callers will 404"])
