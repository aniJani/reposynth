"""Infra tool implementations (plain functions, MCP-agnostic → unit-testable).

Every response carries target + risk tier. Errors return {"error": ...} —
nothing raises through the MCP boundary.
"""
from typing import Optional

from . import _paths  # noqa: F401  (side effect: orchestrator importable)

from orchestrator.infra import differ, snapshots
from orchestrator.infra.connectors.base import get_connector
from orchestrator.infra.connectors import postgres, supabase  # noqa: F401  (register)
from orchestrator.infra.impact import impact as run_impact
from orchestrator.infra.targets import get_target
from orchestrator.infra.verify import verify as run_verify

_RISK_ORDER = {"prod": 0, "staging": 1, "dev": 2}


def _fetch(target_name: str):
    target = get_target(target_name)
    connector = get_connector(target["connector"])
    return target, connector.fetch_state(target)


def infra_state(target: str, section: Optional[str] = None) -> dict:
    try:
        t, doc = _fetch(target)
    except Exception as exc:
        return {"error": str(exc)}
    state = doc if section is None else doc["sections"].get(section)
    if state is None:
        return {"error": f"Target '{target}' has no section '{section}'.",
                "target": target, "risk": t["risk"]}
    return {"target": target, "risk": t["risk"], "state": state}


def infra_verify(target: str, assertions: list) -> dict:
    try:
        t, doc = _fetch(target)
    except Exception as exc:
        return {"error": str(exc)}
    return {"target": target, "risk": t["risk"], **run_verify(doc, assertions)}


def infra_impact(target: str, op: dict) -> dict:
    try:
        t, doc = _fetch(target)
    except Exception as exc:
        return {"error": str(exc)}
    return {"target": target, **run_impact(doc, op, risk=t["risk"])}


def infra_snapshot(target: str, label: Optional[str] = None) -> dict:
    try:
        t, doc = _fetch(target)
    except Exception as exc:
        return {"error": str(exc)}
    return {"target": target, "risk": t["risk"],
            "snapshot": snapshots.save_snapshot(doc, label=label)}


def _resolve_ref(ref: str):
    """Returns (doc, risk_or_None)."""
    if ref.startswith("live:"):
        t, doc = _fetch(ref[len("live:"):])
        return doc, t["risk"]
    return snapshots.load_snapshot(ref), None


def infra_drift(ref_a: str, ref_b: str) -> dict:
    try:
        doc_a, risk_a = _resolve_ref(ref_a)
        doc_b, risk_b = _resolve_ref(ref_b)
    except Exception as exc:
        return {"error": str(exc)}
    live_risks = [r for r in (risk_a, risk_b) if r]
    risk = min(live_risks, key=lambda r: _RISK_ORDER[r]) if live_risks else None
    return {"refA": ref_a, "refB": ref_b, "risk": risk, "diff": differ.diff(doc_a, doc_b)}
