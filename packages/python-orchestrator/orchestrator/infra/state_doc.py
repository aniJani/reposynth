"""StateDoc: one normalized JSON doc per (connector, target, capture).

Redaction happens HERE, at capture time — raw secret values exist only
inside redact_values() and never enter a StateDoc.
"""
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional

SECTION_NAMES = ("schema", "rls", "auth", "storage", "functions", "config")


def section_hash(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def redact_values(values: dict) -> dict:
    """Build the `config` section: names + value hashes, never raw values."""
    return {
        "envNames": sorted(values.keys()),
        "valueHashes": {
            k: "sha256:" + hashlib.sha256(str(v).encode("utf-8")).hexdigest()
            for k, v in values.items()
        },
    }


def make_state_doc(
    connector: str,
    target: str,
    sections: dict,
    captured_at: Optional[str] = None,
) -> dict:
    out = {}
    for name, payload in sections.items():
        if name not in SECTION_NAMES:
            raise ValueError(f"Unknown StateDoc section '{name}'. Valid: {SECTION_NAMES}")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Section '{name}' payload must be a dict, got {type(payload).__name__}"
            )
        entry = dict(payload)
        entry["hash"] = section_hash(payload)
        out[name] = entry
    return {
        "connector": connector,
        "target": target,
        "capturedAt": captured_at or datetime.now(timezone.utc).isoformat(),
        "sections": out,
    }
