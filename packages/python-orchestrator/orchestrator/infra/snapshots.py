"""Content-addressed StateDoc snapshots in .reposynth/snapshots/."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .state_doc import section_hash
from .targets import project_dir


def _snap_dir(project: Optional[Path]) -> Path:
    d = (project or project_dir()) / ".reposynth" / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_snapshot(doc: dict, project: Optional[Path] = None, label: Optional[str] = None) -> dict:
    ts = _utc_ts()
    digest = section_hash(doc)[len("sha256:"):][:8]
    snap_id = f"{ts}-{digest}"
    if label:
        snap_id += "-" + re.sub(r"[^A-Za-z0-9_-]", "_", label)[:32]
    payload = json.dumps(doc, indent=2, default=str)
    base_id = snap_id
    path = _snap_dir(project) / f"{snap_id}.json"
    suffix = 2
    # ponytail: identical re-save is a no-op (content-addressed); a true
    # id collision with different content gets a numeric suffix, never a clobber.
    while path.exists() and path.read_text() != payload:
        snap_id = f"{base_id}-{suffix}"
        path = _snap_dir(project) / f"{snap_id}.json"
        suffix += 1
    path.write_text(payload)
    return {"id": snap_id, "path": str(path),
            "target": doc.get("target"), "capturedAt": doc.get("capturedAt")}


def list_snapshots(project: Optional[Path] = None) -> list:
    out = []
    for path in sorted(_snap_dir(project).glob("*.json")):
        doc = json.loads(path.read_text())
        out.append({"id": path.stem, "target": doc.get("target"),
                    "capturedAt": doc.get("capturedAt")})
    return out


def load_snapshot(snapshot_id: str, project: Optional[Path] = None) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", snapshot_id or ""):
        raise KeyError(f"Invalid snapshot id '{snapshot_id}'")
    path = _snap_dir(project) / f"{snapshot_id}.json"
    if not path.exists():
        available = [p.stem for p in _snap_dir(project).glob("*.json")]
        raise KeyError(f"Snapshot '{snapshot_id}' not found. Available: {available}")
    return json.loads(path.read_text())
