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


def save_snapshot(doc: dict, project: Optional[Path] = None, label: Optional[str] = None) -> dict:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = section_hash(doc)[len("sha256:"):][:8]
    snap_id = f"{ts}-{digest}"
    if label:
        snap_id += "-" + re.sub(r"[^A-Za-z0-9_-]", "_", label)[:32]
    path = _snap_dir(project) / f"{snap_id}.json"
    path.write_text(json.dumps(doc, indent=2, default=str))
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
    path = _snap_dir(project) / f"{snapshot_id}.json"
    if not path.exists():
        available = [p.stem for p in _snap_dir(project).glob("*.json")]
        raise KeyError(f"Snapshot '{snapshot_id}' not found. Available: {available}")
    return json.loads(path.read_text())
