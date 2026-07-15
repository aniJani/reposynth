"""Named targets from .reposynth/targets.json — env-var NAMES only, plus risk tiers."""
import json
import os
from pathlib import Path
from typing import Optional

RISK_TIERS = ("prod", "staging", "dev")


def project_dir() -> Path:
    return Path(os.environ.get("REPOSYNTH_PROJECT_DIR", os.getcwd()))


def load_targets(project: Optional[Path] = None) -> dict:
    path = (project or project_dir()) / ".reposynth" / "targets.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    targets = data.get("targets", {})
    for name, target in targets.items():
        if "connector" not in target:
            raise ValueError(f"Target '{name}' missing 'connector'")
        risk = target.get("risk")
        if risk not in RISK_TIERS:
            raise ValueError(f"Target '{name}' has invalid risk '{risk}'. Valid: {RISK_TIERS}")
        target["name"] = name
    return targets


def get_target(name: str, project: Optional[Path] = None) -> dict:
    targets = load_targets(project)
    if name not in targets:
        raise KeyError(f"Unknown target '{name}'. Available: {sorted(targets)}")
    return targets[name]


def resolve_env(target: dict, key: str) -> str:
    env_name = target.get(key)
    if not env_name:
        raise RuntimeError(f"Target '{target.get('name')}' has no '{key}' configured")
    value = os.environ.get(env_name)
    if value is None:
        raise RuntimeError(f"Environment variable '{env_name}' is not set")
    return value
