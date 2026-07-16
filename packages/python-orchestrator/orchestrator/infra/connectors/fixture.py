"""Fixture connector: serves a recorded StateDoc from JSON. Test/benchmark only.

No network, no write path. `statePath` is a plain file path (a test artifact,
not a secret), so the env-name-only rule for real targets does not apply here.
"""
import json
from pathlib import Path

from orchestrator.infra.targets import project_dir
from orchestrator.infra.connectors import base


class FixtureConnector:
    id = "fixture"

    def detect(self, project_dir: str) -> dict:
        return {"detected": False}

    def fetch_state(self, target: dict) -> dict:
        path = Path(target["statePath"])
        if not path.is_absolute():
            path = project_dir() / path
        return json.loads(path.read_text())


base.register(FixtureConnector())
