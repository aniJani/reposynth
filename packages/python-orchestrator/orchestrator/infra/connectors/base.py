"""Connector protocol + registry. The protocol has NO write method — by design."""
from typing import Any, Callable, Protocol


class Connector(Protocol):
    id: str

    def detect(self, project_dir: str) -> dict: ...
    def fetch_state(self, target: dict) -> dict: ...


_REGISTRY: dict = {}


def register(connector: Any) -> None:
    _REGISTRY[connector.id] = connector


def get_connector(connector_id: str) -> Any:
    if connector_id not in _REGISTRY:
        raise KeyError(f"Unknown connector '{connector_id}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[connector_id]
