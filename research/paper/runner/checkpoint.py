"""Atomic, resumable checkpoint store for paper_results.json.

A run is a stream of per-tuple result rows, each keyed by
(method, seed, tau, partition, layers, max_hops, qid). The checkpoint
file holds metadata + the accumulated rows. Writes are atomic (tmp +
fsync + rename) so a Colab disconnect never produces a half-written
JSON.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable


def make_key(row: dict) -> tuple:
    """Stable hashable key for resume dedup."""
    return (
        row["method"],
        row["seed"],
        row.get("tau"),
        row.get("partition"),
        tuple(row["layers"]) if row.get("layers") is not None else None,
        row.get("max_hops"),
        row["qid"],
    )


class Checkpoint:
    """Resumable JSON-lines-style checkpoint, but stored as a single JSON file.

    On disk format:
        {
          "metadata": {...},     # protocol_sha, model, env, etc.
          "results": [ row, row, ... ]
        }

    Atomic write: write to <path>.tmp, fsync, rename to <path>.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {"metadata": {}, "results": []}
        self._keys: set[tuple] = set()
        self._dirty = False
        self._last_flush = 0.0
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path) as f:
                self._data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            backup = self.path.with_suffix(self.path.suffix + ".corrupt")
            os.rename(self.path, backup)
            raise RuntimeError(
                f"Checkpoint at {self.path} is corrupt (moved to {backup}): {e}"
            ) from e
        self._data.setdefault("metadata", {})
        self._data.setdefault("results", [])
        self._keys = {make_key(r) for r in self._data["results"]}

    def set_metadata(self, **fields) -> None:
        with self._lock:
            self._data["metadata"].update(fields)
            self._dirty = True
        self.flush()

    def metadata(self) -> dict:
        return dict(self._data["metadata"])

    def has(self, row_or_key) -> bool:
        key = row_or_key if isinstance(row_or_key, tuple) else make_key(row_or_key)
        return key in self._keys

    def add(self, row: dict) -> None:
        """Append a result row. Idempotent — duplicate keys are ignored."""
        key = make_key(row)
        with self._lock:
            if key in self._keys:
                return
            self._keys.add(key)
            self._data["results"].append(row)
            self._dirty = True

    def keys(self) -> Iterable[tuple]:
        return iter(self._keys)

    def n_results(self) -> int:
        return len(self._data["results"])

    def results(self) -> list[dict]:
        return list(self._data["results"])

    def flush(self) -> None:
        """Atomically persist current state to disk."""
        with self._lock:
            if not self._dirty:
                return
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with open(tmp, "w") as f:
                json.dump(self._data, f, indent=2, default=_json_default)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            self._dirty = False
            self._last_flush = time.time()

    def maybe_flush(self, every_seconds: float = 30.0) -> None:
        """Flush if more than `every_seconds` since the last persist."""
        if time.time() - self._last_flush >= every_seconds:
            self.flush()


def _json_default(o):
    """Best-effort JSON encoding for non-stdlib types (numpy, torch, paths)."""
    try:
        import numpy as np

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)
