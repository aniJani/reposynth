"""Import side effect: make reposynth's `orchestrator` package importable.

Mirrors worker.py's bootstrap: prepend packages/python-orchestrator to
sys.path so `from orchestrator.infra import ...` resolves.
"""
import sys
from pathlib import Path

_ORCHESTRATOR_PARENT = Path(__file__).resolve().parents[2] / "python-orchestrator"
if _ORCHESTRATOR_PARENT.is_dir() and str(_ORCHESTRATOR_PARENT) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATOR_PARENT))
