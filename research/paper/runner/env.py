"""Determinism setup and environment fingerprinting."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path


def setup_determinism(seed: int) -> None:
    """Seed every randomness source we know about."""
    import random

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except ImportError:
        pass

    try:
        from transformers import set_seed

        set_seed(seed)
    except ImportError:
        pass


def record_environment() -> dict:
    """Return a dict describing the runtime environment for paper_results.json."""
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    for pkg in ("torch", "transformers", "numpy", "faiss", "sentence_transformers", "rank_bm25"):
        try:
            mod = __import__(pkg)
            env[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env[pkg] = None

    try:
        import torch

        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            env["cudnn"] = torch.backends.cudnn.version()
            env["gpu_model"] = torch.cuda.get_device_name(0)
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_total_mem_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        else:
            env["cuda"] = None
    except ImportError:
        env["cuda"] = None

    return env


def git_sha(path: str | Path) -> str | None:
    """Return short git SHA for the file's tree, or None if unavailable."""
    p = Path(path)
    cwd = p.parent if p.is_file() else p
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(p.name) if p.is_file() else "."],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def file_sha256(path: str | Path) -> str:
    """SHA256 of file contents — used when git SHA is unavailable."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
