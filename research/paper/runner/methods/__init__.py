"""Method registry. Add new methods here only after locking PROTOCOL.md §5."""

from __future__ import annotations

from .base import BaseMethod, MethodContext, Question, Result
from .baselines import (
    BM25Method,
    EmbeddingMethod,
    FullContextMethod,
    NoContextMethod,
    RandomMethod,
    ReposynthBaseMethod,
)
from .cce_methods import CCEAdaptiveMethod, CCELearnedMethod, UncertCoTMethod


METHODS: dict[str, type[BaseMethod]] = {
    "no_context": NoContextMethod,
    "random": RandomMethod,
    "bm25": BM25Method,
    "embedding": EmbeddingMethod,
    "full_context": FullContextMethod,
    "reposynth_base": ReposynthBaseMethod,
    "uncert_cot": UncertCoTMethod,
    "cce_adaptive": CCEAdaptiveMethod,
    "cce_learned": CCELearnedMethod,
}


def get(name: str) -> type[BaseMethod]:
    if name not in METHODS:
        raise KeyError(f"unknown method {name!r}. known: {sorted(METHODS)}")
    return METHODS[name]


__all__ = [
    "METHODS",
    "get",
    "BaseMethod",
    "MethodContext",
    "Question",
    "Result",
]
