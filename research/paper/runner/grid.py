"""Configuration grid expansion — turns the protocol matrix into concrete tuples.

A "tuple" is a fully-specified experiment row:
    {method, seed, tau, partition, layers, max_hops}
plus the question id (added downstream during the run).

Modes:
- headline: every method × every seed at the headline config
              (tau=0.5, partition=lenient, layers=[16,24,31], max_hops=3)
- sensitivity: cce_* methods only, full sweep over tau/partition/layers/max_hops
- all: headline + sensitivity
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from itertools import product


HEADLINE_CONFIG = {
    "tau": 0.5,
    "partition": "lenient",
    "layers": (16, 24, 31),
    "max_hops": 3,
}


SENSITIVITY_AXES = {
    "tau": [0.3, 0.5, 0.7],
    "partition": ["strict", "lenient"],
    "layers": [(31,), (16, 24, 31), "all"],
    "max_hops": [1, 3, 5],
}


METHOD_NAMES = [
    "no_context",
    "random",
    "bm25",
    "embedding",
    "full_context",
    "reposynth_base",
    "uncert_cot",
    "cce_adaptive",
    "cce_learned",
]


CCE_METHOD_NAMES = ["cce_adaptive", "cce_learned"]


@dataclass(frozen=True)
class Config:
    method: str
    seed: int
    tau: float | None
    partition: str | None
    layers: tuple | None
    max_hops: int | None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["layers"] = list(self.layers) if self.layers is not None else None
        return d


def headline_grid(methods: list[str], seeds: list[int]) -> list[Config]:
    """Each method × each seed at the single headline config."""
    out = []
    for method, seed in product(methods, seeds):
        is_cce = method in CCE_METHOD_NAMES
        out.append(
            Config(
                method=method,
                seed=seed,
                tau=HEADLINE_CONFIG["tau"] if is_cce else None,
                partition=HEADLINE_CONFIG["partition"] if is_cce else None,
                layers=HEADLINE_CONFIG["layers"] if is_cce else None,
                max_hops=HEADLINE_CONFIG["max_hops"] if is_cce else None,
            )
        )
    return out


def sensitivity_grid(seeds: list[int], factorial: bool = False) -> list[Config]:
    """CCE-only sensitivity sweep.

    Default: cross-sections. For each axis, vary that axis only and hold the
    other three at their headline values. Reveals each axis's marginal effect
    in O(sum) configs instead of O(product).

    factorial=True: full Cartesian product. Not recommended (54x more configs
    per method-seed, generally not informative enough to justify the cost).
    """
    if factorial:
        out = []
        for method, seed, tau, partition, layers, hops in product(
            CCE_METHOD_NAMES,
            seeds,
            SENSITIVITY_AXES["tau"],
            SENSITIVITY_AXES["partition"],
            SENSITIVITY_AXES["layers"],
            SENSITIVITY_AXES["max_hops"],
        ):
            layers_tuple = tuple(layers) if layers != "all" else "all"
            out.append(
                Config(
                    method=method, seed=seed, tau=tau, partition=partition,
                    layers=layers_tuple, max_hops=hops,
                )
            )
        return out

    h_tau = HEADLINE_CONFIG["tau"]
    h_part = HEADLINE_CONFIG["partition"]
    h_layers = HEADLINE_CONFIG["layers"]
    h_hops = HEADLINE_CONFIG["max_hops"]

    out: list[Config] = []
    for method, seed in product(CCE_METHOD_NAMES, seeds):
        for tau in SENSITIVITY_AXES["tau"]:
            if tau == h_tau:
                continue
            out.append(Config(method, seed, tau, h_part, h_layers, h_hops))
        for part in SENSITIVITY_AXES["partition"]:
            if part == h_part:
                continue
            out.append(Config(method, seed, h_tau, part, h_layers, h_hops))
        for layers in SENSITIVITY_AXES["layers"]:
            layers_t = tuple(layers) if layers != "all" else "all"
            if layers_t == h_layers:
                continue
            out.append(Config(method, seed, h_tau, h_part, layers_t, h_hops))
        for hops in SENSITIVITY_AXES["max_hops"]:
            if hops == h_hops:
                continue
            out.append(Config(method, seed, h_tau, h_part, h_layers, hops))
    return out


def expand(mode: str, methods: list[str], seeds: list[int], factorial: bool = False) -> list[Config]:
    if mode == "headline":
        return headline_grid(methods, seeds)
    if mode == "sensitivity":
        return sensitivity_grid(seeds, factorial=factorial)
    if mode == "all":
        return headline_grid(methods, seeds) + sensitivity_grid(seeds, factorial=factorial)
    raise ValueError(f"unknown grid mode: {mode!r}")


def estimate_cost(
    configs: list[Config],
    n_questions: int,
    sec_per_generation: float = 10.0,
    judge_call_cost_usd: float = 0.000135,
    judge_unique_ratio: float = 0.4,
) -> dict:
    """Rough back-of-envelope wall-clock and judge-API cost estimate.

    judge_unique_ratio: fraction of (config, question) tuples that produce a
    judge cache miss. ~0.4 is plausible because greedy-decoded generations are
    often identical across seeds for static-pre-retrieval methods.
    """
    n_tuples = len(configs) * n_questions
    wall_seconds = n_tuples * sec_per_generation
    judge_calls = int(n_tuples * judge_unique_ratio)
    judge_usd = judge_calls * judge_call_cost_usd
    return {
        "n_configs": len(configs),
        "n_questions": n_questions,
        "n_tuples_total": n_tuples,
        "wall_hours_est": round(wall_seconds / 3600, 1),
        "judge_calls_est": judge_calls,
        "judge_usd_est": round(judge_usd, 4),
    }
