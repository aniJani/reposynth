"""CCE-family methods: cce_adaptive, cce_learned, uncert_cot.

These methods all do mid-generation triggering. They share a generate loop
that:
  1. Generates token by token (or in chunks).
  2. Computes an uncertainty signal at trigger points.
  3. If signal > threshold, retrieves and splices new context.
  4. Continues until max_new_tokens or EOS.

The signal differs:
  - uncert_cot: raw entropy at line boundaries
  - cce_adaptive: H_code - H_lang at semantic boundaries
  - cce_learned: same trigger as cce_adaptive, but file scoring uses a
                 trained cross-attention pooler

LIFTING NOTE: working implementations of all three already exist in the
notebooks. See:
  - cce_adaptive: Week15_V6_Balanced_Tasks.ipynb (the RealCCEMultiHopRetriever
                  class) and packages/python-orchestrator/orchestrator/entropy/
  - cce_learned: Week9_V3_Learned_Query_Integration.ipynb (LearnedQueryPooler)
  - uncert_cot:  add_cce_adaptive_features.py (line-boundary trigger logic)

Until those are migrated into this module, the run() methods raise
NotImplementedError. The runner will record status="not_implemented" rows
without aborting the run, so the rest of the grid can still execute.
"""

from __future__ import annotations

import time

from .base import BaseMethod, MethodContext, Question, Result


class UncertCoTMethod(BaseMethod):
    """UnCert-CoT-style baseline: trigger retrieval at line boundaries when raw entropy spikes."""

    name = "uncert_cot"

    def run(self, q: Question) -> Result:
        r = self._empty_result(q.qid, self.config["seed"])
        r.status = "not_implemented"
        r.error = (
            "uncert_cot.run() is not yet wired. Lift implementation from "
            "research/add_cce_adaptive_features.py + tests/entropy/test_calculator.py. "
            "Trigger condition: raw Shannon entropy at every line boundary."
        )
        return r


class CCEAdaptiveMethod(BaseMethod):
    """The proposed method: H_code − H_lang at semantic boundaries, multi-hop on confused tokens."""

    name = "cce_adaptive"

    def run(self, q: Question) -> Result:
        r = self._empty_result(q.qid, self.config["seed"])
        r.status = "not_implemented"
        r.error = (
            "cce_adaptive.run() is not yet wired. Source: Week15_V6_Balanced_Tasks.ipynb "
            "— lift generate_with_features (cell 11), generate_simple (cell 12), "
            "retrieve (cell 9), create_chunks (cell 8). V6 is the version that "
            "produced the 0.95 headline result and therefore the version that must "
            "reproduce. Hyperparams live in self.config (tau, partition, layers, "
            "max_hops). Audit: V6 does not literally name H_code/H_lang variables — "
            "confirm its entropy formula matches the paper's H_code − H_lang claim "
            "before relying on it. Verify the migration by re-running on V6's exact "
            "20-task set and confirming accuracy reproduces within seed noise."
        )
        return r


class CCELearnedMethod(BaseMethod):
    """Same trigger as cce_adaptive; file scoring uses the trained cross-attention pooler."""

    name = "cce_learned"

    def run(self, q: Question) -> Result:
        r = self._empty_result(q.qid, self.config["seed"])
        r.status = "not_implemented"
        r.error = (
            "cce_learned.run() is not yet wired. Lift the LearnedQueryPooler from "
            "Week9_V3_Learned_Query_Integration.ipynb (cell 13). Per PROTOCOL §5.1: "
            "cross-repo train/val/test split, 3 retrained checkpoints (one per outer seed), "
            "best epoch by val composite, no test peeking."
        )
        return r
