"""V6 + CCE-as-feature: the paper-novelty integration.

V6 currently extracts 11 features per generation, all variable-named "cce_*"
but actually plain Shannon entropy over the full vocab. This module ADDS
true Contrastive Code Entropy features (H_code - H_lang) on top of V6's
existing feature set, so the classifier sees both:

  - generic LLM-internals features (SAPLMA-style) — already in V6
  - code-specific contrastive-entropy features (CCE) — NEW, what makes
    the paper non-trivially differentiated against SAPLMA / FLARE

The novelty claim becomes ablative: "for code Q&A error detection, CCE
features carry orthogonal signal beyond generic hidden-state probes; an
ablation that drops CCE features degrades F1 by X points."

Wiring point:
    V6 cell 11's `generate_with_features` collects per-step logits/hidden
    states/attentions in three Python lists. After the loop, it builds an
    11-key features dict. To integrate this module: in the loop, also call
    CCEFeatureExtractor.observe(logits); after the loop, merge
    extractor.aggregate() into the features dict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---- token partitions (lifted from research/add_cce_adaptive_features.py) ----

CODE_KEYWORDS = {
    # control flow
    "if", "else", "elif", "for", "while", "break", "continue", "pass",
    "return", "yield", "raise", "try", "except", "finally", "with", "as",
    # definition
    "def", "class", "lambda", "async", "await",
    # imports / namespacing
    "import", "from", "as", "global", "nonlocal",
    # operators / boolean / type
    "and", "or", "not", "is", "in", "True", "False", "None",
    # common builtins / dunder triggers
    "self", "cls", "__init__", "__call__", "__name__", "__main__",
    # python types / constants frequently uncertain in code
    "int", "str", "float", "bool", "list", "dict", "tuple", "set",
    "Any", "Optional", "Union", "List", "Dict", "Tuple", "Callable",
}


LANGUAGE_WORDS = {
    "what", "how", "why", "when", "where", "which", "who", "whom",
    "the", "a", "an", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being",
    "explain", "show", "tell", "help", "describe", "implement", "use",
    "should", "would", "could", "can", "may", "might", "must",
    "to", "of", "for", "with", "without", "by", "on", "in", "at", "from",
    "and", "or", "but", "however", "therefore", "because",
    "i", "you", "we", "they", "it", "your", "our", "their", "my",
}


def build_partition_indices(tokenizer, partition: str = "lenient"):
    """Walk the tokenizer's full vocabulary, classify each token id as
    CODE / LANGUAGE / UNKNOWN by stripping the BPE prefix and matching
    against the keyword sets.

    Returns (code_ids, lang_ids) as int arrays.

    `partition`:
      - "strict":  exact lowercase match only
      - "lenient": also accept stem-prefix matches; allow camelcase splits
    """
    code_ids = []
    lang_ids = []
    vocab = tokenizer.get_vocab()
    code_set_lower = {k.lower() for k in CODE_KEYWORDS}
    lang_set_lower = {w.lower() for w in LANGUAGE_WORDS}

    for token, idx in vocab.items():
        clean = token.lstrip("Ġ▁ #").lower()
        if not clean:
            continue
        if clean in code_set_lower or token in CODE_KEYWORDS:
            code_ids.append(idx)
            continue
        if clean in lang_set_lower or token in LANGUAGE_WORDS:
            lang_ids.append(idx)
            continue
        if partition == "lenient":
            head = clean.split("_", 1)[0]
            if head in code_set_lower:
                code_ids.append(idx)
            elif head in lang_set_lower:
                lang_ids.append(idx)
    return np.asarray(code_ids, dtype=np.int64), np.asarray(lang_ids, dtype=np.int64)


# ---- per-token CCE computation -----------------------------------------

def cce_step(logits, code_ids: np.ndarray, lang_ids: np.ndarray) -> tuple[float, float, float]:
    """Compute (cce, h_code, h_lang) at a single generation step.

    `logits` is a 1D tensor of shape [vocab_size] (the model's next-token
    logits at the current position). Returns native floats.
    """
    import torch
    import torch.nn.functional as F

    probs = F.softmax(logits.float(), dim=-1)
    code_p = probs[torch.as_tensor(code_ids, device=probs.device)]
    lang_p = probs[torch.as_tensor(lang_ids, device=probs.device)]

    h_code = _entropy_bits(code_p) if code_p.numel() else 0.0
    h_lang = _entropy_bits(lang_p) if lang_p.numel() else 0.0
    return h_code - h_lang, h_code, h_lang


def _entropy_bits(p) -> float:
    import torch

    s = p.sum()
    if s.item() <= 0:
        return 0.0
    p = p / s
    return float(-(p * (p.clamp_min(1e-12)).log2()).sum().item())


# ---- aggregator: collect per-step CCE values, emit features dict --------

@dataclass
class CCEFeatureExtractor:
    code_ids: np.ndarray
    lang_ids: np.ndarray
    cce_values: list[float] = field(default_factory=list)
    h_code_values: list[float] = field(default_factory=list)
    h_lang_values: list[float] = field(default_factory=list)

    def observe(self, logits) -> None:
        cce, hc, hl = cce_step(logits, self.code_ids, self.lang_ids)
        self.cce_values.append(cce)
        self.h_code_values.append(hc)
        self.h_lang_values.append(hl)

    def aggregate(self) -> dict[str, float]:
        if not self.cce_values:
            return {k: 0.0 for k in _CCE_FEATURE_KEYS}
        cce = np.asarray(self.cce_values)
        hc = np.asarray(self.h_code_values)
        hl = np.asarray(self.h_lang_values)
        spike_thresh = cce.mean() + 2 * cce.std() if cce.std() > 0 else math.inf
        return {
            "real_cce_mean": float(cce.mean()),
            "real_cce_max": float(cce.max()),
            "real_cce_std": float(cce.std()),
            "real_cce_spikes": int((cce > spike_thresh).sum()),
            "h_code_mean": float(hc.mean()),
            "h_code_max": float(hc.max()),
            "h_lang_mean": float(hl.mean()),
            "h_lang_max": float(hl.max()),
        }


_CCE_FEATURE_KEYS = (
    "real_cce_mean", "real_cce_max", "real_cce_std", "real_cce_spikes",
    "h_code_mean", "h_code_max", "h_lang_mean", "h_lang_max",
)


# ---- ablation: feature subsets the paper compares ---------------------

V6_GENERIC_FEATURES = (
    "hs_mean_norm", "hs_std_norm", "hs_max_norm",
    "cce_mean", "cce_max", "cce_std", "cce_spikes",   # plain Shannon, kept for backward-compat
    "attn_mean", "attn_max", "attn_std",
    "response_length",
)


CCE_NEW_FEATURES = _CCE_FEATURE_KEYS


ALL_FEATURES = V6_GENERIC_FEATURES + CCE_NEW_FEATURES


# Paper's ablation arms — each is a feature-name subset that defines a
# classifier variant. Run all six on the same training data and the same
# evaluation set to populate the ablation table.
ABLATION_ARMS = {
    "saplma_only":      ("hs_mean_norm", "hs_std_norm", "hs_max_norm"),
    "flare_only":       ("cce_mean", "cce_max", "cce_std", "cce_spikes"),  # plain entropy, FLARE-style
    "v6_full":          V6_GENERIC_FEATURES,
    "cce_only":         CCE_NEW_FEATURES,
    "v6_plus_cce":      ALL_FEATURES,                                       # the paper's headline
    "drop_cce":         V6_GENERIC_FEATURES,                                # control = v6_full
}
