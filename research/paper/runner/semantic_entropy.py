"""Semantic entropy computation, following Kuhn et al. 2023 / Farquhar et al. 2024.

We sample N generations per question, cluster them by semantic equivalence,
and compute the Shannon entropy over cluster sizes. High semantic entropy
means the model produces semantically distinct answers under sampling -
a strong signal that it's uncertain about the answer.

Original methods use NLI for clustering; we use sentence-embedding cosine
similarity above a threshold as a tractable approximation. This is the
recipe used in much follow-up work (e.g., the FLARE-plus-SemanticEntropy
literature) and is cheap enough to fit in a Colab session.

Paper note: "We follow Kuhn et al. (2023) and Farquhar et al. (2024) and use
embedding-based semantic equivalence clustering as a compute-efficient
approximation; we report results at sim_threshold=0.85 with N=5 samples."
"""

from __future__ import annotations

import math


def cluster_by_embedding(generations: list[str], embedder,
                          sim_threshold: float = 0.85) -> list[list[int]]:
    """Greedy cluster generations by sentence-embedding cosine similarity.

    Returns clusters as lists of indices into `generations`. Greedy assignment:
    a generation joins the first cluster whose representative embedding has
    cosine similarity >= sim_threshold; otherwise it starts a new cluster.
    """
    if not generations:
        return []
    import torch
    embs = embedder.encode(generations, normalize_embeddings=True,
                            convert_to_tensor=True)

    clusters: list[list[int]] = []
    cluster_reps: list = []
    for i in range(len(generations)):
        e = embs[i]
        assigned = False
        for k, rep in enumerate(cluster_reps):
            if float((e * rep).sum()) >= sim_threshold:
                clusters[k].append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])
            cluster_reps.append(e)
    return clusters


def semantic_entropy(generations: list[str], embedder,
                      sim_threshold: float = 0.85) -> dict[str, float | int]:
    """Compute semantic entropy + auxiliary stats.

    Returns:
      {
        "semantic_entropy": float,    # H over cluster size distribution (nats)
        "semantic_entropy_norm": float, # normalized to [0, 1] by log(N)
        "n_clusters": int,             # number of distinct clusters found
        "n_samples": int,              # input N
        "largest_cluster_frac": float  # mass of mode cluster
      }
    """
    n = len(generations)
    if n == 0:
        return {"semantic_entropy": 0.0, "semantic_entropy_norm": 0.0,
                "n_clusters": 0, "n_samples": 0, "largest_cluster_frac": 0.0}

    clusters = cluster_by_embedding(generations, embedder,
                                     sim_threshold=sim_threshold)
    sizes = [len(c) for c in clusters]
    probs = [s / n for s in sizes]
    h = -sum(p * math.log(p + 1e-12) for p in probs)
    norm = h / math.log(n) if n > 1 else 0.0
    return {
        "semantic_entropy": float(h),
        "semantic_entropy_norm": float(norm),
        "n_clusters": int(len(clusters)),
        "n_samples": int(n),
        "largest_cluster_frac": float(max(sizes) / n),
    }


SE_FEATURE_KEYS = (
    "semantic_entropy",
    "semantic_entropy_norm",
    "n_clusters",
    "largest_cluster_frac",
)
