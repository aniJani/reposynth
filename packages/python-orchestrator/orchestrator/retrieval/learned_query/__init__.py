"""
Learned Query Pooling Module for CCE Research.

This module implements a learned cross-attention mechanism to replace
the heuristic-based TopicInferrer with trainable query vectors that
attend over confused token embeddings.

Phase 3, Week 10-11: Learned Query Pooling + Hybrid Scoring

Components:
- LearnedQueryPooler: Cross-attention pooling with learnable queries
- ConfusedTokenEncoder: Encodes confused tokens from CCE spikes
- ContextEncoder: Encodes query and generation context
- LearnedFileScorer: Scores files using learned embeddings (requires training)
- HybridFileScorer: Combines semantic + keyword scoring (zero-shot, recommended)
- ContentAwareFileCache: Embeds files using content, not just paths
- LearnedQueryModule: Main orchestration class
- QueryPoolerLoss: Training loss (ranking + contrastive)

Recommended Usage (Zero-Shot):
    >>> from orchestrator.retrieval.learned_query import create_learned_query_module
    >>> module = create_learned_query_module()  # Uses hybrid scoring by default
    >>> module.set_available_files(file_paths, file_contents)
    >>> result = module(
    ...     confused_tokens=["async", "await"],
    ...     original_query="How do I use async?",
    ... )
"""

from .encoders import ConfusedTokenEncoder, ContextEncoder
from .pooler import LearnedQueryPooler
from .scorer import LearnedFileScorer, HybridFileScorer, FileEmbeddingCache, ContentAwareFileCache
from .module import LearnedQueryModule, LearnedQueryResult, create_learned_query_module
from .loss import QueryPoolerLoss
from .repo_tuning import RepoTuner, CCEDataCollector, HeuristicLabeler, setup_repo

__all__ = [
    # Encoders
    'ConfusedTokenEncoder',
    'ContextEncoder',
    # Pooler
    'LearnedQueryPooler',
    # Scorers
    'LearnedFileScorer',
    'HybridFileScorer',
    'FileEmbeddingCache',
    'ContentAwareFileCache',
    # Main module
    'LearnedQueryModule',
    'LearnedQueryResult',
    'create_learned_query_module',
    # Training
    'QueryPoolerLoss',
    # Per-repo tuning
    'RepoTuner',
    'CCEDataCollector',
    'HeuristicLabeler',
    'setup_repo',
]
