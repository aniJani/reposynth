"""
Learned Query Module - Main Orchestration Class.

This module combines all components (encoders, pooler, scorer) into
a single trainable module that produces retrieval query embeddings
and file scores from CCE spike data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from .encoders import ConfusedTokenEncoder, ContextEncoder
from .pooler import LearnedQueryPooler
from .scorer import LearnedFileScorer, FileEmbeddingCache, HybridFileScorer, ContentAwareFileCache


@dataclass
class LearnedQueryResult:
    """Result from the learned query module."""
    query_embedding: torch.Tensor           # Pooled query embedding [D]
    matched_files: List[str]                # Files matched above threshold
    file_scores: Dict[str, float]           # All file scores
    confidence: float                       # Model confidence (attention entropy)

    # Attention metadata for visualization
    token_attention: Optional[torch.Tensor] = None   # [Q, K]
    context_attention: Optional[torch.Tensor] = None  # [Q, L]

    # Input metadata
    confused_tokens: Optional[List[str]] = None
    token_weights: Optional[torch.Tensor] = None

    def __repr__(self) -> str:
        return (
            f"LearnedQueryResult(matched={self.matched_files[:3]}, "
            f"conf={self.confidence:.3f})"
        )


class LearnedQueryModule(nn.Module):
    """
    Main module combining all learned query components.

    This module takes:
    - Confused tokens from CCE spikes (with probabilities)
    - Original query and generation context
    - Available file paths

    And produces:
    - Retrieval query embedding
    - File scores and rankings
    - Attention visualizations

    Architecture:
        confused_tokens → ConfusedTokenEncoder → [K, D]
        (query, context) → ContextEncoder → [L, D]
        ↓
        LearnedQueryPooler (cross-attention) → [D]
        ↓
        LearnedFileScorer → file scores [N]

    Example:
        >>> module = LearnedQueryModule()
        >>> module.set_available_files(["src/auth.py", "src/api.py"])
        >>> result = module(
        ...     confused_tokens=["auth", "jwt", "bearer"],
        ...     confused_probs=[0.3, 0.25, 0.2],
        ...     original_query="How does authentication work?",
        ...     generated_context="In our Flask app, authentication uses"
        ... )
        >>> print(result.matched_files)  # ["src/auth.py"]
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_query_vectors: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        model_name: str = 'all-MiniLM-L6-v2',
        scoring_method: str = "hybrid",
        score_threshold: float = 0.1,
        top_k: int = 3,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.4,
        exact_match_boost: float = 0.1,
    ):
        """
        Initialize learned query module.

        Args:
            embed_dim: Embedding dimension (384 for all-MiniLM-L6-v2)
            num_query_vectors: Number of learnable queries
            num_heads: Number of attention heads
            dropout: Dropout rate
            model_name: SentenceTransformer model name
            scoring_method: File scoring method:
                - "hybrid": Combines semantic + keyword (recommended, no per-repo training)
                - "cosine": Pure semantic similarity
                - "bilinear": Learned bilinear (requires training)
                - "mlp": Learned MLP (requires training)
            score_threshold: Minimum score for file matching
            top_k: Maximum files to return
            semantic_weight: Weight for semantic scores in hybrid mode
            keyword_weight: Weight for keyword scores in hybrid mode
            exact_match_boost: Bonus for exact token matches in hybrid mode
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.score_threshold = score_threshold
        self.top_k = top_k
        self.scoring_method = scoring_method

        # Component modules
        self.token_encoder = ConfusedTokenEncoder(
            embed_dim=embed_dim,
            model_name=model_name,
            dropout=dropout,
        )

        self.context_encoder = ContextEncoder(
            embed_dim=embed_dim,
            model_name=model_name,
            dropout=dropout,
        )

        self.pooler = LearnedQueryPooler(
            embed_dim=embed_dim,
            num_query_vectors=num_query_vectors,
            num_heads=num_heads,
            dropout=dropout,
            fusion_method="concat",
        )

        # Scoring: hybrid (default) or learned
        if scoring_method == "hybrid":
            self.hybrid_scorer = HybridFileScorer(
                embed_dim=embed_dim,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                exact_match_boost=exact_match_boost,
                model_name=model_name,
            )
            self.scorer = None  # Not used in hybrid mode
            self.file_cache = None
        else:
            self.hybrid_scorer = None
            self.scorer = LearnedFileScorer(
                embed_dim=embed_dim,
                scoring_method=scoring_method,
                dropout=dropout,
            )
            self.file_cache = ContentAwareFileCache(model_name=model_name)

        self._file_paths: List[str] = []
        self._file_embeddings: Optional[torch.Tensor] = None
        self._file_contents: Optional[Dict[str, str]] = None
        self._is_fine_tuned: bool = False  # Set to True after fine-tuning

    def set_available_files(
        self,
        file_paths: List[str],
        file_contents: Optional[Dict[str, str]] = None,
    ):
        """
        Set available file paths for scoring.

        Args:
            file_paths: List of file paths in the codebase
            file_contents: Optional dict mapping paths to file contents.
                          Strongly recommended for better matching accuracy.
        """
        self._file_paths = file_paths
        self._file_contents = file_contents

        if self.scoring_method == "hybrid":
            # Hybrid scorer handles its own file embeddings
            device = next(self.parameters()).device if list(self.parameters()) else None
            self.hybrid_scorer.set_files(file_paths, file_contents, device)
        else:
            # Learned scorer uses file cache
            if file_contents:
                self._file_embeddings = self.file_cache.build_cache_with_content(
                    file_paths, file_contents
                )
            else:
                self._file_embeddings = self.file_cache.build_cache(file_paths)

    def forward(
        self,
        confused_tokens: List[str],
        confused_probs: Optional[List[float]] = None,
        original_query: str = "",
        generated_context: str = "",
        return_attention: bool = False,
    ) -> LearnedQueryResult:
        """
        Process CCE spike data to produce file rankings.

        Args:
            confused_tokens: Tokens the model is confused about
            confused_probs: Probabilities of confused tokens
            original_query: Original user query
            generated_context: Generated text so far
            return_attention: Whether to include attention weights

        Returns:
            LearnedQueryResult with matched files and scores
        """
        # Handle empty inputs
        if not confused_tokens:
            return LearnedQueryResult(
                query_embedding=torch.zeros(self.embed_dim),
                matched_files=[],
                file_scores={},
                confidence=0.0,
            )

        # Encode confused tokens
        token_embs, token_weights = self.token_encoder(
            confused_tokens,
            confused_probs,
            return_weights=True,
        )

        # Encode context
        context_embs = self.context_encoder(
            original_query,
            generated_context,
        )

        # Pool to query embedding
        query_emb, attention_weights = self.pooler(
            token_embs,
            context_embs,
            return_attention_weights=True,
        )

        # Score files
        matched_files = []
        file_scores = {}

        if len(self._file_paths) > 0:
            if self.scoring_method == "hybrid":
                # Use hybrid scorer (semantic + keyword)
                # If fine-tuned, use the learned query_emb for semantic matching
                # Otherwise, let scorer create embedding from text (better for zero-shot)
                matched_files, file_scores = self.hybrid_scorer.score_files(
                    query_emb=query_emb if self._is_fine_tuned else None,
                    query_tokens=confused_tokens,
                    query_text=original_query,
                    top_k=self.top_k,
                    threshold=self.score_threshold,
                    use_learned_embedding=self._is_fine_tuned,
                )
            elif self._file_embeddings is not None:
                # Use learned scorer
                device = query_emb.device if isinstance(query_emb, torch.Tensor) else 'cpu'
                file_embs = self._file_embeddings.to(device)

                matched_files, file_scores = self.scorer.score_with_threshold(
                    query_emb,
                    file_embs,
                    self._file_paths,
                    threshold=self.score_threshold,
                    top_k=self.top_k,
                )

        # Compute confidence from attention entropy
        confidence = self._compute_confidence(attention_weights)

        result = LearnedQueryResult(
            query_embedding=query_emb,
            matched_files=matched_files,
            file_scores=file_scores,
            confidence=confidence,
            confused_tokens=confused_tokens,
            token_weights=token_weights,
        )

        if return_attention and attention_weights:
            result.token_attention = attention_weights.get('token_attention')
            result.context_attention = attention_weights.get('context_attention')

        return result

    def _compute_confidence(
        self,
        attention_weights: Optional[Dict[str, torch.Tensor]],
    ) -> float:
        """
        Compute confidence from attention weights.

        Uses attention entropy - more focused attention = higher confidence.
        """
        if attention_weights is None:
            return 0.5

        token_attn = attention_weights.get('token_attention')
        if token_attn is None:
            return 0.5

        # Compute entropy of attention distribution
        # Lower entropy = more focused = higher confidence
        with torch.no_grad():
            # Average across queries and batch
            attn = token_attn.mean(dim=(0, 1))  # [K]
            attn = attn.clamp(min=1e-8)

            entropy = -(attn * torch.log(attn)).sum()
            max_entropy = np.log(len(attn))

            # Normalized inverse entropy as confidence
            if max_entropy > 0:
                confidence = 1.0 - (entropy.item() / max_entropy)
            else:
                confidence = 0.5

        return float(np.clip(confidence, 0.0, 1.0))

    def get_pooler_attention(self) -> Dict[str, Any]:
        """Get attention visualization data from the pooler."""
        return self.pooler.get_attention_weights()

    def encode_query_only(
        self,
        confused_tokens: List[str],
        confused_probs: Optional[List[float]] = None,
        original_query: str = "",
        generated_context: str = "",
    ) -> torch.Tensor:
        """
        Get just the query embedding without file scoring.

        Useful for batch processing or external scoring.
        """
        if not confused_tokens:
            return torch.zeros(self.embed_dim)

        token_embs = self.token_encoder(confused_tokens, confused_probs)
        context_embs = self.context_encoder(original_query, generated_context)
        query_emb, _ = self.pooler(token_embs, context_embs)

        return query_emb

    @torch.no_grad()
    def explain(
        self,
        confused_tokens: List[str],
        confused_probs: Optional[List[float]] = None,
        original_query: str = "",
        generated_context: str = "",
    ) -> Dict[str, Any]:
        """
        Get detailed explanation of the module's decisions.

        Returns attention heatmaps and token contributions.
        """
        result = self.forward(
            confused_tokens,
            confused_probs,
            original_query,
            generated_context,
            return_attention=True,
        )

        # Get token contributions
        token_contributions = []
        if result.token_weights is not None:
            weights = result.token_weights.cpu().numpy()
            for i, token in enumerate(confused_tokens[:len(weights)]):
                token_contributions.append({
                    'token': token,
                    'weight': float(weights[i]),
                    'prob': confused_probs[i] if confused_probs else None,
                })

        explanation = {
            'matched_files': result.matched_files,
            'file_scores': result.file_scores,
            'confidence': result.confidence,
            'token_contributions': token_contributions,
            'attention_data': self.pooler.visualize_attention(
                token_names=confused_tokens,
            ),
        }

        return explanation


def create_learned_query_module(
    config: Optional[Dict[str, Any]] = None,
    pretrained_path: Optional[str] = None,
) -> LearnedQueryModule:
    """
    Factory function to create learned query modules.

    Args:
        config: Optional configuration dict
        pretrained_path: Optional path to pretrained weights

    Returns:
        Configured LearnedQueryModule

    Example:
        >>> # Zero-shot (no training needed, works on any repo)
        >>> module = create_learned_query_module()  # Uses hybrid scoring
        >>> module.set_available_files(file_paths, file_contents)
        >>>
        >>> # With pretrained weights (for learned scoring methods)
        >>> module = create_learned_query_module(
        ...     config={'scoring_method': 'bilinear'},
        ...     pretrained_path='model.pt'
        ... )
    """
    default_config = {
        'embed_dim': 384,
        'num_query_vectors': 4,
        'num_heads': 4,
        'dropout': 0.1,
        'model_name': 'all-MiniLM-L6-v2',
        'scoring_method': 'hybrid',  # Default to hybrid for zero-shot
        'score_threshold': 0.1,
        'top_k': 3,
        'semantic_weight': 0.5,
        'keyword_weight': 0.4,
        'exact_match_boost': 0.1,
    }

    if config:
        default_config.update(config)

    module = LearnedQueryModule(**default_config)

    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        # Only load compatible weights (in case of architecture changes)
        model_dict = module.state_dict()
        compatible = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(compatible)
        module.load_state_dict(model_dict, strict=False)

    return module
