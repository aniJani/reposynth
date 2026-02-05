"""
Learned Query Pooler with Cross-Attention.

This module implements the core cross-attention mechanism that uses
learnable query vectors to attend over confused token embeddings,
producing optimized retrieval query embeddings.

Architecture:
    ┌─────────────────────────────────────┐
    │   Learnable Query Vectors [Q x D]   │
    └─────────────────────────────────────┘
                     │
    ┌────────────────┴────────────────┐
    ▼                                 ▼
┌───────────────┐         ┌───────────────┐
│ Token Cross-  │         │ Context Cross-│
│   Attention   │         │   Attention   │
│ Q→tokens [K]  │         │ Q→context [L] │
└───────────────┘         └───────────────┘
         │                         │
         └────────────┬────────────┘
                      ▼
              ┌───────────────┐
              │    Fusion     │
              │ concat + MLP  │
              └───────────────┘
                      │
                      ▼
              ┌───────────────┐
              │  Pool + Proj  │
              │   → [1, D]    │
              └───────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class LearnedQueryPooler(nn.Module):
    """
    Cross-attention pooler with learnable query vectors.

    Uses learnable query vectors to attend over:
    1. Confused token embeddings (what the model is uncertain about)
    2. Context embeddings (query + generation context)

    The attended representations are fused and pooled to produce
    a single retrieval query embedding.

    Example:
        >>> pooler = LearnedQueryPooler(embed_dim=384, num_query_vectors=4)
        >>> confused_embs = torch.randn(10, 384)  # 10 confused tokens
        >>> context_embs = torch.randn(5, 384)    # 5 context chunks
        >>> query_emb = pooler(confused_embs, context_embs)  # [384]
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_query_vectors: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        fusion_method: str = "concat",  # concat, add, gated
    ):
        """
        Initialize learned query pooler.

        Args:
            embed_dim: Embedding dimension (should match all-MiniLM-L6-v2 = 384)
            num_query_vectors: Number of learnable query vectors
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
            fusion_method: How to combine token and context attention
                - "concat": Concatenate and project
                - "add": Simple addition
                - "gated": Learned gating mechanism
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_query_vectors = num_query_vectors
        self.num_heads = num_heads
        self.fusion_method = fusion_method

        # Learnable query vectors
        # Initialize with small random values for stability
        self.query_vectors = nn.Parameter(
            torch.randn(num_query_vectors, embed_dim) * 0.02
        )

        # Cross-attention for confused tokens
        self.token_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention for context
        self.context_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Fusion network
        if fusion_method == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif fusion_method == "gated":
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid(),
            )
            self.fusion = None
        else:  # add
            self.fusion = None

        # Output projection
        self.output_projection = nn.Linear(embed_dim, embed_dim)

        # Layer norms
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.token_norm = nn.LayerNorm(embed_dim)
            self.context_norm = nn.LayerNorm(embed_dim)
            self.output_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

        # For attention visualization
        self._last_token_attn_weights = None
        self._last_context_attn_weights = None

    def forward(
        self,
        confused_token_embs: torch.Tensor,
        context_embs: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Pool confused tokens and context into retrieval query embedding.

        Args:
            confused_token_embs: Embeddings of confused tokens [K, D]
            context_embs: Embeddings of context chunks [L, D]
            return_attention_weights: Whether to return attention weights

        Returns:
            query_embedding: Pooled query embedding [D]
            attention_weights: Optional dict with token and context attention weights
        """
        # Add batch dimension if needed
        if confused_token_embs.dim() == 2:
            confused_token_embs = confused_token_embs.unsqueeze(0)  # [1, K, D]
        if context_embs.dim() == 2:
            context_embs = context_embs.unsqueeze(0)  # [1, L, D]

        batch_size = confused_token_embs.shape[0]

        # Expand query vectors for batch
        queries = self.query_vectors.unsqueeze(0).expand(batch_size, -1, -1)  # [B, Q, D]

        # Cross-attend to confused tokens
        # Q: learnable queries, K/V: confused token embeddings
        token_attended, token_attn_weights = self.token_cross_attn(
            query=queries,
            key=confused_token_embs,
            value=confused_token_embs,
        )  # [B, Q, D], [B, Q, K]

        if self.use_layer_norm:
            token_attended = self.token_norm(token_attended)

        # Cross-attend to context
        # Q: learnable queries, K/V: context embeddings
        context_attended, context_attn_weights = self.context_cross_attn(
            query=queries,
            key=context_embs,
            value=context_embs,
        )  # [B, Q, D], [B, Q, L]

        if self.use_layer_norm:
            context_attended = self.context_norm(context_attended)

        # Store attention weights for visualization
        self._last_token_attn_weights = token_attn_weights.detach()
        self._last_context_attn_weights = context_attn_weights.detach()

        # Fuse token and context attended representations
        if self.fusion_method == "concat":
            # Concatenate along feature dimension and project
            fused = torch.cat([token_attended, context_attended], dim=-1)  # [B, Q, 2D]
            fused = self.fusion(fused)  # [B, Q, D]
        elif self.fusion_method == "gated":
            # Learned gating
            gate_input = torch.cat([token_attended, context_attended], dim=-1)
            gate = self.gate(gate_input)  # [B, Q, D]
            fused = gate * token_attended + (1 - gate) * context_attended
        else:  # add
            fused = token_attended + context_attended

        # Pool query vectors (mean pooling)
        pooled = fused.mean(dim=1)  # [B, D]

        # Output projection
        output = self.output_projection(pooled)
        output = self.dropout(output)

        if self.use_layer_norm:
            output = self.output_norm(output)

        # Remove batch dimension if batch_size == 1
        if batch_size == 1:
            output = output.squeeze(0)  # [D]

        # Prepare attention weights if requested
        attention_weights = None
        if return_attention_weights:
            attention_weights = {
                'token_attention': token_attn_weights,  # [B, Q, K]
                'context_attention': context_attn_weights,  # [B, Q, L]
            }

        return output, attention_weights

    def get_attention_weights(self) -> Dict[str, Optional[torch.Tensor]]:
        """
        Get the last computed attention weights.

        Returns:
            Dict with 'token_attention' and 'context_attention' tensors
        """
        return {
            'token_attention': self._last_token_attn_weights,
            'context_attention': self._last_context_attn_weights,
        }

    def get_query_vectors(self) -> torch.Tensor:
        """Get the learnable query vectors."""
        return self.query_vectors.data

    @torch.no_grad()
    def visualize_attention(
        self,
        token_names: Optional[list] = None,
        context_names: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Get attention visualization data.

        Args:
            token_names: Optional names for confused tokens
            context_names: Optional names for context chunks

        Returns:
            Dict with attention matrices and labels for visualization
        """
        token_attn = self._last_token_attn_weights
        context_attn = self._last_context_attn_weights

        if token_attn is None or context_attn is None:
            return {}

        # Convert to numpy for visualization
        token_attn_np = token_attn.cpu().numpy()
        context_attn_np = context_attn.cpu().numpy()

        return {
            'token_attention_matrix': token_attn_np,
            'context_attention_matrix': context_attn_np,
            'token_names': token_names or [f'token_{i}' for i in range(token_attn_np.shape[-1])],
            'context_names': context_names or [f'ctx_{i}' for i in range(context_attn_np.shape[-1])],
            'query_names': [f'Q{i}' for i in range(self.num_query_vectors)],
        }


class AttentionPooler(nn.Module):
    """
    Simpler attention-based pooler (no learnable queries).

    Uses self-attention to pool confused token embeddings into a single
    representation. Used as a baseline for comparison.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        token_embs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool token embeddings using self-attention.

        Args:
            token_embs: Token embeddings [K, D] or [B, K, D]
            mask: Optional attention mask

        Returns:
            Pooled embedding [D] or [B, D]
        """
        if token_embs.dim() == 2:
            token_embs = token_embs.unsqueeze(0)

        # Self-attention
        attended, _ = self.self_attention(
            token_embs, token_embs, token_embs,
            key_padding_mask=mask,
        )

        attended = self.layer_norm(attended + token_embs)  # Residual

        # Mean pool
        pooled = attended.mean(dim=1)

        output = self.output_projection(pooled)

        if output.shape[0] == 1:
            output = output.squeeze(0)

        return output
