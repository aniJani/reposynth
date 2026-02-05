"""
Encoders for Learned Query Pooling.

This module provides encoders for converting confused tokens and context
into embeddings suitable for cross-attention.

Components:
- ConfusedTokenEncoder: Encodes confused tokens with probability weighting
- ContextEncoder: Encodes query and generation context
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Union


class ConfusedTokenEncoder(nn.Module):
    """
    Encodes confused tokens from CCE spikes into embeddings.

    Takes confused tokens (strings) with their probabilities and produces
    embeddings suitable for cross-attention. Uses a pre-trained sentence
    transformer backbone with optional probability-based weighting.

    Architecture:
        tokens → SentenceTransformer → [K x D] embeddings
        probabilities → softmax → attention weights

    Example:
        >>> encoder = ConfusedTokenEncoder()
        >>> tokens = ["auth", "jwt", "bearer"]
        >>> probs = [0.3, 0.25, 0.2]
        >>> embeddings = encoder(tokens, probs)  # [3, 384]
    """

    def __init__(
        self,
        embed_dim: int = 384,
        model_name: str = 'all-MiniLM-L6-v2',
        use_probability_weighting: bool = True,
        dropout: float = 0.1,
        max_tokens: int = 20,
    ):
        """
        Initialize confused token encoder.

        Args:
            embed_dim: Output embedding dimension (should match model)
            model_name: SentenceTransformer model name
            use_probability_weighting: Whether to weight by token probabilities
            dropout: Dropout rate
            max_tokens: Maximum number of tokens to encode
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.model_name = model_name
        self.use_probability_weighting = use_probability_weighting
        self.max_tokens = max_tokens

        # Lazy load sentence transformer
        self._encoder = None

        # Optional projection if model dim differs from embed_dim
        self._projection = None

        # Probability weighting network
        if use_probability_weighting:
            self.prob_transform = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )
        else:
            self.prob_transform = None

        self.dropout = nn.Dropout(dropout)

        # Layer norm for output stability
        self.output_norm = nn.LayerNorm(embed_dim)

    @property
    def encoder(self):
        """Lazy load the sentence transformer encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name)

                # Check if projection is needed
                test_emb = self._encoder.encode(["test"])
                actual_dim = test_emb.shape[1]

                if actual_dim != self.embed_dim:
                    self._projection = nn.Linear(actual_dim, self.embed_dim)

            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install sentence-transformers"
                )
        return self._encoder

    def forward(
        self,
        tokens: List[str],
        probabilities: Optional[List[float]] = None,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode confused tokens into embeddings.

        Args:
            tokens: List of confused token strings
            probabilities: Optional list of token probabilities from logits
            return_weights: Whether to return probability weights

        Returns:
            Tensor of shape [K, embed_dim] where K = min(len(tokens), max_tokens)
            If return_weights=True, also returns weight tensor [K]
        """
        if not tokens:
            # Return empty tensor with correct shape
            empty = torch.zeros(1, self.embed_dim)
            if return_weights:
                return empty, torch.ones(1)
            return empty

        # Limit tokens
        tokens = tokens[:self.max_tokens]

        # Encode tokens using sentence transformer
        with torch.no_grad():
            raw_embeddings = self.encoder.encode(
                tokens,
                convert_to_tensor=True,
                show_progress_bar=False,
            )

        # Move to same device as model parameters
        device = next(self.parameters()).device if list(self.parameters()) else raw_embeddings.device
        embeddings = raw_embeddings.to(device)

        # Project if needed
        if self._projection is not None:
            embeddings = self._projection(embeddings)

        # Apply dropout
        embeddings = self.dropout(embeddings)

        # Compute probability weights
        weights = None
        if self.use_probability_weighting and probabilities is not None:
            probs = probabilities[:len(tokens)]
            prob_tensor = torch.tensor(probs, dtype=torch.float32, device=device)
            prob_tensor = prob_tensor.unsqueeze(-1)  # [K, 1]

            # Transform probabilities to weights
            weights = self.prob_transform(prob_tensor).squeeze(-1)  # [K]

            # Weight embeddings
            embeddings = embeddings * weights.unsqueeze(-1)
        else:
            weights = torch.ones(len(tokens), device=device)

        # Normalize output
        embeddings = self.output_norm(embeddings)

        if return_weights:
            return embeddings, weights

        return embeddings

    def encode_with_metadata(
        self,
        tokens: List[str],
        probabilities: Optional[List[float]] = None,
    ) -> dict:
        """
        Encode tokens and return detailed metadata.

        Args:
            tokens: List of confused tokens
            probabilities: Optional probabilities

        Returns:
            Dict with embeddings, weights, tokens used, etc.
        """
        embeddings, weights = self.forward(tokens, probabilities, return_weights=True)

        return {
            'embeddings': embeddings,
            'weights': weights,
            'tokens': tokens[:self.max_tokens],
            'num_tokens': len(tokens[:self.max_tokens]),
            'probabilities': probabilities[:len(tokens[:self.max_tokens])] if probabilities else None,
        }


class ContextEncoder(nn.Module):
    """
    Encodes query and generation context into embeddings.

    Combines the original user query with the generated context so far
    to provide positional/semantic information for cross-attention.

    Architecture:
        [query, context] → SentenceTransformer → split/pool → [L x D]

    Example:
        >>> encoder = ContextEncoder()
        >>> query = "How does authentication work?"
        >>> context = "In our Flask app, the authentication system uses"
        >>> embeddings = encoder(query, context)  # [L, 384]
    """

    def __init__(
        self,
        embed_dim: int = 384,
        model_name: str = 'all-MiniLM-L6-v2',
        max_context_length: int = 512,
        chunk_size: int = 128,
        dropout: float = 0.1,
    ):
        """
        Initialize context encoder.

        Args:
            embed_dim: Output embedding dimension
            model_name: SentenceTransformer model name
            max_context_length: Maximum context characters to use
            chunk_size: Characters per chunk for encoding
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.chunk_size = chunk_size

        # Lazy load encoder
        self._encoder = None
        self._projection = None

        self.dropout = nn.Dropout(dropout)

        # Query vs context type embeddings
        self.type_embeddings = nn.Embedding(2, embed_dim)  # 0=query, 1=context

        # Layer norm
        self.output_norm = nn.LayerNorm(embed_dim)

    @property
    def encoder(self):
        """Lazy load the sentence transformer encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name)

                # Check if projection is needed
                test_emb = self._encoder.encode(["test"])
                actual_dim = test_emb.shape[1]

                if actual_dim != self.embed_dim:
                    self._projection = nn.Linear(actual_dim, self.embed_dim)

            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install sentence-transformers"
                )
        return self._encoder

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for encoding."""
        text = text[-self.max_context_length:]  # Take recent context

        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)

        return chunks if chunks else [""]

    def forward(
        self,
        query: str,
        context: str,
        return_separate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode query and context into embeddings.

        Args:
            query: Original user query
            context: Generated text so far
            return_separate: Whether to return query and context separately

        Returns:
            Tensor of shape [L, embed_dim] combining query and context
            If return_separate=True, returns (query_emb, context_emb)
        """
        # Prepare texts to encode
        texts_to_encode = []
        text_types = []  # 0=query, 1=context

        # Add query (as single embedding)
        if query:
            texts_to_encode.append(query)
            text_types.append(0)

        # Add context chunks
        context_chunks = self._chunk_text(context)
        texts_to_encode.extend(context_chunks)
        text_types.extend([1] * len(context_chunks))

        if not texts_to_encode:
            texts_to_encode = [""]
            text_types = [1]

        # Encode all texts
        with torch.no_grad():
            raw_embeddings = self.encoder.encode(
                texts_to_encode,
                convert_to_tensor=True,
                show_progress_bar=False,
            )

        # Move to device
        device = next(self.parameters()).device if list(self.parameters()) else raw_embeddings.device
        embeddings = raw_embeddings.to(device)

        # Project if needed
        if self._projection is not None:
            embeddings = self._projection(embeddings)

        # Add type embeddings
        type_tensor = torch.tensor(text_types, dtype=torch.long, device=device)
        type_embs = self.type_embeddings(type_tensor)
        embeddings = embeddings + type_embs

        # Apply dropout and norm
        embeddings = self.dropout(embeddings)
        embeddings = self.output_norm(embeddings)

        if return_separate and query:
            query_emb = embeddings[0:1]  # [1, D]
            context_emb = embeddings[1:] if embeddings.shape[0] > 1 else embeddings
            return query_emb, context_emb

        return embeddings

    def encode_query_only(self, query: str) -> torch.Tensor:
        """
        Encode only the query.

        Args:
            query: User query string

        Returns:
            Query embedding [1, embed_dim]
        """
        if not query:
            return torch.zeros(1, self.embed_dim)

        with torch.no_grad():
            raw_embedding = self.encoder.encode(
                [query],
                convert_to_tensor=True,
                show_progress_bar=False,
            )

        device = next(self.parameters()).device if list(self.parameters()) else raw_embedding.device
        embedding = raw_embedding.to(device)

        if self._projection is not None:
            embedding = self._projection(embedding)

        # Add query type embedding
        type_emb = self.type_embeddings(torch.tensor([0], device=device))
        embedding = embedding + type_emb

        return self.output_norm(embedding)
