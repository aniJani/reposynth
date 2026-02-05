"""
Learned File Scorer.

This module scores files against the pooled query embedding using
various scoring functions:
- Learned: bilinear or MLP-based (requires training)
- Cosine: pure semantic similarity (no training)
- Hybrid: combines semantic + keyword matching (recommended for zero-shot)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from typing import List, Dict, Optional, Tuple, Union, Set


class LearnedFileScorer(nn.Module):
    """
    Scores files using learned embeddings and a scoring function.

    Takes a query embedding (from LearnedQueryPooler) and scores it
    against file embeddings to produce retrieval rankings.

    Scoring Methods:
        - "cosine": Cosine similarity (baseline)
        - "bilinear": Learned bilinear scoring W @ query @ file
        - "mlp": MLP on concatenated embeddings

    Example:
        >>> scorer = LearnedFileScorer(embed_dim=384, scoring_method="bilinear")
        >>> query_emb = torch.randn(384)
        >>> file_embs = torch.randn(100, 384)  # 100 files
        >>> scores = scorer(query_emb, file_embs)  # [100]
        >>> top_indices = scores.topk(3).indices
    """

    def __init__(
        self,
        embed_dim: int = 384,
        scoring_method: str = "bilinear",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        """
        Initialize file scorer.

        Args:
            embed_dim: Embedding dimension
            scoring_method: Scoring method ("cosine", "bilinear", "mlp")
            hidden_dim: Hidden dimension for MLP scorer
            dropout: Dropout rate
            temperature: Temperature for softmax scaling
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.scoring_method = scoring_method
        self.temperature = temperature

        if scoring_method == "bilinear":
            # Bilinear scoring: query @ W @ file
            self.bilinear = nn.Bilinear(embed_dim, embed_dim, 1, bias=True)

        elif scoring_method == "mlp":
            # MLP on concatenated embeddings
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

        elif scoring_method == "cosine":
            # No learnable parameters for cosine
            pass

        else:
            raise ValueError(f"Unknown scoring method: {scoring_method}")

        # Optional query projection
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_emb: torch.Tensor,
        file_embs: torch.Tensor,
        return_probabilities: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Score files against query embedding.

        Args:
            query_emb: Query embedding [D] or [B, D]
            file_embs: File embeddings [N, D] or [B, N, D]
            return_probabilities: Whether to return softmax probabilities

        Returns:
            scores: Raw scores [N] or [B, N]
            probabilities: Optional softmax probabilities [N] or [B, N]
        """
        # Handle dimensions
        query_batch = query_emb.dim() >= 2
        file_batch = file_embs.dim() >= 3

        if not query_batch:
            query_emb = query_emb.unsqueeze(0)  # [1, D]
        if not file_batch:
            file_embs = file_embs.unsqueeze(0)  # [1, N, D]

        batch_size = query_emb.shape[0]
        num_files = file_embs.shape[1]

        # Project query
        query_emb = self.query_projection(query_emb)
        query_emb = self.dropout(query_emb)

        # Compute scores based on method
        if self.scoring_method == "cosine":
            scores = self._cosine_score(query_emb, file_embs)

        elif self.scoring_method == "bilinear":
            scores = self._bilinear_score(query_emb, file_embs)

        elif self.scoring_method == "mlp":
            scores = self._mlp_score(query_emb, file_embs)

        # Remove batch dimension if not batched
        if not query_batch:
            scores = scores.squeeze(0)

        if return_probabilities:
            probs = F.softmax(scores / self.temperature, dim=-1)
            return scores, probs

        return scores

    def _cosine_score(
        self,
        query_emb: torch.Tensor,  # [B, D]
        file_embs: torch.Tensor,  # [B, N, D]
    ) -> torch.Tensor:
        """Compute cosine similarity scores."""
        # Normalize
        query_norm = F.normalize(query_emb, p=2, dim=-1)  # [B, D]
        file_norm = F.normalize(file_embs, p=2, dim=-1)  # [B, N, D]

        # Cosine similarity
        scores = torch.bmm(
            query_norm.unsqueeze(1),  # [B, 1, D]
            file_norm.transpose(1, 2),  # [B, D, N]
        ).squeeze(1)  # [B, N]

        return scores

    def _bilinear_score(
        self,
        query_emb: torch.Tensor,  # [B, D]
        file_embs: torch.Tensor,  # [B, N, D]
    ) -> torch.Tensor:
        """Compute bilinear scores."""
        batch_size = query_emb.shape[0]
        num_files = file_embs.shape[1]

        # Expand query for bilinear
        query_expanded = query_emb.unsqueeze(1).expand(-1, num_files, -1)  # [B, N, D]

        # Reshape for bilinear
        query_flat = query_expanded.reshape(-1, self.embed_dim)  # [B*N, D]
        file_flat = file_embs.reshape(-1, self.embed_dim)  # [B*N, D]

        # Bilinear scoring
        scores = self.bilinear(query_flat, file_flat)  # [B*N, 1]
        scores = scores.reshape(batch_size, num_files)  # [B, N]

        return scores

    def _mlp_score(
        self,
        query_emb: torch.Tensor,  # [B, D]
        file_embs: torch.Tensor,  # [B, N, D]
    ) -> torch.Tensor:
        """Compute MLP-based scores."""
        batch_size = query_emb.shape[0]
        num_files = file_embs.shape[1]

        # Expand query
        query_expanded = query_emb.unsqueeze(1).expand(-1, num_files, -1)  # [B, N, D]

        # Concatenate query and file embeddings
        combined = torch.cat([query_expanded, file_embs], dim=-1)  # [B, N, 2D]

        # Flatten for MLP
        combined_flat = combined.reshape(-1, self.embed_dim * 2)  # [B*N, 2D]

        # MLP scoring
        scores = self.mlp(combined_flat)  # [B*N, 1]
        scores = scores.reshape(batch_size, num_files)  # [B, N]

        return scores

    def score_with_threshold(
        self,
        query_emb: torch.Tensor,
        file_embs: torch.Tensor,
        file_paths: List[str],
        threshold: float = 0.5,
        top_k: int = 3,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Score files and return those above threshold.

        Args:
            query_emb: Query embedding
            file_embs: File embeddings
            file_paths: List of file path strings
            threshold: Minimum score threshold
            top_k: Maximum number of files to return

        Returns:
            matched_files: List of matched file paths
            score_dict: Dict mapping file paths to scores
        """
        with torch.no_grad():
            scores, probs = self.forward(query_emb, file_embs, return_probabilities=True)

        # Convert to numpy
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
            probs = probs.cpu().numpy()

        # Build score dict
        score_dict = {}
        for i, path in enumerate(file_paths):
            score_dict[path] = float(probs[i])

        # Filter by threshold and sort
        above_threshold = [
            (path, score_dict[path])
            for path in file_paths
            if score_dict[path] >= threshold
        ]
        above_threshold.sort(key=lambda x: -x[1])

        matched_files = [path for path, _ in above_threshold[:top_k]]

        return matched_files, score_dict


class FileEmbeddingCache:
    """
    Cache for file embeddings to avoid recomputation.

    Uses a sentence transformer to embed file paths and caches
    the results for efficient retrieval scoring.
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        cache_size: int = 10000,
    ):
        """
        Initialize file embedding cache.

        Args:
            model_name: SentenceTransformer model name
            cache_size: Maximum number of embeddings to cache
        """
        self.model_name = model_name
        self.cache_size = cache_size
        self._encoder = None
        self._cache: Dict[str, np.ndarray] = {}
        self._embeddings_tensor: Optional[torch.Tensor] = None
        self._file_paths: List[str] = []

    @property
    def encoder(self):
        """Lazy load encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install sentence-transformers"
                )
        return self._encoder

    def _path_to_text(self, path: str) -> str:
        """Convert file path to human-readable text for embedding."""
        import re

        parts = re.split(r'[/\\]', path)
        words = []

        for part in parts:
            name = re.sub(r'\.[^.]+$', '', part)

            for subpart in name.split('_'):
                if len(subpart) >= 2:
                    words.append(subpart.lower())

            camel_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', name)
            for cp in camel_parts:
                if len(cp) >= 2 and cp.lower() not in words:
                    words.append(cp.lower())

        return ' '.join(words)

    def build_cache(self, file_paths: List[str]) -> torch.Tensor:
        """
        Build embedding cache for file paths.

        Args:
            file_paths: List of file paths to embed

        Returns:
            Tensor of file embeddings [N, D]
        """
        self._file_paths = file_paths

        # Convert paths to readable text
        texts = [self._path_to_text(path) for path in file_paths]

        # Embed all paths
        embeddings = self.encoder.encode(texts, show_progress_bar=False)

        # Cache
        for i, path in enumerate(file_paths):
            self._cache[path] = embeddings[i]

        self._embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

        return self._embeddings_tensor

    def get_embeddings(self, file_paths: Optional[List[str]] = None) -> torch.Tensor:
        """
        Get embeddings for file paths.

        Args:
            file_paths: Optional specific paths (uses cached paths if None)

        Returns:
            Tensor of embeddings
        """
        if file_paths is None:
            if self._embeddings_tensor is None:
                raise ValueError("No embeddings cached. Call build_cache first.")
            return self._embeddings_tensor

        # Get specific embeddings
        embeddings = []
        for path in file_paths:
            if path in self._cache:
                embeddings.append(self._cache[path])
            else:
                # Compute and cache
                text = self._path_to_text(path)
                emb = self.encoder.encode([text], show_progress_bar=False)[0]
                self._cache[path] = emb
                embeddings.append(emb)

        return torch.tensor(np.stack(embeddings), dtype=torch.float32)

    def get_file_paths(self) -> List[str]:
        """Get the cached file paths in order."""
        return self._file_paths.copy()

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._embeddings_tensor = None
        self._file_paths = []


class HybridFileScorer(nn.Module):
    """
    Hybrid scorer combining semantic similarity with keyword matching.

    This scorer works well for zero-shot transfer to new codebases because:
    1. Semantic scoring captures meaning ("authentication" ≈ "auth")
    2. Keyword scoring captures exact matches (file has "auth" in path/content)
    3. Exact match boost rewards direct token matches in file paths

    No per-repo training required - just embed files once.

    Example:
        >>> scorer = HybridFileScorer()
        >>> scorer.set_files(file_paths, file_contents)
        >>> scores = scorer.score(
        ...     query_emb=pooled_query,
        ...     query_tokens=["async", "await"],
        ...     query_text="How do I use async?"
        ... )
    """

    def __init__(
        self,
        embed_dim: int = 384,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.4,
        exact_match_boost: float = 0.1,
        model_name: str = 'all-MiniLM-L6-v2',
    ):
        """
        Initialize hybrid scorer.

        Args:
            embed_dim: Embedding dimension
            semantic_weight: Weight for cosine similarity score (0-1)
            keyword_weight: Weight for keyword overlap score (0-1)
            exact_match_boost: Bonus when query token appears in file path (0-1)
            model_name: SentenceTransformer model for embeddings
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.exact_match_boost = exact_match_boost
        self.model_name = model_name

        # Learnable projection to map pooler output to file embedding space
        # This is trained during fine-tuning to align the spaces
        self.query_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # File data (set via set_files)
        self._file_paths: List[str] = []
        self._file_keywords: List[Set[str]] = []
        self._file_embeddings: Optional[torch.Tensor] = None
        self._encoder = None

    @property
    def encoder(self):
        """Lazy load encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install sentence-transformers"
                )
        return self._encoder

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text (lowercase)."""
        # Split on non-alphanumeric
        words = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', text.lower())
        # Filter short words and common ones
        stopwords = {'the', 'is', 'to', 'and', 'for', 'in', 'of', 'a', 'an', 'it', 'be'}
        return {w for w in words if len(w) >= 2 and w not in stopwords}

    def _extract_file_summary(self, path: str, content: str) -> str:
        """Extract meaningful summary from file for embedding."""
        parts = []

        # 1. Path parts
        path_parts = re.split(r'[/\\._]', path.lower())
        parts.append(' '.join(p for p in path_parts if len(p) >= 2))

        # 2. Module docstring
        docstring_match = re.search(r'^"""(.+?)"""', content, re.DOTALL)
        if docstring_match:
            parts.append(docstring_match.group(1).strip()[:200])

        # 3. Class names
        for match in re.finditer(r'class\s+(\w+)', content):
            parts.append(match.group(1))

        # 4. Function names (non-private)
        for match in re.finditer(r'def\s+([a-zA-Z][a-zA-Z0-9_]*)\s*\(', content):
            name = match.group(1)
            if not name.startswith('_'):
                parts.append(name)

        # 5. First N chars
        parts.append(content[:300])

        return ' '.join(parts)[:1000]

    def _extract_file_keywords(self, path: str, content: str) -> Set[str]:
        """Extract keywords from file path and content."""
        keywords = set()

        # Path keywords
        path_parts = re.split(r'[/\\._]', path.lower())
        keywords.update(p for p in path_parts if len(p) >= 2)

        # Class names
        for match in re.finditer(r'class\s+(\w+)', content):
            keywords.add(match.group(1).lower())

        # Function names
        for match in re.finditer(r'def\s+(\w+)', content):
            name = match.group(1).lower()
            if not name.startswith('_'):
                keywords.add(name)

        # Important identifiers
        for match in re.finditer(r'\b([A-Z][a-z]+[A-Z]\w*)\b', content):  # CamelCase
            keywords.add(match.group(1).lower())

        return keywords

    def set_files(
        self,
        file_paths: List[str],
        file_contents: Optional[Dict[str, str]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Set files for scoring. Call this once per repo.

        Args:
            file_paths: List of file paths
            file_contents: Optional dict mapping paths to contents
                          (if None, uses path-based embeddings only)
            device: Torch device for embeddings
        """
        self._file_paths = file_paths
        file_contents = file_contents or {}

        # Extract keywords for each file
        self._file_keywords = []
        summaries = []

        for path in file_paths:
            content = file_contents.get(path, '')

            # Keywords from path and content
            keywords = self._extract_file_keywords(path, content)
            self._file_keywords.append(keywords)

            # Summary for embedding
            if content:
                summary = self._extract_file_summary(path, content)
            else:
                # Fallback to path-based
                path_parts = re.split(r'[/\\._]', path.lower())
                summary = ' '.join(p for p in path_parts if len(p) >= 2)
            summaries.append(summary)

        # Embed summaries
        embeddings = self.encoder.encode(summaries, show_progress_bar=len(summaries) > 20)
        self._file_embeddings = torch.tensor(embeddings, dtype=torch.float32)

        if device is not None:
            self._file_embeddings = self._file_embeddings.to(device)

    def forward(
        self,
        query_emb: Optional[torch.Tensor] = None,
        query_tokens: Optional[List[str]] = None,
        query_text: Optional[str] = None,
        return_components: bool = False,
        use_learned_embedding: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Score files using hybrid approach.

        Args:
            query_emb: Optional query embedding from pooler [D] or [B, D]
                      If None, uses query_text + query_tokens for semantic matching
            query_tokens: Confused tokens for keyword matching
            query_text: Original query text for keyword/semantic matching
            return_components: Whether to return individual score components
            use_learned_embedding: If True and query_emb is provided, use it directly
                                  for semantic matching instead of creating new embedding.
                                  Set this to True after fine-tuning.

        Returns:
            scores: Combined scores [N] or [B, N]
            components: Optional dict with semantic_scores, keyword_scores, exact_boost
        """
        if self._file_embeddings is None:
            raise ValueError("No files set. Call set_files first.")

        num_files = len(self._file_paths)

        # Determine whether to use provided embedding or create from text
        # If use_learned_embedding=True and query_emb is provided, use it directly
        # Otherwise, create embedding from query_text + query_tokens (better for zero-shot)
        if use_learned_embedding and query_emb is not None:
            # Use the learned embedding with projection to align with file embedding space
            query_emb = self.query_projection(query_emb)
        elif query_emb is None or (query_text or query_tokens):
            # Build query from text and tokens (zero-shot mode)
            query_parts = []
            if query_text:
                query_parts.append(query_text)
            if query_tokens:
                query_parts.append(' '.join(query_tokens))

            if query_parts:
                combined_query = ' '.join(query_parts)
                query_emb_np = self.encoder.encode([combined_query], show_progress_bar=False)
                query_emb = torch.tensor(query_emb_np, dtype=torch.float32)
            elif query_emb is None:
                raise ValueError("Must provide query_emb, query_text, or query_tokens")

        # Ensure proper dimensions
        single_query = query_emb.dim() == 1
        if single_query:
            query_emb = query_emb.unsqueeze(0)

        batch_size = query_emb.shape[0]
        device = query_emb.device

        # Move file embeddings to same device
        file_embs = self._file_embeddings.to(device)

        # 1. Semantic score (cosine similarity)
        query_norm = F.normalize(query_emb, p=2, dim=-1)  # [B, D]
        file_norm = F.normalize(file_embs, p=2, dim=-1)   # [N, D]
        semantic_scores = torch.mm(query_norm, file_norm.T)  # [B, N]

        # Normalize to [0, 1]
        semantic_scores = (semantic_scores + 1) / 2  # cosine is [-1, 1]

        # 2. Keyword score (overlap)
        query_keywords = set()
        if query_tokens:
            query_keywords.update(t.lower() for t in query_tokens if len(t) >= 2)
        if query_text:
            query_keywords.update(self._extract_keywords(query_text))

        keyword_scores = torch.zeros(batch_size, num_files, device=device)
        if query_keywords:
            for i, file_kw in enumerate(self._file_keywords):
                overlap = len(query_keywords & file_kw)
                # Normalize by query keywords count
                keyword_scores[:, i] = overlap / len(query_keywords)

        # 3. Exact match boost (token appears in file path)
        exact_boost = torch.zeros(batch_size, num_files, device=device)
        if query_tokens:
            for i, path in enumerate(self._file_paths):
                path_lower = path.lower()
                for token in query_tokens:
                    if token.lower() in path_lower:
                        exact_boost[:, i] = 1.0
                        break

        # 4. Combine scores
        combined = (
            self.semantic_weight * semantic_scores +
            self.keyword_weight * keyword_scores +
            self.exact_match_boost * exact_boost
        )

        # Remove batch dim if single query
        if single_query:
            combined = combined.squeeze(0)
            semantic_scores = semantic_scores.squeeze(0)
            keyword_scores = keyword_scores.squeeze(0)
            exact_boost = exact_boost.squeeze(0)

        if return_components:
            return combined, {
                'semantic': semantic_scores,
                'keyword': keyword_scores,
                'exact_boost': exact_boost,
            }

        return combined

    def score_files(
        self,
        query_emb: Optional[torch.Tensor] = None,
        query_tokens: Optional[List[str]] = None,
        query_text: Optional[str] = None,
        top_k: int = 3,
        threshold: float = 0.0,
        use_learned_embedding: bool = False,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Score files and return top matches.

        Args:
            query_emb: Query embedding
            query_tokens: Confused tokens
            query_text: Original query
            top_k: Number of files to return
            threshold: Minimum score threshold
            use_learned_embedding: If True, use query_emb directly for semantic matching

        Returns:
            matched_files: Top-k file paths
            score_dict: All file scores
        """
        with torch.no_grad():
            scores = self.forward(
                query_emb, query_tokens, query_text,
                use_learned_embedding=use_learned_embedding
            )

        # Handle batch dimension
        if scores.dim() > 1:
            scores = scores.squeeze(0)

        scores_np = scores.cpu().numpy()

        # Build score dict
        score_dict = {
            path: float(scores_np[i])
            for i, path in enumerate(self._file_paths)
        }

        # Sort by score
        sorted_files = sorted(score_dict.items(), key=lambda x: -x[1])

        # Filter and take top-k
        matched_files = [
            path for path, score in sorted_files[:top_k]
            if score >= threshold
        ]

        return matched_files, score_dict

    def get_file_paths(self) -> List[str]:
        """Get cached file paths."""
        return self._file_paths.copy()


class ContentAwareFileCache(FileEmbeddingCache):
    """
    Enhanced file embedding cache that uses file content, not just paths.

    This produces better embeddings for semantic matching because it
    captures what files actually contain, not just their names.

    Example:
        >>> cache = ContentAwareFileCache()
        >>> embeddings = cache.build_cache(file_paths, file_contents)
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        cache_size: int = 10000,
    ):
        super().__init__(model_name, cache_size)
        self._file_keywords: Dict[str, Set[str]] = {}

    def _extract_summary(self, path: str, content: str) -> str:
        """Extract content-aware summary for embedding."""
        parts = []

        # Path parts
        path_parts = re.split(r'[/\\._]', path.lower())
        parts.append(' '.join(p for p in path_parts if len(p) >= 2))

        # Module docstring
        doc_match = re.search(r'^"""(.+?)"""', content, re.DOTALL)
        if doc_match:
            parts.append(doc_match.group(1).strip()[:200])

        # Class names with docstrings
        for match in re.finditer(r'class\s+(\w+)[^:]*:\s*(?:"""(.+?)""")?', content, re.DOTALL):
            class_name = match.group(1)
            class_doc = (match.group(2) or '')[:50]
            parts.append(f"class {class_name} {class_doc}")

        # Function signatures
        for match in re.finditer(r'def\s+([a-zA-Z]\w*)\s*\([^)]*\)', content):
            name = match.group(1)
            if not name.startswith('_'):
                parts.append(name)

        # First 300 chars
        parts.append(content[:300])

        return ' '.join(parts)[:1000]

    def _extract_keywords(self, path: str, content: str) -> Set[str]:
        """Extract keywords for keyword matching."""
        keywords = set()

        # Path keywords
        path_parts = re.split(r'[/\\._]', path.lower())
        keywords.update(p for p in path_parts if len(p) >= 2)

        # Class names
        for match in re.finditer(r'class\s+(\w+)', content):
            keywords.add(match.group(1).lower())

        # Function names
        for match in re.finditer(r'def\s+(\w+)', content):
            name = match.group(1).lower()
            if not name.startswith('_'):
                keywords.add(name)

        return keywords

    def build_cache_with_content(
        self,
        file_paths: List[str],
        file_contents: Dict[str, str],
    ) -> torch.Tensor:
        """
        Build embedding cache using file contents.

        Args:
            file_paths: List of file paths
            file_contents: Dict mapping paths to file contents

        Returns:
            Tensor of file embeddings [N, D]
        """
        self._file_paths = file_paths

        # Build summaries and extract keywords
        summaries = []
        for path in file_paths:
            content = file_contents.get(path, '')

            if content:
                summary = self._extract_summary(path, content)
                keywords = self._extract_keywords(path, content)
            else:
                summary = self._path_to_text(path)
                keywords = set(re.split(r'[/\\._]', path.lower()))

            summaries.append(summary)
            self._file_keywords[path] = keywords

        # Embed summaries
        embeddings = self.encoder.encode(summaries, show_progress_bar=len(summaries) > 20)

        # Cache
        for i, path in enumerate(file_paths):
            self._cache[path] = embeddings[i]

        self._embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        return self._embeddings_tensor

    def get_keywords(self, path: str) -> Set[str]:
        """Get keywords for a file."""
        return self._file_keywords.get(path, set())

    def get_all_keywords(self) -> Dict[str, Set[str]]:
        """Get all file keywords."""
        return self._file_keywords.copy()
