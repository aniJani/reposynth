"""
RepoSynth Retriever Module

This module provides a retriever that uses RepoSynth pack artifacts
for code retrieval during generation. It bridges the gap between
RepoSynth's analysis pipeline and the AdaptiveGenerator.

Combines:
- hybrid_search() for finding relevant symbols (keyword + FAISS)
- source_spans.json for full source code content
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from ..retriever import hybrid_search


class RepoSynthRetriever:
    """
    Retriever that uses RepoSynth pack artifacts.

    This retriever loads pre-computed embeddings and metadata from a
    RepoSynth pack directory and provides retrieval compatible with
    the CCEQueryPlusTopKRetriever interface.

    Example:
        >>> retriever = RepoSynthRetriever("./output/my-repo/pack")
        >>> results = retriever.retrieve("How does authentication work?", top_k=3)
        >>> for r in results:
        ...     print(f"{r['source']}: {len(r['content'])} chars")

    Pack directory should contain:
        - name_registry.json: Symbol metadata
        - vectors.faiss: Pre-computed embeddings
        - source_spans.json: Full source code (optional but recommended)
    """

    def __init__(self, pack_dir: str):
        """
        Initialize retriever with RepoSynth pack directory.

        Args:
            pack_dir: Path to RepoSynth pack directory
        """
        self.pack_dir = Path(pack_dir)

        if not self.pack_dir.exists():
            raise ValueError(f"Pack directory not found: {pack_dir}")

        # Load name registry (required for hybrid_search)
        registry_path = self.pack_dir / "name_registry.json"
        if not registry_path.exists():
            raise ValueError(f"name_registry.json not found in {pack_dir}")

        with open(registry_path, 'r', encoding='utf-8') as f:
            self.registry = json.load(f)

        # Load source spans (for full file content) - strongly recommended
        source_spans_path = self.pack_dir / "source_spans.json"
        if source_spans_path.exists():
            with open(source_spans_path, 'r', encoding='utf-8') as f:
                self.source_spans = json.load(f)
            print(f"RepoSynthRetriever: Loaded {len(self.source_spans)} files with full source")
        else:
            self.source_spans = {}
            print("⚠️  WARNING: No source_spans.json found!")
            print("   Retrieval quality will be degraded (only symbol metadata available).")
            print("   Re-run pipeline with --store-spans for best results:"
                  )
            print(f"   python -m orchestrator {pack_dir.parent} --with-embeddings --store-spans")

        # Track retrieved sources for deduplication
        self._retrieved_sources: set = set()

        print(f"RepoSynthRetriever initialized with {len(self.registry)} symbols")

    def reset(self):
        """Reset retrieved sources tracking for new generation."""
        self._retrieved_sources = set()

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        deduplicate: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant code using hybrid search + full source content.

        Args:
            query: Search query
            top_k: Maximum number of results to return
            deduplicate: Skip previously retrieved files (default True)

        Returns:
            List of dicts with format compatible with CCEQueryPlusTopKRetriever:
            [{'source': file_path, 'content': full_code, 'score': float}, ...]
        """
        # Use hybrid_search to find relevant symbols
        symbols = hybrid_search(
            query,
            self.pack_dir,
            max_items=top_k * 2,  # Get extra to account for deduplication
            registry=self.registry
        )

        # Build results with full source code
        results = []
        seen_files = set()

        for symbol in symbols:
            file_path = symbol.get('file_path', '')

            if not file_path:
                continue

            # Skip duplicates within this retrieval
            if file_path in seen_files:
                continue
            seen_files.add(file_path)

            # Skip if already retrieved in this generation (deduplication)
            if deduplicate and file_path in self._retrieved_sources:
                continue

            # Get full source from source_spans if available
            if file_path in self.source_spans:
                content = self.source_spans[file_path].get('full_source', '')
            else:
                # Fallback: construct minimal context from available metadata
                symbol_name = symbol.get('name', symbol.get('fqn', file_path).split('/')[-1])
                fqn = symbol.get('fqn', '')
                kind = symbol.get('kind', 'symbol')
                content = (
                    f"# {kind}: {symbol_name}\n"
                    f"# FQN: {fqn}\n"
                    f"# File: {file_path}\n"
                    f"# (Full source not available - run pipeline with --store-spans)"
                )

            if not content:
                continue

            results.append({
                'source': file_path,
                'content': content,
                'score': 1.0,  # hybrid_search doesn't return scores
            })

            # Track as retrieved
            self._retrieved_sources.add(file_path)

            if len(results) >= top_k:
                break

        return results

    def get_file_content(self, file_path: str) -> Optional[str]:
        """
        Get full content for a specific file.

        Args:
            file_path: Path to file

        Returns:
            Full source code or None if not found
        """
        if file_path in self.source_spans:
            return self.source_spans[file_path].get('full_source')
        return None

    def get_file_list(self) -> List[str]:
        """Get list of all file paths in the pack."""
        return list(self.source_spans.keys())

    def get_file_list_context(self) -> str:
        """
        Get formatted file list for use as context.

        Returns:
            String listing all files in the codebase
        """
        files = self.get_file_list()
        if not files:
            # Fall back to files from registry
            files = list(set(s.get('file_path', '') for s in self.registry.values() if s.get('file_path')))

        return "Available files:\n" + "\n".join(f"- {f}" for f in sorted(files))
