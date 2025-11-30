# --- FILE: packages/python-orchestrator/orchestrator/context_optimizer.py ---
"""
Context Optimizer - Graph-Knapsack Algorithm for LLM Context Window Optimization.

This module implements the core "ContextLoom" algorithm that intelligently selects
which files to include in an LLM prompt while respecting token budget constraints.

The algorithm uses a combination of:
1. Import graph analysis to determine file importance
2. Scoring based on relationship to seed files
3. Knapsack optimization to maximize value within token budget
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger("reposynth.context_optimizer")

# Scoring constants (tuneable)
SCORE_SEED_FILE = 1000       # Must-include files selected by user
SCORE_DIRECT_IMPORT = 50     # Files directly imported by seed
SCORE_SECONDARY_IMPORT = 25  # Files imported by direct imports (2 hops)
SCORE_TERTIARY_IMPORT = 10   # Files 3 hops away
SCORE_REVERSE_IMPORT = 30    # Files that import the seed (dependents)
SCORE_HUB_BONUS = 5          # Bonus per import (for highly-connected files)


@dataclass
class FileScore:
    """Represents a file's optimization score and metadata."""
    file_path: str
    tokens: int
    score: int
    reasons: List[str] = field(default_factory=list)
    depth: int = 999  # Distance from seed (999 = unrelated)


@dataclass
class PruningReport:
    """Report of what was included/excluded and why."""
    included_files: List[Dict]  # [{file_path, tokens, score, reasons}]
    pruned_files: List[Dict]    # [{file_path, tokens, score, reason}]
    total_tokens_included: int
    total_tokens_pruned: int
    token_budget: int
    seed_files: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "included": self.included_files,
            "pruned": self.pruned_files,
            "summary": {
                "files_included": len(self.included_files),
                "files_pruned": len(self.pruned_files),
                "tokens_included": self.total_tokens_included,
                "tokens_pruned": self.total_tokens_pruned,
                "token_budget": self.token_budget,
                "budget_utilization": f"{(self.total_tokens_included / self.token_budget * 100):.1f}%" if self.token_budget > 0 else "N/A"
            },
            "seed_files": self.seed_files
        }


def build_reverse_graph(import_graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Build a reverse import graph (who imports this file?).
    
    Args:
        import_graph: {file_path: [imported_files]}
        
    Returns:
        Reverse graph: {file_path: [files_that_import_it]}
    """
    reverse = defaultdict(list)
    for file_path, imports in import_graph.items():
        for imported in imports:
            reverse[imported].append(file_path)
    return dict(reverse)


def calculate_file_scores(
    seed_files: List[str],
    import_graph: Dict[str, List[str]],
    token_map: Dict[str, int],
    candidate_files: Optional[List[str]] = None
) -> Dict[str, FileScore]:
    """
    Calculate importance scores for all files based on their relationship to seed files.
    
    Args:
        seed_files: Files explicitly selected by the user (entry points).
        import_graph: {file_path: [imported_files]}
        token_map: {file_path: token_count}
        candidate_files: Optional list of files to consider. If None, uses all files in token_map.
        
    Returns:
        Dictionary of file_path -> FileScore
    """
    scores: Dict[str, FileScore] = {}
    reverse_graph = build_reverse_graph(import_graph)
    
    # Determine candidate set
    all_files = set(candidate_files) if candidate_files else set(token_map.keys())
    
    # Initialize all files with base score
    for file_path in all_files:
        tokens = token_map.get(file_path, 0)
        scores[file_path] = FileScore(
            file_path=file_path,
            tokens=tokens,
            score=0,
            reasons=[],
            depth=999
        )
    
    # BFS to calculate depth and scores from seed files
    visited: Set[str] = set()
    current_level = set(seed_files)
    depth = 0
    
    while current_level and depth <= 3:
        next_level: Set[str] = set()
        
        for file_path in current_level:
            if file_path in visited:
                continue
            visited.add(file_path)
            
            if file_path not in scores:
                # File not in candidate set, skip
                continue
            
            score_obj = scores[file_path]
            score_obj.depth = depth
            
            if depth == 0:
                # Seed file
                score_obj.score += SCORE_SEED_FILE
                score_obj.reasons.append("seed_file")
            elif depth == 1:
                # Direct import
                score_obj.score += SCORE_DIRECT_IMPORT
                score_obj.reasons.append("direct_import")
            elif depth == 2:
                # Secondary import
                score_obj.score += SCORE_SECONDARY_IMPORT
                score_obj.reasons.append("secondary_import")
            elif depth == 3:
                # Tertiary import
                score_obj.score += SCORE_TERTIARY_IMPORT
                score_obj.reasons.append("tertiary_import")
            
            # Add imports of this file to next level
            for imported in import_graph.get(file_path, []):
                if imported not in visited and imported in all_files:
                    next_level.add(imported)
        
        current_level = next_level
        depth += 1
    
    # Add reverse import scores (files that import seed files)
    for seed in seed_files:
        for dependent in reverse_graph.get(seed, []):
            if dependent in scores and dependent not in seed_files:
                if "reverse_import" not in scores[dependent].reasons:
                    scores[dependent].score += SCORE_REVERSE_IMPORT
                    scores[dependent].reasons.append("reverse_import")
    
    # Add hub bonus for highly-connected files
    for file_path, score_obj in scores.items():
        import_count = len(import_graph.get(file_path, []))
        dependent_count = len(reverse_graph.get(file_path, []))
        total_connections = import_count + dependent_count
        
        if total_connections >= 5:
            hub_bonus = min(total_connections * SCORE_HUB_BONUS, 50)  # Cap at 50
            score_obj.score += hub_bonus
            score_obj.reasons.append(f"hub({total_connections})")
    
    return scores


def optimize_context(
    candidate_files: List[str],
    import_graph: Dict[str, List[str]],
    token_map: Dict[str, int],
    max_tokens: int,
    seed_files: Optional[List[str]] = None,
    reserve_tokens: int = 500  # Reserve for prompt overhead
) -> Tuple[List[str], PruningReport]:
    """
    Optimize file selection to fit within token budget using Graph-Knapsack algorithm.
    
    Args:
        candidate_files: List of files that could be included.
        import_graph: {file_path: [imported_files]}
        token_map: {file_path: token_count}
        max_tokens: Maximum token budget for context.
        seed_files: Files that must be included (entry points). If None, uses heuristics.
        reserve_tokens: Tokens to reserve for prompt overhead.
        
    Returns:
        Tuple of (final_file_list, pruning_report)
    """
    effective_budget = max_tokens - reserve_tokens
    
    # If no seed files specified, try to identify entry points
    if not seed_files:
        seed_files = _identify_entry_points(candidate_files, import_graph)
    
    # Ensure seed files are in candidate list
    seed_files = [f for f in seed_files if f in candidate_files or f in token_map]
    
    # Calculate scores for all files
    scores = calculate_file_scores(seed_files, import_graph, token_map, candidate_files)
    
    # Sort by score (descending), then by tokens (ascending for tie-breaking)
    sorted_files = sorted(
        scores.values(),
        key=lambda x: (-x.score, x.tokens)
    )
    
    # Knapsack selection
    included: List[FileScore] = []
    pruned: List[Tuple[FileScore, str]] = []  # (file, reason)
    current_tokens = 0
    
    for file_score in sorted_files:
        if file_score.tokens == 0:
            # Skip files with no content
            continue
            
        if current_tokens + file_score.tokens <= effective_budget:
            # File fits - include it
            included.append(file_score)
            current_tokens += file_score.tokens
        else:
            # File doesn't fit
            if file_score.score >= SCORE_SEED_FILE:
                # Seed file doesn't fit - this is a problem
                # Include it anyway but log warning
                logger.warning(f"Seed file {file_score.file_path} exceeds budget, including anyway")
                included.append(file_score)
                current_tokens += file_score.tokens
                pruned.append((file_score, "budget_exceeded_but_required"))
            else:
                # Regular file - prune it
                pruned.append((file_score, "budget_exceeded"))
    
    # Also add files that weren't scored at all (unrelated to seeds)
    unscored_files = [f for f in candidate_files if f not in scores]
    for file_path in unscored_files:
        tokens = token_map.get(file_path, 0)
        pruned.append((
            FileScore(file_path=file_path, tokens=tokens, score=0, reasons=["unrelated"]),
            "unrelated_to_seed"
        ))
    
    # Build final file list
    final_files = [f.file_path for f in included]
    
    # Build pruning report
    included_dicts = [
        {
            "file_path": f.file_path,
            "tokens": f.tokens,
            "score": f.score,
            "reasons": f.reasons
        }
        for f in included
    ]
    
    pruned_dicts = [
        {
            "file_path": f.file_path,
            "tokens": f.tokens,
            "score": f.score,
            "reason": reason
        }
        for f, reason in pruned
    ]
    
    report = PruningReport(
        included_files=included_dicts,
        pruned_files=pruned_dicts,
        total_tokens_included=current_tokens,
        total_tokens_pruned=sum(f.tokens for f, _ in pruned),
        token_budget=max_tokens,
        seed_files=seed_files
    )
    
    logger.info(
        f"Context optimization: {len(final_files)} files included "
        f"({current_tokens:,} tokens), {len(pruned)} files pruned "
        f"({report.total_tokens_pruned:,} tokens)"
    )
    
    return final_files, report


def _identify_entry_points(
    files: List[str],
    import_graph: Dict[str, List[str]]
) -> List[str]:
    """
    Heuristically identify entry point files when none are specified.
    
    Looks for:
    - index.ts/js, main.py, app.py, etc.
    - Files with many dependents but few imports (likely entry points)
    """
    entry_patterns = [
        "index.ts", "index.js", "index.tsx", "index.jsx",
        "main.py", "app.py", "__init__.py", "__main__.py",
        "main.ts", "main.js", "app.ts", "app.js",
        "src/index", "src/main", "src/app",
        "lib/index", "lib/main"
    ]
    
    entry_points = []
    
    # Check for pattern matches
    for file_path in files:
        file_lower = file_path.lower()
        for pattern in entry_patterns:
            if pattern in file_lower:
                entry_points.append(file_path)
                break
    
    # If no pattern matches, find files with high in-degree (many dependents)
    if not entry_points:
        reverse_graph = build_reverse_graph(import_graph)
        
        # Calculate in-degree for each file
        in_degrees = {f: len(reverse_graph.get(f, [])) for f in files}
        
        # Get top 3 files by in-degree
        sorted_by_indegree = sorted(in_degrees.items(), key=lambda x: -x[1])
        entry_points = [f for f, _ in sorted_by_indegree[:3] if in_degrees[f] > 0]
    
    return entry_points[:5]  # Limit to 5 entry points


def get_context_window_presets() -> Dict[str, int]:
    """
    Return predefined context window sizes for common LLM models.
    """
    return {
        "gpt-4": 8_192,
        "gpt-4-32k": 32_768,
        "gpt-4-turbo": 128_000,
        "claude-3-haiku": 200_000,
        "claude-3-sonnet": 200_000,
        "claude-3-opus": 200_000,
        "gemini-1.5-pro": 1_000_000,
        "gemini-1.5-flash": 1_000_000,
        "unlimited": 999_999_999,
    }


def suggest_context_limit(total_tokens: int) -> Tuple[str, int]:
    """
    Suggest an appropriate context window based on total repository tokens.
    
    Args:
        total_tokens: Total tokens in the repository.
        
    Returns:
        Tuple of (suggested_model, token_limit)
    """
    presets = get_context_window_presets()
    
    if total_tokens <= 6_000:
        return "gpt-4", presets["gpt-4"]
    elif total_tokens <= 25_000:
        return "gpt-4-32k", presets["gpt-4-32k"]
    elif total_tokens <= 100_000:
        return "gpt-4-turbo", presets["gpt-4-turbo"]
    elif total_tokens <= 500_000:
        return "claude-3-sonnet", presets["claude-3-sonnet"]
    else:
        return "gemini-1.5-pro", presets["gemini-1.5-pro"]
