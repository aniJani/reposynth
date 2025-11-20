"""
Simple retrieval module for selecting relevant context from name_registry.

This is the MINIMAL FIX to reduce token usage from 95K to ~5K.
No fancy embeddings, no graph traversal - just dead simple keyword matching.
"""

import json
from typing import Dict, List, Any, Tuple
from pathlib import Path
from collections import defaultdict


def retrieve_relevant_symbols(query: str, name_registry: Dict[str, Any], max_items: int = 5) -> List[Dict[str, Any]]:
    """
    Dead simple keyword-based retrieval.

    Args:
        query: The question/prompt from the user
        name_registry: The full name registry dict
        max_items: Maximum number of symbols to return

    Returns:
        List of relevant symbol dicts with their metadata

    Example:
        query = "What does ArraySchema inherit from?"
        -> Returns top 5 symbols matching "ArraySchema" or "inherit"
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())

    matches = []
    for symbol_fqn, symbol_data in name_registry.items():
        # Extract symbol name from FQN (file:name format)
        symbol_name = symbol_fqn.split(':')[-1].lower()

        # Simple scoring: count how many query words appear in symbol name
        score = 0
        for word in query_words:
            if word in symbol_name:
                score += 2  # Exact word match
            elif word in symbol_fqn.lower():
                score += 1  # Match in file path

        if score > 0:
            # Include all symbol data plus the FQN
            match_data = {"fqn": symbol_fqn, **symbol_data}
            matches.append((score, match_data))

    # Sort by score (descending) and return top-k
    matches.sort(reverse=True, key=lambda x: x[0])
    return [match[1] for match in matches[:max_items]]


def retrieve_relevant_files(query: str, name_registry: Dict[str, Any], max_files: int = 10) -> List[Dict[str, Any]]:
    """
    Enhanced retrieval with file-level aggregation and fuzzy matching.

    Designed for blast-radius mode where we need files, not individual symbols.

    Improvements over retrieve_relevant_symbols:
    1. Aggregates scores at FILE level (files with multiple matches rank higher)
    2. Fuzzy substring matching (handles "authentication" matching "authenticate")
    3. Returns files directly with aggregate scores

    Args:
        query: Natural language query (e.g., "add google oauth to login")
        name_registry: Full name registry
        max_files: Maximum number of files to return

    Returns:
        List of dicts with file metadata and aggregate scores
        [{"file_path": "auth/service.py", "score": 12, "matching_symbols": [...]}, ...]

    Example:
        query = "login authentication user"

        File scoring:
        - auth/service.py: login(), authenticate(), validateUser() -> score = 6
        - models/user.py: User class -> score = 2
        - utils/helpers.py: login_url constant -> score = 1

        Returns: [auth/service.py, models/user.py] (top 2)
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())

    # File-level aggregation: {file_path: {"score": int, "symbols": [...]}}
    file_data = defaultdict(lambda: {"score": 0, "symbols": []})

    for symbol_fqn, symbol_info in name_registry.items():
        file_path = symbol_info.get('file_path', '')
        if not file_path:
            continue

        symbol_name = symbol_fqn.split(':')[-1].lower()

        # Calculate match score for this symbol
        score = 0

        for word in query_words:
            # Exact word match in symbol name (highest priority)
            if word == symbol_name:
                score += 3
            # Fuzzy substring match in symbol name (handles variations)
            elif word in symbol_name or symbol_name in word:
                score += 2
            # Match in file path (medium priority)
            elif word in symbol_fqn.lower():
                score += 1

        if score > 0:
            file_data[file_path]["score"] += score
            file_data[file_path]["symbols"].append({
                "name": symbol_name,
                "fqn": symbol_fqn,
                "score": score,
                **symbol_info
            })

    # Convert to list and sort by aggregate score
    ranked_files = []
    for file_path, data in file_data.items():
        ranked_files.append({
            "file_path": file_path,
            "score": data["score"],
            "matching_symbols": data["symbols"],
            "match_count": len(data["symbols"])
        })

    ranked_files.sort(reverse=True, key=lambda x: x["score"])
    return ranked_files[:max_files]


def retrieve_with_relationships(query: str, name_registry: Dict[str, Any], max_items: int = 5) -> Dict[str, Any]:
    """
    Enhanced retrieval that also pulls in related symbols.

    If query asks about inheritance/exports, automatically include related info.

    Args:
        query: The question/prompt
        name_registry: Full registry
        max_items: Max symbols to return

    Returns:
        Dict with 'primary_matches' and 'related_symbols'
    """
    query_lower = query.lower()

    # Get primary matches
    primary = retrieve_relevant_symbols(query, name_registry, max_items)

    related = []

    # If query asks about inheritance, include parent classes
    if any(word in query_lower for word in ['inherit', 'extends', 'base', 'parent']):
        for symbol in primary:
            if 'inherits_from' in symbol and symbol['inherits_from']:
                # Find parent class definitions
                for parent_name in symbol['inherits_from']:
                    for fqn, data in name_registry.items():
                        if fqn.endswith(f":{parent_name}"):
                            related.append({"fqn": fqn, **data})
                            break

    # If query asks about exports, include export info
    if any(word in query_lower for word in ['export', 'exported', 'from', 'where']):
        for symbol in primary:
            if 'exported_from' in symbol and symbol['exported_from']:
                # Export info is already in the symbol, no need to fetch more
                pass

    return {
        "primary_matches": primary,
        "related_symbols": related
    }


def format_context_for_llm(matches: List[Dict[str, Any]], include_code: bool = False) -> str:
    """
    Format retrieved symbols into a compact string for LLM context.

    Args:
        matches: List of symbol dicts from retrieve_relevant_symbols
        include_code: Whether to include full source code (expensive!)

    Returns:
        Formatted string ready to append to LLM prompt
    """
    lines = []
    lines.append(f"# Relevant Code Symbols ({len(matches)} items)\n")

    for i, symbol in enumerate(matches, 1):
        lines.append(f"\n## {i}. {symbol.get('fqn', 'Unknown')}")
        lines.append(f"- **Kind**: {symbol.get('kind', 'unknown')}")
        lines.append(f"- **File**: {symbol.get('file_path', 'unknown')}")
        lines.append(f"- **Public**: {symbol.get('is_public', False)}")

        if 'inherits_from' in symbol and symbol['inherits_from']:
            lines.append(f"- **Inherits From**: {', '.join(symbol['inherits_from'])}")

        if 'exported_from' in symbol and symbol['exported_from']:
            lines.append(f"- **Exported From**: {', '.join(symbol['exported_from'])}")

        if include_code and 'start_byte' in symbol and 'end_byte' in symbol:
            lines.append(f"- **Location**: bytes {symbol['start_byte']}-{symbol['end_byte']}")

    return "\n".join(lines)


def load_and_retrieve(query: str, pack_dir: str, max_items: int = 5) -> str:
    """
    Convenience function to load name_registry and retrieve in one step.

    Args:
        query: The question
        pack_dir: Path to pack directory containing name_registry.json
        max_items: Max symbols to return

    Returns:
        Formatted context string ready for LLM
    """
    pack_path = Path(pack_dir)
    registry_path = pack_path / "name_registry.json"

    if not registry_path.exists():
        raise FileNotFoundError(f"name_registry.json not found at {registry_path}")

    with open(registry_path, 'r', encoding='utf-8') as f:
        name_registry = json.load(f)

    matches = retrieve_relevant_symbols(query, name_registry, max_items)
    return format_context_for_llm(matches)


# Example usage:
if __name__ == "__main__":
    # Test the retrieval function
    query = "What does ArraySchema inherit from?"
    context = load_and_retrieve(query, "../../pack", max_items=5)
    print(context)
    print(f"\n\nEstimated tokens: ~{len(context.split()) * 1.3:.0f}")
