"""
TOON v2 Formatter Module

Converts RepoSynth analysis artifacts (name_registry.json, import_graph.json)
into TOON (Tabular Object Oriented Notation) v2 format for LLM consumption.

TOON Format Specification:
- Compact, tabular representation of structured data
- Format: name[count]{header1,header2}: \n  val1,val2
- Optimized for token efficiency while maintaining readability
"""

from typing import Dict, List, Any


def format_table(name: str, headers: List[str], rows: List[List[str]]) -> str:
    """
    Generate TOON v2 Tabular Array.

    Format:
        name[count]{header1,header2}:
          val1,val2
          val3,val4

    Args:
        name: Table name (e.g., "symbols", "dependencies")
        headers: Column headers
        rows: Data rows (must match header count)

    Returns:
        Formatted TOON string

    Example:
        >>> format_table("symbols", ["name", "kind"], [["User", "class"], ["auth", "function"]])
        symbols[2]{name,kind}:
          User,class
          auth,function
    """
    if not rows:
        return f"{name}[0]{{{','.join(headers)}}}:\n  (empty)\n"

    # 1. Clean data (remove newlines/commas from values to prevent CSV break)
    clean_rows = []
    for row in rows:
        clean_row = []
        for cell in row:
            s = str(cell).replace('\n', ' ').replace(',', ';')
            # Truncate extremely long values to save tokens
            if len(s) > 50:
                s = s[:47] + "..."
            clean_row.append(s)
        clean_rows.append(clean_row)

    # 2. Construct Header
    header_str = f"{name}[{len(rows)}]{{{','.join(headers)}}}:"

    # 3. Construct Body
    body_str = "\n".join(["  " + ",".join(r) for r in clean_rows])

    return f"{header_str}\n{body_str}\n"


def convert_registry_to_toon(registry: Dict[str, Any]) -> str:
    """
    Convert name_registry.json to TOON format.

    The name registry contains all symbols (functions, classes, variables, etc.)
    found in the codebase with their metadata.

    Args:
        registry: name_registry.json as dict

    Returns:
        TOON formatted string

    Example Input:
        {
          "src/auth.ts:User": {
            "name": "User",
            "kind": "class",
            "file_path": "src/auth.ts",
            "is_public": true,
            "inherits_from": ["BaseModel"]
          }
        }

    Example Output:
        symbols[1]{name,kind,file,public,inherits}:
          User,class,src/auth.ts,yes,BaseModel
    """
    headers = ["name", "kind", "file", "public", "inherits"]
    rows = []

    for key, data in registry.items():
        # Extract symbol name from FQN (format: "file:name")
        symbol_name = data.get('name', key.split(':')[-1] if ':' in key else key)

        rows.append([
            symbol_name,
            data.get('kind', 'unknown'),
            data.get('file_path', ''),
            "yes" if data.get('is_public', False) else "no",
            "|".join(data.get('inherits_from', [])) if data.get('inherits_from') else ""
        ])

    return format_table("symbols", headers, rows)


def convert_graph_to_toon(graph: Dict[str, List[str]]) -> str:
    """
    Convert import_graph.json to TOON format.

    The import graph shows file-to-file dependencies.

    Args:
        graph: import_graph.json as dict where keys are source files
               and values are lists of imported files

    Returns:
        TOON formatted string

    Example Input:
        {
          "src/auth.ts": ["src/user.ts", "src/db.ts"],
          "src/user.ts": ["src/db.ts"]
        }

    Example Output:
        dependencies[3]{source,target}:
          src/auth.ts,src/user.ts
          src/auth.ts,src/db.ts
          src/user.ts,src/db.ts
    """
    headers = ["source", "target"]
    rows = []

    for source, targets in graph.items():
        if isinstance(targets, list):
            for target in targets:
                rows.append([source, target])
        else:
            # Handle old format where targets might be a string
            rows.append([source, str(targets)])

    return format_table("dependencies", headers, rows)


def convert_complexity_to_toon(complexity_data: List[Dict[str, Any]]) -> str:
    """
    Convert complexity metrics to TOON format.

    Args:
        complexity_data: List of complexity metrics from analysis.sqlite

    Returns:
        TOON formatted string

    Example Input:
        [
          {"symbol_name": "authenticate", "file_path": "src/auth.ts", "complexity": 15},
          {"symbol_name": "validateUser", "file_path": "src/user.ts", "complexity": 8}
        ]

    Example Output:
        complexity[2]{symbol,file,score}:
          authenticate,src/auth.ts,15
          validateUser,src/user.ts,8
    """
    headers = ["symbol", "file", "score"]
    rows = []

    for item in complexity_data:
        rows.append([
            item.get('symbol_name', 'unknown'),
            item.get('file_path', ''),
            str(item.get('complexity', 0))
        ])

    return format_table("complexity", headers, rows)


def generate_toon_blueprint(registry: Dict[str, Any], graph: Dict[str, List[str]]) -> str:
    """
    Generate a complete TOON blueprint (structure only, no source code).

    This is the foundation for all three modes: Blueprint, Focus, and Bundle.

    Args:
        registry: name_registry.json as dict
        graph: import_graph.json as dict

    Returns:
        Complete TOON formatted string with symbols and dependencies
    """
    sections = []

    sections.append("# CODEBASE STRUCTURE (TOON v2)\n")
    sections.append("This is a structural map of the codebase.\n")
    sections.append("Use this to understand architecture, dependencies, and symbol locations.\n\n")

    # Add symbol registry
    sections.append("## Symbol Registry\n")
    sections.append(convert_registry_to_toon(registry))
    sections.append("\n")

    # Add dependency graph
    sections.append("## Import Dependencies\n")
    sections.append(convert_graph_to_toon(graph))
    sections.append("\n")

    return "".join(sections)


# Utility functions for token estimation
def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (1 token ≈ 4 characters for English text).

    This is a simple heuristic. For precise counts, use tiktoken.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    # Rough heuristic: 1 token ≈ 4 chars for code/structured data
    return len(text) // 4


def summarize_toon(toon_text: str) -> Dict[str, Any]:
    """
    Generate summary statistics for a TOON document.

    Args:
        toon_text: TOON formatted text

    Returns:
        Dict with statistics (token count, table counts, etc.)
    """
    # Count tables by looking for pattern: name[count]{...}:
    import re

    table_pattern = r'(\w+)\[(\d+)\]'
    tables = re.findall(table_pattern, toon_text)

    table_stats = {}
    for name, count in tables:
        table_stats[name] = int(count)

    return {
        "estimated_tokens": estimate_tokens(toon_text),
        "character_count": len(toon_text),
        "line_count": toon_text.count('\n'),
        "tables": table_stats,
        "total_rows": sum(table_stats.values())
    }


if __name__ == "__main__":
    # Test the formatter with sample data
    print("=== TOON Formatter Test ===\n")

    # Sample registry
    sample_registry = {
        "src/auth.ts:User": {
            "name": "User",
            "kind": "class",
            "file_path": "src/auth.ts",
            "is_public": True,
            "inherits_from": ["BaseModel"]
        },
        "src/auth.ts:authenticate": {
            "name": "authenticate",
            "kind": "function",
            "file_path": "src/auth.ts",
            "is_public": True,
            "inherits_from": []
        }
    }

    # Sample graph
    sample_graph = {
        "src/auth.ts": ["src/user.ts", "src/db.ts"],
        "src/user.ts": ["src/db.ts"]
    }

    # Generate TOON
    blueprint = generate_toon_blueprint(sample_registry, sample_graph)
    print(blueprint)

    # Show statistics
    stats = summarize_toon(blueprint)
    print("\n=== Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
