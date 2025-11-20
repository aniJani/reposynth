"""
Blast Radius Mode - "Just Right" Context for VibeCoders

This module implements intelligent context gathering that provides:
- Full source code for files being modified (epicenter)
- Skeleton interfaces for dependencies (shockwave)
- Smart pruning based on import frequency and complexity

The goal: Give LLMs exactly what they need to write correct code without
hallucinations or token budget explosions.
"""

import json
from pathlib import Path
from typing import Dict, Set, List, Tuple, Any
from collections import defaultdict


def load_ast_for_file(pack_dir: Path, file_path: str) -> List[Dict[str, Any]]:
    """
    Load the parsed AST for a specific file from ast_raw/*.jsonl

    Returns a list of AST nodes in the same format as the JSONL file.
    """
    # Convert file path to AST filename (e.g., "orchestrator/api.py" -> "orchestrator_api_py.jsonl")
    ast_filename = file_path.replace('/', '_').replace('\\', '_').replace('.', '_') + '.jsonl'
    ast_file = pack_dir / "ast_raw" / ast_filename

    if not ast_file.exists():
        return []

    nodes = []
    with open(ast_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                nodes.append(json.loads(line))

    return nodes


def build_ast_tree(nodes: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Convert flat JSONL nodes into a tree structure for easier traversal.

    Returns a dict mapping node_id -> node_with_children_expanded
    """
    tree = {}
    for node in nodes:
        tree[node['id']] = node

    # Expand children references
    for node_id, node in tree.items():
        if 'children' in node:
            node['children_nodes'] = [tree[child_id] for child_id in node.get('children', []) if child_id in tree]

    return tree


def extract_source_span(source_code: str, start_byte: int, end_byte: int) -> str:
    """
    Extract a span of source code by byte positions.
    """
    return source_code[start_byte:end_byte]


def generate_skeleton_from_ast(
    ast_nodes: List[Dict[str, Any]],
    source_code: str,
    file_extension: str
) -> str:
    """
    Generate a skeleton from AST that preserves:
    - Type definitions and interfaces (fully intact)
    - Function/class signatures (no bodies)
    - Import statements
    - Exports

    Implementation uses Tree-sitter AST for precision.
    """
    if not ast_nodes:
        # Fallback to original source if no AST available
        return source_code

    tree = build_ast_tree(ast_nodes)
    root = tree.get(0)

    if not root:
        return source_code

    skeleton_parts = []

    def should_keep_node_fully(node: Dict[str, Any]) -> bool:
        """Determine if a node should be kept with full content."""
        kind = node.get('kind', '')

        # Keep type-related nodes fully (NOT class_definition - needs special handling)
        type_nodes = {
            'type_alias_declaration',  # TypeScript type alias
            'interface_declaration',    # TypeScript interface
            'type_definition',          # Generic type definition
            'typed_dict',              # Python TypedDict
            'import_statement',        # Keep imports
            'import_from_statement',   # Keep from imports
            'export_statement',        # Keep exports
        }

        return kind in type_nodes

    def should_keep_signature_only(node: Dict[str, Any]) -> bool:
        """Determine if a node should have body removed."""
        kind = node.get('kind', '')

        signature_nodes = {
            'function_definition',
            'function_declaration',
            'method_definition',
            'arrow_function',
        }

        return kind in signature_nodes

    def extract_signature(node: Dict[str, Any]) -> str:
        """
        Extract just the signature of a function/method.

        For Python: def foo(x: int) -> str:
        For JavaScript/TypeScript: function foo(x: number): string
        """
        start_byte = node.get('start_byte', 0)
        end_byte = node.get('end_byte', 0)

        # Get full node content
        full_content = source_code[start_byte:end_byte]

        # Find the body and remove it
        if file_extension in ['py']:
            # Python: find the colon, keep everything before it + ":"
            lines = full_content.split('\n')
            signature_line = lines[0] if lines else full_content

            # Keep decorators if present
            decorators = []
            for line in lines:
                if line.strip().startswith('@'):
                    decorators.append(line)
                elif line.strip().startswith('def ') or line.strip().startswith('async def '):
                    signature_line = line
                    break

            result = '\n'.join(decorators)
            if result:
                result += '\n'
            result += signature_line

            # Add ellipsis to indicate omitted body
            if not signature_line.rstrip().endswith(':'):
                result += ':'
            result += '\n    ...'

            return result

        elif file_extension in ['js', 'ts', 'jsx', 'tsx']:
            # JavaScript/TypeScript: keep everything until first {
            brace_pos = full_content.find('{')
            if brace_pos != -1:
                signature = full_content[:brace_pos].rstrip()
                return signature + ' { /* ... */ }'
            return full_content

        return full_content

    def traverse_node(node: Dict[str, Any], indent: int = 0) -> None:
        """
        Recursively traverse AST and build skeleton.
        """
        kind = node.get('kind', '')
        start_byte = node.get('start_byte', 0)
        end_byte = node.get('end_byte', 0)

        # Keep certain nodes fully
        if should_keep_node_fully(node):
            content = source_code[start_byte:end_byte]
            skeleton_parts.append((start_byte, content))

        # Keep signature only
        elif should_keep_signature_only(node):
            signature = extract_signature(node)
            skeleton_parts.append((start_byte, signature))

        # Recurse into children for module-level constructs and classes
        elif kind in ['module', 'program', 'source_file', 'class_definition', 'class_body', 'decorated_definition']:
            for child in node.get('children_nodes', []):
                traverse_node(child, indent)

    # Start traversal from root
    traverse_node(root)

    # Sort by byte position and join
    skeleton_parts.sort(key=lambda x: x[0])
    skeleton = '\n\n'.join(part[1] for part in skeleton_parts)

    # If skeleton is empty, include imports at minimum
    if not skeleton.strip():
        # Extract just import statements as fallback
        for node in ast_nodes:
            if node.get('kind', '') in ['import_statement', 'import_from_statement']:
                start = node.get('start_byte', 0)
                end = node.get('end_byte', 0)
                skeleton_parts.append((start, source_code[start:end]))

        skeleton_parts.sort(key=lambda x: x[0])
        skeleton = '\n'.join(part[1] for part in skeleton_parts)

    return skeleton if skeleton.strip() else "# No public interfaces found"


def calculate_file_scores(
    import_graph: Dict[str, List[str]],
    complexity_data: Dict[str, int]
) -> Dict[str, float]:
    """
    Score files based on:
    1. Import frequency (in-degree: how many files import this)
    2. Complexity (high complexity = higher impact)

    Returns {file_path: score} dict
    """
    # Build reverse graph for in-degree calculation
    reverse_graph = defaultdict(list)
    for file_path, imports in import_graph.items():
        for imported_file in imports:
            reverse_graph[imported_file].append(file_path)

    scores = {}
    for file_path in import_graph.keys():
        # Import frequency score (in-degree)
        in_degree = len(reverse_graph.get(file_path, []))

        # Complexity score
        max_complexity = complexity_data.get(file_path, 1)

        # Combined score: weighted sum
        # Higher weight on import frequency (0.7) vs complexity (0.3)
        score = (in_degree * 0.7) + (max_complexity * 0.3)
        scores[file_path] = score

    return scores


def traverse_dependencies(
    epicenter_files: Set[str],
    import_graph: Dict[str, List[str]],
    file_scores: Dict[str, float],
    max_files: int = 50
) -> Set[str]:
    """
    Traverse dependencies with unlimited depth but smart pruning.

    Algorithm:
    1. Start from epicenter files
    2. Traverse both downstream (what epicenter imports) and upstream (what imports epicenter)
    3. Score each encountered file
    4. Keep traversing until we hit max_files limit
    5. Return top-scored files

    Args:
        epicenter_files: The files being directly modified
        import_graph: Dict mapping file -> list of files it imports
        file_scores: Pre-calculated scores for each file
        max_files: Maximum number of files to include in shockwave

    Returns:
        Set of file paths in the shockwave
    """
    # Build reverse graph for upstream traversal
    reverse_graph = defaultdict(list)
    for file_path, imports in import_graph.items():
        for imported_file in imports:
            reverse_graph[imported_file].append(file_path)

    # BFS traversal with scoring
    visited = set()
    candidates = []  # List of (score, file_path)
    queue = list(epicenter_files)

    while queue:
        current_file = queue.pop(0)

        if current_file in visited:
            continue

        visited.add(current_file)

        # Skip if this is an epicenter file
        if current_file in epicenter_files:
            # Add its dependencies to queue
            downstream = import_graph.get(current_file, [])
            upstream = reverse_graph.get(current_file, [])
            queue.extend(downstream)
            queue.extend(upstream)
            continue

        # Score this file
        score = file_scores.get(current_file, 0)
        candidates.append((score, current_file))

        # Continue traversing from this file
        downstream = import_graph.get(current_file, [])
        upstream = reverse_graph.get(current_file, [])
        queue.extend(downstream)
        queue.extend(upstream)

    # Sort candidates by score (descending) and take top N
    candidates.sort(reverse=True, key=lambda x: x[0])
    shockwave_files = {file_path for score, file_path in candidates[:max_files]}

    return shockwave_files


def get_file_content(repo_path: Path, file_rel_path: str) -> str:
    """Safely read file content."""
    full_path = repo_path / file_rel_path
    if full_path.exists():
        try:
            return full_path.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            return f"# Error reading file: {e}"
    return "# File not found"


def load_complexity_data(pack_dir: Path) -> Dict[str, int]:
    """
    Load complexity data from analysis.sqlite.

    Returns dict mapping file_path -> max_complexity
    """
    import sqlite3

    db_path = pack_dir / "analysis.sqlite"
    if not db_path.exists():
        return {}

    complexity_by_file = {}

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get max complexity per file
        cursor.execute("""
            SELECT file_path, MAX(complexity) as max_complexity
            FROM complexity
            GROUP BY file_path
        """)

        for row in cursor.fetchall():
            file_path, max_complexity = row
            complexity_by_file[file_path] = max_complexity

        conn.close()
    except Exception as e:
        print(f"⚠️  Warning: Could not load complexity data: {e}")

    return complexity_by_file


def generate_dependency_map(
    epicenter_files: Set[str],
    shockwave_files: Set[str],
    import_graph: Dict[str, List[str]]
) -> str:
    """
    Generate a visual dependency topology using simple text tree.

    Shows how epicenter files relate to shockwave files.
    """
    lines = ["# Dependency Topology\n"]

    # Build reverse graph
    reverse_graph = defaultdict(list)
    for file_path, imports in import_graph.items():
        for imported_file in imports:
            reverse_graph[imported_file].append(file_path)

    for epicenter_file in sorted(epicenter_files):
        lines.append(f"**Epicenter:** `{epicenter_file}`")

        # Show what it imports (that's in shockwave)
        imports = [f for f in import_graph.get(epicenter_file, []) if f in shockwave_files]
        if imports:
            lines.append(f"  ├─ **Imports:** {len(imports)} files")
            for imp in sorted(imports)[:5]:  # Show first 5
                lines.append(f"  │  • `{imp}`")
            if len(imports) > 5:
                lines.append(f"  │  • ... and {len(imports) - 5} more")

        # Show what imports it (that's in shockwave)
        importers = [f for f in reverse_graph.get(epicenter_file, []) if f in shockwave_files]
        if importers:
            lines.append(f"  └─ **Imported by:** {len(importers)} files")
            for imp in sorted(importers)[:5]:  # Show first 5
                lines.append(f"     • `{imp}`")
            if len(importers) > 5:
                lines.append(f"     • ... and {len(importers) - 5} more")

        lines.append("")

    return '\n'.join(lines)


def calculate_blast_radius(
    repo_path: Path,
    pack_dir: Path,
    query: str,
    max_shockwave_files: int = 50
) -> str:
    """
    Main entry point for Blast Radius Mode.

    Args:
        repo_path: Path to the repository root
        pack_dir: Path to the pack directory (contains import_graph.json, etc.)
        query: User's query to find relevant files
        max_shockwave_files: Maximum number of dependency files to include

    Returns:
        Markdown-formatted context ready for LLM consumption
    """
    print(f"💥 Computing Blast Radius for query: '{query}'")

    # 1. Load intelligence
    import_graph_path = pack_dir / "import_graph.json"
    name_registry_path = pack_dir / "name_registry.json"

    if not import_graph_path.exists() or not name_registry_path.exists():
        return "Error: import_graph.json or name_registry.json not found. Run with --build-graphs first."

    with open(import_graph_path) as f:
        import_graph = json.load(f)

    with open(name_registry_path) as f:
        name_registry = json.load(f)

    # 2. Find the epicenter (seed files) using enhanced file-level retrieval
    from .retriever import retrieve_relevant_files

    ranked_files = retrieve_relevant_files(query, name_registry, max_files=10)

    if not ranked_files:
        return f"No files found matching query: '{query}'. Try a different search term."

    epicenter_files = set(f['file_path'] for f in ranked_files)

    print(f"🎯 Epicenter detected: {len(epicenter_files)} files")
    for file_data in ranked_files:
        file_path = file_data['file_path']
        score = file_data['score']
        match_count = file_data['match_count']
        print(f"   • {file_path} (score: {score}, {match_count} matching symbols)")

    # 3. Load complexity data for scoring
    complexity_data = load_complexity_data(pack_dir)

    # 4. Calculate file scores
    file_scores = calculate_file_scores(import_graph, complexity_data)

    # 5. Traverse dependencies with smart pruning
    shockwave_files = traverse_dependencies(
        epicenter_files,
        import_graph,
        file_scores,
        max_files=max_shockwave_files
    )

    print(f"📡 Blast Radius: {len(shockwave_files)} dependency files")

    # 6. Generate the output
    output = []

    # Header
    output.append(f"# 🚀 VibeCode Context: {query}\n")
    output.append(f"**Epicenter:** {len(epicenter_files)} files (Full Source)")
    output.append(f"**Blast Radius:** {len(shockwave_files)} files (Interfaces Only)")
    output.append(f"**Total Context:** {len(epicenter_files) + len(shockwave_files)} files\n")

    # Dependency map
    dep_map = generate_dependency_map(epicenter_files, shockwave_files, import_graph)
    output.append(dep_map)

    # System instructions
    output.append("\n---\n")
    output.append("## 🛑 System Instructions\n")
    output.append("You are coding in **Blast Radius** mode. You have:\n")
    output.append("- **Full source code** for PRIMARY FILES (these are being modified)")
    output.append("- **Skeleton interfaces** for CONTEXT FILES (read-only dependencies)\n")
    output.append("**Rules:**")
    output.append("- USE the skeleton interfaces to verify method signatures and types")
    output.append("- DO NOT hallucinate new methods on skeleton files")
    output.append("- DO NOT modify CONTEXT FILES (they're read-only)")
    output.append("- ONLY modify PRIMARY FILES\n")

    # Primary files (full source)
    output.append("\n---\n")
    output.append("## 🟢 PRIMARY FILES (Full Source)\n")
    output.append("These are the files you should modify to complete the task.\n")

    for rel_path in sorted(epicenter_files):
        content = get_file_content(repo_path, rel_path)
        ext = Path(rel_path).suffix.replace('.', '') or 'txt'

        output.append(f"\n### File: `{rel_path}`\n")
        output.append(f"```{ext}")
        output.append(content)
        output.append("```\n")

    # Context files (skeletons)
    if shockwave_files:
        output.append("\n---\n")
        output.append("## 🟡 CONTEXT FILES (Read-Only Interfaces)\n")
        output.append("These files are dependencies. Use them to understand types and interfaces.\n")
        output.append("**DO NOT modify these files.** They are shown as skeletons (signatures only).\n")

        for rel_path in sorted(shockwave_files):
            content = get_file_content(repo_path, rel_path)
            ext = Path(rel_path).suffix.replace('.', '') or 'txt'

            # Generate skeleton using AST
            ast_nodes = load_ast_for_file(pack_dir, rel_path)
            skeleton = generate_skeleton_from_ast(ast_nodes, content, ext)

            output.append(f"\n### File: `{rel_path}`\n")
            output.append(f"```{ext}")
            output.append(skeleton)
            output.append("```\n")

    return '\n'.join(output)
