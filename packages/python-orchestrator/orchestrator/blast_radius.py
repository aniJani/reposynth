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
import logging
from pathlib import Path
from typing import Dict, Set, List, Tuple, Any, Optional
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)


def load_ast_for_file(pack_dir: Path, file_path: str) -> List[Dict[str, Any]]:
    """
    Load the parsed AST for a specific file from ast_raw/*.jsonl

    Returns a list of AST nodes in the same format as the JSONL file.
    Returns empty list if file not found or parsing fails.
    """
    try:
        # Convert file path to AST filename (e.g., "orchestrator/api.py" -> "orchestrator_api_py.jsonl")
        ast_filename = file_path.replace('/', '_').replace('\\', '_').replace('.', '_') + '.jsonl'
        ast_file = pack_dir / "ast_raw" / ast_filename

        if not ast_file.exists():
            logger.debug(f"AST file not found: {ast_file}")
            return []

        nodes = []
        with open(ast_file, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        nodes.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON at {ast_file}:{line_num}: {e}")
                        continue

        return nodes

    except Exception as e:
        logger.error(f"Error loading AST for {file_path}: {e}", exc_info=True)
        return []


def build_ast_tree(nodes: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Convert flat JSONL nodes into a tree structure for easier traversal.

    Returns a dict mapping node_id -> node_with_children_expanded
    """
    try:
        tree = {}
        for node in nodes:
            if 'id' not in node:
                logger.warning(f"AST node missing 'id' field: {node}")
                continue
            tree[node['id']] = node

        # Expand children references
        for node_id, node in tree.items():
            if 'children' in node:
                node['children_nodes'] = [
                    tree[child_id]
                    for child_id in node.get('children', [])
                    if child_id in tree
                ]

        return tree

    except Exception as e:
        logger.error(f"Error building AST tree: {e}", exc_info=True)
        return {}


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
    Returns original source or error message if skeleton generation fails.
    """
    try:
        if not ast_nodes:
            # Fallback to original source if no AST available
            logger.debug("No AST nodes available, returning original source")
            return source_code

        tree = build_ast_tree(ast_nodes)
        root = tree.get(0)

        if not root:
            logger.warning("AST tree has no root node, returning original source")
            return source_code

        skeleton_parts = []

    except Exception as e:
        logger.error(f"Error initializing skeleton generation: {e}", exc_info=True)
        return f"# Error generating skeleton: {e}\n\n{source_code}"

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

    try:
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
                    if 0 <= start < len(source_code) and 0 <= end <= len(source_code):
                        skeleton_parts.append((start, source_code[start:end]))

            skeleton_parts.sort(key=lambda x: x[0])
            skeleton = '\n'.join(part[1] for part in skeleton_parts)

        return skeleton if skeleton.strip() else "# No public interfaces found"

    except Exception as e:
        logger.error(f"Error during skeleton traversal: {e}", exc_info=True)
        return f"# Error generating skeleton: {e}\n\n{source_code}"


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
    """
    Safely read file content with comprehensive error handling.

    Returns file content or error message if reading fails.
    """
    try:
        full_path = repo_path / file_rel_path

        if not full_path.exists():
            logger.warning(f"File not found: {full_path}")
            return f"# File not found: {file_rel_path}"

        if not full_path.is_file():
            logger.warning(f"Path is not a file: {full_path}")
            return f"# Not a file: {file_rel_path}"

        # Try reading with UTF-8, replace errors
        content = full_path.read_text(encoding='utf-8', errors='replace')
        return content

    except PermissionError as e:
        logger.error(f"Permission denied reading {file_rel_path}: {e}")
        return f"# Permission denied: {file_rel_path}"
    except OSError as e:
        logger.error(f"OS error reading {file_rel_path}: {e}")
        return f"# Error reading file: {e}"
    except Exception as e:
        logger.error(f"Unexpected error reading {file_rel_path}: {e}", exc_info=True)
        return f"# Error reading file: {e}"


def load_complexity_data(pack_dir: Path) -> Dict[str, int]:
    """
    Load complexity data from analysis.sqlite.

    Returns dict mapping file_path -> max_complexity.
    Returns empty dict if database doesn't exist or loading fails.
    """
    import sqlite3

    db_path = pack_dir / "analysis.sqlite"

    if not db_path.exists():
        logger.debug(f"Complexity database not found at {db_path}")
        return {}

    complexity_by_file = {}

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get max complexity per file
        cursor.execute("""
            SELECT file_path, MAX(complexity) as max_complexity
            FROM complexity
            GROUP BY file_path
        """)

        for row in cursor.fetchall():
            file_path, max_complexity = row
            if max_complexity is not None:
                complexity_by_file[file_path] = max_complexity

        conn.close()
        logger.info(f"Loaded complexity data for {len(complexity_by_file)} files")

    except sqlite3.Error as e:
        logger.warning(f"SQLite error loading complexity data: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading complexity data: {e}", exc_info=True)

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
    max_shockwave_files: int = 50,
    use_semantic_search: bool = True
) -> str:
    """
    Main entry point for Blast Radius Mode.

    Args:
        repo_path: Path to the repository root
        pack_dir: Path to the pack directory (contains import_graph.json, etc.)
        query: User's query to find relevant files
        max_shockwave_files: Maximum number of dependency files to include
        use_semantic_search: If True, use embeddings; if False, use keyword matching

    Returns:
        Markdown-formatted context ready for LLM consumption
    """
    logger.info(f"💥 Computing Blast Radius for query: '{query}'")
    print(f"💥 Computing Blast Radius for query: '{query}'")

    # Validate inputs (defensive programming)
    if not query or not query.strip():
        error_msg = "Error: Query cannot be empty or whitespace only"
        logger.error(error_msg)
        return error_msg

    if max_shockwave_files < 1:
        error_msg = f"Error: max_shockwave_files must be at least 1 (got {max_shockwave_files})"
        logger.error(error_msg)
        return error_msg

    try:
        # 1. Load intelligence
        import_graph_path = pack_dir / "import_graph.json"
        name_registry_path = pack_dir / "name_registry.json"

        if not import_graph_path.exists():
            error_msg = f"Error: import_graph.json not found at {import_graph_path}. Run with --build-graphs first."
            logger.error(error_msg)
            return error_msg

        if not name_registry_path.exists():
            error_msg = f"Error: name_registry.json not found at {name_registry_path}. Run with --build-graphs first."
            logger.error(error_msg)
            return error_msg

        try:
            with open(import_graph_path, 'r', encoding='utf-8') as f:
                import_graph = json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"Error: Invalid JSON in import_graph.json: {e}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error loading import_graph.json: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg

        try:
            with open(name_registry_path, 'r', encoding='utf-8') as f:
                name_registry = json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"Error: Invalid JSON in name_registry.json: {e}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error loading name_registry.json: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg

        logger.info(f"Loaded {len(import_graph)} files from import graph")
        logger.info(f"Loaded {len(name_registry)} symbols from name registry")

        # 2. Find the epicenter (seed files) using semantic or keyword search
        try:
            if use_semantic_search:
                from .retriever import retrieve_relevant_files_semantic
                logger.info("Using semantic search with embeddings")
                ranked_files = retrieve_relevant_files_semantic(
                    query, pack_dir, name_registry, max_files=10, fallback_to_keyword=True
                )
            else:
                from .retriever import retrieve_relevant_files
                logger.info("Using keyword-based search")
                ranked_files = retrieve_relevant_files(query, name_registry, max_files=10)

        except Exception as e:
            logger.error(f"Error during file retrieval: {e}", exc_info=True)
            # Fallback to keyword search
            from .retriever import retrieve_relevant_files
            logger.info("Falling back to keyword-based search due to error")
            ranked_files = retrieve_relevant_files(query, name_registry, max_files=10)

        if not ranked_files:
            error_msg = f"No files found matching query: '{query}'. Try a different search term."
            logger.warning(error_msg)
            return error_msg

        epicenter_files = set(f['file_path'] for f in ranked_files)

        logger.info(f"🎯 Epicenter detected: {len(epicenter_files)} files")
        print(f"🎯 Epicenter detected: {len(epicenter_files)} files")
        for file_data in ranked_files:
            file_path = file_data['file_path']
            score = file_data['score']
            match_count = file_data['match_count']
            print(f"   • {file_path} (score: {score:.2f}, {match_count} matching symbols)")

    except Exception as e:
        error_msg = f"Fatal error during blast radius initialization: {e}"
        logger.error(error_msg, exc_info=True)
        return f"Error: {error_msg}"

    try:
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

        logger.info(f"📡 Blast Radius: {len(shockwave_files)} dependency files")
        print(f"📡 Blast Radius: {len(shockwave_files)} dependency files")

    except Exception as e:
        error_msg = f"Error during dependency analysis: {e}"
        logger.error(error_msg, exc_info=True)
        return f"Error: {error_msg}"

    try:
        # 6. Generate the output
        output = []

        # Header
        output.append(f"# 🚀 VibeCode Context: {query}\n")
        output.append(f"**Epicenter:** {len(epicenter_files)} files (Full Source)")
        output.append(f"**Blast Radius:** {len(shockwave_files)} files (Interfaces Only)")
        output.append(f"**Total Context:** {len(epicenter_files) + len(shockwave_files)} files\n")

        # Dependency map
        try:
            dep_map = generate_dependency_map(epicenter_files, shockwave_files, import_graph)
            output.append(dep_map)
        except Exception as e:
            logger.warning(f"Error generating dependency map: {e}")
            output.append("# Dependency map generation failed\n")

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
            try:
                content = get_file_content(repo_path, rel_path)
                ext = Path(rel_path).suffix.replace('.', '') or 'txt'

                output.append(f"\n### File: `{rel_path}`\n")
                output.append(f"```{ext}")
                output.append(content)
                output.append("```\n")
            except Exception as e:
                logger.error(f"Error processing epicenter file {rel_path}: {e}")
                output.append(f"\n### File: `{rel_path}`\n")
                output.append(f"```\n# Error loading file: {e}\n```\n")

        # Context files (skeletons)
        if shockwave_files:
            output.append("\n---\n")
            output.append("## 🟡 CONTEXT FILES (Read-Only Interfaces)\n")
            output.append("These files are dependencies. Use them to understand types and interfaces.\n")
            output.append("**DO NOT modify these files.** They are shown as skeletons (signatures only).\n")

            for rel_path in sorted(shockwave_files):
                try:
                    content = get_file_content(repo_path, rel_path)
                    ext = Path(rel_path).suffix.replace('.', '') or 'txt'

                    # Generate skeleton using AST
                    ast_nodes = load_ast_for_file(pack_dir, rel_path)
                    skeleton = generate_skeleton_from_ast(ast_nodes, content, ext)

                    output.append(f"\n### File: `{rel_path}`\n")
                    output.append(f"```{ext}")
                    output.append(skeleton)
                    output.append("```\n")
                except Exception as e:
                    logger.error(f"Error processing shockwave file {rel_path}: {e}")
                    output.append(f"\n### File: `{rel_path}`\n")
                    output.append(f"```\n# Error loading file: {e}\n```\n")

        logger.info("Blast radius calculation completed successfully")
        return '\n'.join(output)

    except Exception as e:
        error_msg = f"Error generating output: {e}"
        logger.error(error_msg, exc_info=True)
        return f"Error: {error_msg}"
