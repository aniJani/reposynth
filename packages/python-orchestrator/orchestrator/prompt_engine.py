"""
Prompt Engine Module

Generates optimized "Vibe Coding" prompts for LLMs from RepoSynth pack artifacts.
Supports three compression modes:
- Blueprint Mode: Structure only (5-10K tokens)
- Focus Mode: Structure + relevant files based on query (20-50K tokens)
- Bundle Mode: Structure + dependency slice from entry point (50-200K+ tokens)
"""

import json
import zipfile
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .toon_formatter import generate_toon_blueprint, estimate_tokens, summarize_toon
from .retriever import retrieve_relevant_symbols, hybrid_search
from .context_optimizer import optimize_context, PruningReport, get_context_window_presets


class PromptEngine:
    """
    Assembles LLM-optimized prompts from RepoSynth pack artifacts.
    """

    def __init__(self, pack_path: Path, repo_url: Optional[str] = None):
        """
        Initialize the prompt engine with a pack directory or ZIP file.

        Args:
            pack_path: Path to pack directory or ZIP file
            repo_url: Optional URL of the repository (for fallback source reading)
        """
        self.pack_path = Path(pack_path)
        self.is_zip = pack_path.suffix == '.zip'
        self.is_json = pack_path.suffix == '.json'
        self.repo_url = repo_url
        self.temp_extract_dir = None

        # Load artifacts
        self._load_artifacts()

    def _load_artifacts(self):
        """Load artifacts from pack directory or ZIP file."""
        if self.is_zip:
            self._extract_zip()
            # Check if 'pack' folder exists inside the extracted zip
            if (self.temp_extract_dir / "pack").exists():
                work_dir = self.temp_extract_dir / "pack"
            else:
                work_dir = self.temp_extract_dir
        elif self.is_json:
            # Load from single JSON file
            self._load_from_json()
            return
        elif self.pack_path.suffix == '.toon':
            self._load_from_toon()
            return
        elif self.pack_path.is_dir():
            # Pack is a directory - check if it has essential files directly
            # or if there's a 'pack' subdirectory
            if (self.pack_path / "name_registry.json").exists():
                work_dir = self.pack_path
            elif (self.pack_path / "pack" / "name_registry.json").exists():
                work_dir = self.pack_path / "pack"
            else:
                work_dir = self.pack_path
        else:
            work_dir = self.pack_path

        # Load name_registry.json
        registry_path = work_dir / "name_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r', encoding='utf-8') as f:
                self.name_registry = json.load(f)
        else:
            print(f"Warning: name_registry.json not found at {registry_path}", file=sys.stderr)
            self.name_registry = {}

        # Load import_graph.json
        graph_path = work_dir / "import_graph.json"
        if graph_path.exists():
            try:
                with open(graph_path, 'r', encoding='utf-8') as f:
                    raw_graph = json.load(f)
                    self.import_graph = self._normalize_graph(raw_graph)
            except Exception as e:
                print(f"Error loading import_graph.json: {e}", file=sys.stderr)
                self.import_graph = {}
        else:
            print(f"Warning: import_graph.json not found at {graph_path}", file=sys.stderr)
            self.import_graph = {}

        # Load variable_registry.json (optional)
        var_registry_path = work_dir / "variable_registry.json"
        if var_registry_path.exists():
            with open(var_registry_path, 'r', encoding='utf-8') as f:
                self.variable_registry = json.load(f)
        else:
            self.variable_registry = {}

        # Load token_map.json (optional, for context optimization)
        token_map_path = work_dir / "token_map.json"
        if token_map_path.exists():
            with open(token_map_path, 'r', encoding='utf-8') as f:
                self.token_map = json.load(f)
        else:
            self.token_map = {}

        # Store work directory for file reading
        self.work_dir = work_dir

    def _load_from_json(self):
        """Load artifacts from a single JSON pack file."""
        with open(self.pack_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.name_registry = data.get('name_registry', {})
        self.import_graph = self._normalize_graph(data.get('import_graph', {}))
        self.variable_registry = data.get('variable_registry', {})
        self.token_map = data.get('token_map', {})
        
        # Store source files if available
        self.source_files = data.get('source_files', {})
        
        # For JSON packs, we don't have a work_dir with files
        # We might need to handle source code differently if it's not in the JSON
        self.work_dir = None
        self.json_data = data

    def _load_from_toon(self):
        """Load artifacts from a TOON pack file."""
        with open(self.pack_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Simple parsing of TOON format
        # This is a basic parser, a full parser would be more robust
        self.name_registry = {}
        self.import_graph = {}
        self.source_files = {}
        self.token_map = {}  # TOON packs don't include token map
        
        # Parse symbols
        import re
        # symbols[count]{headers}:
        #   row1
        symbol_section = re.search(r'symbols\[\d+\]\{([^}]+)\}:\n((?:  .*\n)+)', content)
        if symbol_section:
            headers = symbol_section.group(1).split(',')
            body = symbol_section.group(2)
            for line in body.strip().split('\n'):
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    name = parts[0]
                    kind = parts[1]
                    file_path = parts[2]
                    fqn = f"{file_path}:{name}"
                    self.name_registry[fqn] = {
                        "name": name,
                        "kind": kind,
                        "file_path": file_path,
                        "is_public": parts[3] == "yes" if len(parts) > 3 else False
                    }

        # Parse dependencies
        dep_section = re.search(r'dependencies\[\d+\]\{([^}]+)\}:\n((?:  .*\n)+)', content)
        if dep_section:
            body = dep_section.group(2)
            for line in body.strip().split('\n'):
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    source = parts[0]
                    target = parts[1]
                    if source not in self.import_graph: self.import_graph[source] = []
                    self.import_graph[source].append(target)

        # Parse source code
        # Try Markdown format first (v2.1)
        # ### File: path/to/file.ts
        # ```lang
        # content
        # ```
        source_pattern_md = r'### File: ([^\n]+)\n```[^\n]*\n(.*?)\n```'
        found_md = False
        for match in re.finditer(source_pattern_md, content, re.DOTALL):
            found_md = True
            file_path = match.group(1).strip()
            source_code = match.group(2)
            self.source_files[file_path] = {
                "full_source": source_code
            }
            
        # Fallback to legacy format (v2.0)
        # source[count]{file,lang}:
        #   file,lang
        #   content
        #   @@@
        if not found_md:
            source_pattern_legacy = r'  ([^,]+),([^\n]+)\n(.*?)\n  @@@'
            for match in re.finditer(source_pattern_legacy, content, re.DOTALL):
                file_path = match.group(1).strip()
                source_code = match.group(3)
                self.source_files[file_path] = {
                    "full_source": source_code
                }
            
        self.work_dir = None

    def _extract_zip(self):
        """Extract ZIP file to temporary directory."""
        import tempfile
        import shutil

        self.temp_extract_dir = Path(tempfile.mkdtemp(prefix="reposynth_prompt_"))

        with zipfile.ZipFile(self.pack_path, 'r') as zf:
            zf.extractall(self.temp_extract_dir)

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_extract_dir and self.temp_extract_dir.exists():
            import shutil
            shutil.rmtree(self.temp_extract_dir)

    def generate_blueprint(self) -> Dict[str, Any]:
        """
        Generate Blueprint Mode prompt (structure only, no source code).

        Returns:
            Dict with 'prompt' (str) and 'metadata' (dict)
        """
        # Generate TOON structure
        toon_structure = generate_toon_blueprint(self.name_registry, self.import_graph)

        # Assemble prompt
        prompt_parts = []
        prompt_parts.append("# CODEBASE BLUEPRINT\n")
        prompt_parts.append("You are analyzing a software project. Below is the structural blueprint.\n")
        prompt_parts.append("Use this to understand architecture, dependencies, and symbol locations.\n\n")
        prompt_parts.append(toon_structure)

        # Add usage instructions
        prompt_parts.append("\n## How to Use This Blueprint\n")
        prompt_parts.append("- **symbols** table shows all functions, classes, and variables\n")
        prompt_parts.append("- **dependencies** table shows file-to-file imports\n")
        prompt_parts.append("- Use this to navigate the codebase structure\n")
        prompt_parts.append("- For specific code, request a Focus or Bundle mode prompt\n")

        prompt = "".join(prompt_parts)

        # Generate metadata
        stats = summarize_toon(toon_structure)

        return {
            "prompt": prompt,
            "metadata": {
                "mode": "blueprint",
                "token_estimate": stats["estimated_tokens"],
                "symbol_count": stats["tables"].get("symbols", 0),
                "dependency_count": stats["tables"].get("dependencies", 0),
                "description": "Structure-only view (no source code)"
            }
        }

    def generate_focus(self, query: str, max_files: int = 5) -> Dict[str, Any]:
        """
        Generate Focus Mode prompt (structure + relevant files based on query).

        Args:
            query: User's question or task (e.g., "Fix login logic")
            max_files: Maximum number of files to include

        Returns:
            Dict with 'prompt' (str) and 'metadata' (dict)
        """
        # Start with blueprint
        toon_structure = generate_toon_blueprint(self.name_registry, self.import_graph)

        # Use hybrid_search to find relevant symbols/files
        # Pass pack_path (if it's a dir) or work_dir
        search_dir = self.work_dir if self.work_dir else self.pack_path
        matches = hybrid_search(query, search_dir, max_items=max_files, registry=self.name_registry)

        # Deduplicate files
        relevant_files = []
        seen_files = set()
        for match in matches:
            file_path = match.get('file_path')
            if file_path and file_path not in seen_files:
                relevant_files.append(file_path)
                seen_files.add(file_path)
                if len(relevant_files) >= max_files:
                    break

        # Assemble prompt
        prompt_parts = []
        prompt_parts.append(f"# CODEBASE CONTEXT: {query}\n\n")
        prompt_parts.append("## Structural Blueprint\n")
        prompt_parts.append(toon_structure)
        prompt_parts.append("\n")

        # Add source code for relevant files
        prompt_parts.append(f"## Relevant Source Files ({len(relevant_files)} files)\n\n")

        for file_path in relevant_files:
            source_code = self._read_source_file(file_path)
            if source_code:
                # Detect language from file extension
                lang = self._detect_language(file_path)
                prompt_parts.append(f"### File: {file_path}\n")
                prompt_parts.append(f"```{lang}\n")
                prompt_parts.append(source_code)
                prompt_parts.append("\n```\n\n")

        prompt = "".join(prompt_parts)

        return {
            "prompt": prompt,
            "metadata": {
                "mode": "focus",
                "token_estimate": estimate_tokens(prompt),
                "query": query,
                "files_included": len(relevant_files),
                "file_list": relevant_files,
                "description": f"Focused context for: {query}"
            }
        }

    def generate_bundle(self, entry_point: Optional[str] = None, query: Optional[str] = None, max_depth: int = 3, token_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate Bundle Mode prompt (structure + dependency slice from entry point).
        Supports "Focused Tree Mode" if query is provided instead of entry_point.
        Supports context optimization if token_limit is provided.

        Args:
            entry_point: Starting file path (e.g., "src/auth/AuthService.ts")
            query: Search query to find entry point (if entry_point is None)
            max_depth: Maximum dependency depth to traverse
            token_limit: Optional token budget for context optimization

        Returns:
            Dict with 'prompt' (str) and 'metadata' (dict)
        """
        # Resolve entry point from query if needed
        if not entry_point and query:
            search_dir = self.work_dir if self.work_dir else self.pack_path
            matches = hybrid_search(query, search_dir, max_items=1, registry=self.name_registry)
            if matches and matches[0].get('file_path'):
                entry_point = matches[0]['file_path']
            else:
                raise ValueError(f"Could not find any file matching query: {query}")

        if not entry_point:
            raise ValueError("Either entry_point or query must be provided for bundle mode")

        # Traverse dependency graph from entry point using helper
        bundled_files = get_recursive_dependencies(self.import_graph, entry_point)
        
        # Determine if we should include the full blueprint based on token budget
        # For tight budgets (< 32K), skip or minimize the blueprint to save tokens
        include_full_blueprint = True
        toon_structure = ""
        blueprint_tokens = 0
        
        if token_limit and token_limit < 32000:
            # For small context windows, skip the large blueprint
            include_full_blueprint = False
            blueprint_tokens = 0
        else:
            # Generate full blueprint for larger context windows
            toon_structure = generate_toon_blueprint(self.name_registry, self.import_graph)
            blueprint_tokens = estimate_tokens(toon_structure)
        
        # Calculate overhead tokens (blueprint + headers + formatting)
        # Base overhead for headers, code fences, file names, etc.
        base_overhead = 300 + (len(bundled_files) * 50)  # ~50 tokens per file for headers/fences
        estimated_overhead = blueprint_tokens + base_overhead

        # Apply context optimization if token_limit is provided
        pruning_report = None
        effective_token_limit = None
        if token_limit:
            # Subtract overhead from token limit to get budget for source files
            effective_token_limit = max(token_limit - estimated_overhead, 1000)  # Minimum 1000 tokens for source
            
            # Generate token_map on-the-fly if not available
            token_map_to_use = self.token_map
            if not token_map_to_use:
                token_map_to_use = {}
                for file_path in bundled_files:
                    source_code = self._read_source_file(file_path)
                    if source_code:
                        token_map_to_use[file_path] = estimate_tokens(source_code)
                    else:
                        token_map_to_use[file_path] = 0
            
            if token_map_to_use:
                optimized_files, pruning_report = optimize_context(
                    candidate_files=bundled_files,
                    import_graph=self.import_graph,
                    token_map=token_map_to_use,
                    max_tokens=effective_token_limit,
                    seed_files=[entry_point]
                )
                bundled_files = optimized_files

        # Assemble prompt
        prompt_parts = []
        prompt_parts.append(f"# CODEBASE BUNDLE: {entry_point}\n\n")
        if query:
            prompt_parts.append(f"Query: {query}\n\n")
        
        # Add pruning report at the top if optimization was applied
        if pruning_report:
            prompt_parts.append("## Context Optimization Report\n")
            prompt_parts.append(f"Target Token Limit: {token_limit:,}\n")
            if not include_full_blueprint:
                prompt_parts.append("Blueprint: Skipped (tight budget)\n")
            prompt_parts.append(f"Source Code Budget: {effective_token_limit:,}\n")
            prompt_parts.append(f"Source Code Used: {pruning_report.total_tokens_included:,}\n")
            prompt_parts.append(f"Files Included: {len(pruning_report.included_files)}\n")
            prompt_parts.append(f"Files Pruned: {len(pruning_report.pruned_files)}\n")
            if pruning_report.pruned_files:
                prompt_parts.append("\n**Pruned Files (not included due to budget):**\n")
                for pf in pruning_report.pruned_files[:10]:  # Show top 10
                    prompt_parts.append(f"- `{pf['file_path']}` ({pf['tokens']:,} tokens) - {pf['reason']}\n")
                if len(pruning_report.pruned_files) > 10:
                    prompt_parts.append(f"- ... and {len(pruning_report.pruned_files) - 10} more\n")
            prompt_parts.append("\n")
        
        # Only include blueprint if we have room for it
        if include_full_blueprint and toon_structure:
            prompt_parts.append("## Structural Blueprint\n")
            prompt_parts.append(toon_structure)
            prompt_parts.append("\n")

        # Add dependency tree visualization (compact)
        prompt_parts.append("## Dependency Tree\n")
        prompt_parts.append(f"Entry Point: {entry_point}\n")
        prompt_parts.append(f"Files: {len(bundled_files)}\n\n")
        for file_path in bundled_files:
            prompt_parts.append(f"- {file_path}\n")
        prompt_parts.append("\n")

        # Add source code for all bundled files
        prompt_parts.append(f"## Source Code ({len(bundled_files)} files)\n\n")

        for file_path in bundled_files:
            source_code = self._read_source_file(file_path)
            if source_code:
                lang = self._detect_language(file_path)
                prompt_parts.append(f"### File: {file_path}\n")
                prompt_parts.append(f"```{lang}\n")
                prompt_parts.append(source_code)
                prompt_parts.append("\n```\n\n")

        prompt = "".join(prompt_parts)

        # Build metadata
        metadata = {
            "mode": "bundle",
            "token_estimate": estimate_tokens(prompt),
            "entry_point": entry_point,
            "query": query,
            "files_included": len(bundled_files),
            "file_list": bundled_files,
            "max_depth": max_depth,
            "description": f"Complete dependency bundle for: {entry_point}"
        }
        
        # Add optimization info if applicable
        if pruning_report:
            metadata["optimization"] = {
                "optimization_applied": True,
                "token_limit": token_limit,
                "token_budget": token_limit,  # User's requested limit
                "effective_budget": effective_token_limit,  # After overhead subtraction
                "tokens_used": pruning_report.total_tokens_included + estimated_overhead,  # Total prompt tokens
                "source_tokens_used": pruning_report.total_tokens_included,
                "overhead_tokens": estimated_overhead,
                "files_pruned": len(pruning_report.pruned_files),
                "pruned_files": [pf["file_path"] for pf in pruning_report.pruned_files],
            }
        else:
            metadata["optimization"] = {
                "optimization_applied": False
            }
        
        return {
            "prompt": prompt,
            "metadata": metadata
        }

    def _read_source_file(self, file_path: str) -> Optional[str]:
        """
        Read source code from source_spans.json, spans.zip, or original repository.

        Args:
            file_path: Relative file path

        Returns:
            Source code string or None if not found
        """
        # Check if we have source files loaded from JSON or TOON
        if hasattr(self, 'source_files') and self.source_files:
            if file_path in self.source_files:
                # Handle both direct string (TOON) and dict (JSON) formats
                data = self.source_files[file_path]
                if isinstance(data, dict):
                    return data.get('full_source', '')
                return str(data)

        # Try to read from source_spans.json (flat file format after cleanup)
        if self.work_dir:
            spans_json_path = self.work_dir / "source_spans.json"
            if spans_json_path.exists():
                try:
                    with open(spans_json_path, 'r', encoding='utf-8') as f:
                        spans_data = json.load(f)
                    
                    # source_spans.json can be a list of dicts or a dict keyed by file path
                    if isinstance(spans_data, list):
                        for file_data in spans_data:
                            if file_data.get('file_path') == file_path:
                                return file_data.get('source_code', file_data.get('full_source', ''))
                    elif isinstance(spans_data, dict):
                        if file_path in spans_data:
                            data = spans_data[file_path]
                            if isinstance(data, dict):
                                return data.get('source_code', data.get('full_source', ''))
                            return str(data)
                except Exception as e:
                    print(f"Error reading source_spans.json: {e}", file=sys.stderr)

        # Try to read from spans.zip
        if self.work_dir:
            spans_zip_path = self.work_dir / "spans.zip"
            if spans_zip_path.exists():
                try:
                    with zipfile.ZipFile(spans_zip_path, 'r') as zf:
                        # Try reading file directly from zip (new format)
                        try:
                            with zf.open(file_path) as f:
                                return f.read().decode('utf-8', errors='replace')
                        except KeyError:
                            # Fallback to old format (source_spans.json inside zip)
                            try:
                                spans_json = zf.read('source_spans.json').decode('utf-8')
                                spans_data = json.loads(spans_json)
                                for file_data in spans_data:
                                    if file_data.get('file_path') == file_path:
                                        return file_data.get('source_code', '')
                            except KeyError:
                                pass
                except Exception:
                    pass

        # Fallback: try to read from repoBrief or other sources
        # If reading from extracted pack or temp_repo
        if self.work_dir:
            # Try to find the file in temp_repos if we are in a dev environment
            # This is a hack for local testing when pack doesn't have source
            project_root = self.work_dir.parent.parent # Assuming work_dir is temp/pack
            
            # Try to guess repo name from path or use self.repo_url
            repo_name = None
            if self.repo_url:
                repo_name = self.repo_url.split('/')[-1].replace('.git', '')
            
            # If we can't determine repo name, try to find any folder in temp_repos that contains this file
            temp_repos = project_root / "temp_repos"
            if temp_repos.exists():
                # If we have a repo name, check that specific folder
                if repo_name:
                    local_path = temp_repos / repo_name / file_path
                    if local_path.exists():
                        try:
                            with open(local_path, "r", encoding="utf-8", newline='') as f:
                                return f.read()
                        except Exception:
                            pass
                
                # Fallback: check all folders in temp_repos
                for repo_dir in temp_repos.iterdir():
                    if repo_dir.is_dir():
                        local_path = repo_dir / file_path
                        if local_path.exists():
                            try:
                                with open(local_path, "r", encoding="utf-8", newline='') as f:
                                    return f.read()
                            except Exception:
                                pass

        return f"// Source code for {file_path} not available in this pack mode\n"

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        lang_map = {
            '.js': 'javascript',
            '.jsx': 'jsx',
            '.ts': 'typescript',
            '.tsx': 'tsx',
            '.py': 'python',
            '.rs': 'rust',
            '.go': 'go',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.rb': 'ruby',
            '.php': 'php',
        }
        return lang_map.get(ext, 'text')

    def _traverse_dependencies(self, entry_point: str, max_depth: int) -> List[str]:
        """
        Traverse import graph from entry point to collect all dependencies.

        Args:
            entry_point: Starting file
            max_depth: Maximum recursion depth

        Returns:
            List of file paths (entry point + all dependencies)
        """
        visited = set()
        result = []

        def dfs(file_path: str, depth: int):
            if depth > max_depth or file_path in visited:
                return
            visited.add(file_path)
            result.append(file_path)

            # Get dependencies from import graph
            dependencies = self.import_graph.get(file_path, [])
            for dep in dependencies:
                dfs(dep, depth + 1)

        dfs(entry_point, 0)
        return result

    def _normalize_graph(self, graph: Dict[str, Any]) -> Dict[str, List[str]]:
        """Ensure graph is in adjacency list format."""
        if isinstance(graph, dict) and "edges" in graph:
            # Convert Node/Edge format to Adjacency List
            adj = {}
            for edge in graph["edges"]:
                source = edge["source"]
                target = edge["target"]
                if source not in adj: adj[source] = []
                adj[source].append(target)
            return adj
        return graph


# --- Helper: Suggest Entry Points ---
def suggest_entry_points(graph: dict) -> List[str]:
    # Handle Node/Edge format
    if "edges" in graph:
        all_files = set()
        imported = set()
        for edge in graph["edges"]:
            all_files.add(edge["source"])
            all_files.add(edge["target"])
            imported.add(edge["target"])
        roots = list(all_files - imported)
        return sorted(roots, key=lambda f: (0 if 'index' in f or 'main' in f else 1, f))

    # Handle Adjacency List format
    all_files = set(graph.keys())
    imported = set()
    for targets in graph.values():
        if isinstance(targets, list):
            for target in targets:
                imported.add(target)
        elif isinstance(targets, str):
             imported.add(targets)
             
    roots = list(all_files - imported)
    
    # Sort with preference for index/main/app
    return sorted(roots, key=lambda f: (0 if 'index' in f or 'main' in f else 1, f))


# --- Helper: Recursive Bundle ---
def get_recursive_dependencies(graph: dict, entry_file: str, visited=None) -> List[str]:
    if visited is None: visited = set()
    if entry_file in visited: return []
    visited.add(entry_file)
    
    results = [entry_file]
    
    targets = graph.get(entry_file, [])
    # Normalize targets to list
    if not isinstance(targets, list):
        targets = [targets] if targets else []
        
    for child in targets:
        results.extend(get_recursive_dependencies(graph, child, visited))
    return results


def generate_vibe_prompt(
    pack_path: str,
    mode: str,
    query: Optional[str] = None,
    entry_point: Optional[str] = None,
    max_files: int = 5,
    max_depth: int = 3,
    repo_url: Optional[str] = None,
    token_limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate a vibe coding prompt.

    Args:
        pack_path: Path to pack directory or ZIP file
        mode: One of "blueprint", "focus", or "bundle"
        query: Query string (required for "focus" mode)
        entry_point: Entry point file path (required for "bundle" mode)
        max_files: Max files to include in focus mode
        max_depth: Max dependency depth in bundle mode
        repo_url: Optional URL of the repository (for fallback source reading)
        token_limit: Optional token budget for context optimization (bundle mode only)

    Returns:
        Dict with 'prompt' and 'metadata'

    Raises:
        ValueError: If required parameters are missing
    """
    engine = PromptEngine(Path(pack_path), repo_url=repo_url)

    try:
        if mode == "blueprint":
            return engine.generate_blueprint()

        elif mode == "focus":
            if not query:
                raise ValueError("Query is required for focus mode")
            return engine.generate_focus(query, max_files)

        elif mode == "bundle":
            if not entry_point and not query:
                raise ValueError("Entry point or query is required for bundle mode")
            return engine.generate_bundle(
                entry_point=entry_point, 
                query=query, 
                max_depth=max_depth,
                token_limit=token_limit
            )

        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'blueprint', 'focus', or 'bundle'")

    finally:
        engine.cleanup()


if __name__ == "__main__":
    import sys

    # Test the prompt engine
    if len(sys.argv) < 3:
        print("Usage: python prompt_engine.py <pack_path> <mode> [query/entry_point]")
        print("Modes: blueprint, focus, bundle")
        sys.exit(1)

    pack_path = sys.argv[1]
    mode = sys.argv[2]
    query_or_entry = sys.argv[3] if len(sys.argv) > 3 else None

    result = generate_vibe_prompt(
        pack_path,
        mode,
        query=query_or_entry if mode == "focus" else None,
        entry_point=query_or_entry if mode == "bundle" else None
    )

    print("=== PROMPT ===")
    print(result["prompt"])
    print("\n=== METADATA ===")
    for key, value in result["metadata"].items():
        print(f"{key}: {value}")
