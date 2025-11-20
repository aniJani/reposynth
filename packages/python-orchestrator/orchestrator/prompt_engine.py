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
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .toon_formatter import generate_toon_blueprint, estimate_tokens, summarize_toon
from .retriever import retrieve_relevant_symbols


class PromptEngine:
    """
    Assembles LLM-optimized prompts from RepoSynth pack artifacts.
    """

    def __init__(self, pack_path: Path):
        """
        Initialize the prompt engine with a pack directory or ZIP file.

        Args:
            pack_path: Path to pack directory or ZIP file
        """
        self.pack_path = Path(pack_path)
        self.is_zip = pack_path.suffix == '.zip'
        self.temp_extract_dir = None

        # Load artifacts
        self._load_artifacts()

    def _load_artifacts(self):
        """Load JSON artifacts from pack directory or ZIP file."""
        if self.is_zip:
            self._extract_zip()
            work_dir = self.temp_extract_dir / "pack"
        else:
            work_dir = self.pack_path

        # Load name_registry.json
        registry_path = work_dir / "name_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r', encoding='utf-8') as f:
                self.name_registry = json.load(f)
        else:
            self.name_registry = {}

        # Load import_graph.json
        graph_path = work_dir / "import_graph.json"
        if graph_path.exists():
            with open(graph_path, 'r', encoding='utf-8') as f:
                self.import_graph = json.load(f)
        else:
            self.import_graph = {}

        # Load variable_registry.json (optional)
        var_registry_path = work_dir / "variable_registry.json"
        if var_registry_path.exists():
            with open(var_registry_path, 'r', encoding='utf-8') as f:
                self.variable_registry = json.load(f)
        else:
            self.variable_registry = {}

        # Store work directory for file reading
        self.work_dir = work_dir

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

        # Use retriever to find relevant symbols
        matches = retrieve_relevant_symbols(query, self.name_registry, max_items=max_files * 3)

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

    def generate_bundle(self, entry_point: str, max_depth: int = 3) -> Dict[str, Any]:
        """
        Generate Bundle Mode prompt (structure + dependency slice from entry point).

        Args:
            entry_point: Starting file path (e.g., "src/auth/AuthService.ts")
            max_depth: Maximum dependency depth to traverse

        Returns:
            Dict with 'prompt' (str) and 'metadata' (dict)
        """
        # Start with blueprint
        toon_structure = generate_toon_blueprint(self.name_registry, self.import_graph)

        # Traverse dependency graph from entry point
        bundled_files = self._traverse_dependencies(entry_point, max_depth)

        # Assemble prompt
        prompt_parts = []
        prompt_parts.append(f"# CODEBASE BUNDLE: {entry_point}\n\n")
        prompt_parts.append("## Structural Blueprint\n")
        prompt_parts.append(toon_structure)
        prompt_parts.append("\n")

        # Add dependency tree visualization
        prompt_parts.append("## Dependency Tree\n")
        prompt_parts.append(f"Entry Point: {entry_point}\n")
        prompt_parts.append(f"Dependencies: {len(bundled_files) - 1}\n\n")
        for i, file_path in enumerate(bundled_files):
            indent = "  " * (min(i, max_depth))
            prompt_parts.append(f"{indent}- {file_path}\n")
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

        return {
            "prompt": prompt,
            "metadata": {
                "mode": "bundle",
                "token_estimate": estimate_tokens(prompt),
                "entry_point": entry_point,
                "files_included": len(bundled_files),
                "file_list": bundled_files,
                "max_depth": max_depth,
                "description": f"Complete dependency bundle for: {entry_point}"
            }
        }

    def _read_source_file(self, file_path: str) -> Optional[str]:
        """
        Read source code from spans.zip or original repository.

        Args:
            file_path: Relative file path

        Returns:
            Source code string or None if not found
        """
        # Try to read from spans.zip first
        spans_zip_path = self.work_dir / "spans.zip"
        if spans_zip_path.exists():
            try:
                with zipfile.ZipFile(spans_zip_path, 'r') as zf:
                    # spans.zip contains JSON with source code
                    spans_json = zf.read('source_spans.json').decode('utf-8')
                    spans_data = json.loads(spans_json)

                    # Find the file in spans data
                    for file_data in spans_data:
                        if file_data.get('file_path') == file_path:
                            return file_data.get('source_code', '')
            except Exception:
                pass

        # Fallback: try to read from repoBrief or other sources
        # For now, return placeholder
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
        Traverse import graph to collect all dependencies.

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


def generate_vibe_prompt(
    pack_path: str,
    mode: str,
    query: Optional[str] = None,
    entry_point: Optional[str] = None,
    max_files: int = 5,
    max_depth: int = 3
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

    Returns:
        Dict with 'prompt' and 'metadata'

    Raises:
        ValueError: If required parameters are missing
    """
    engine = PromptEngine(Path(pack_path))

    try:
        if mode == "blueprint":
            return engine.generate_blueprint()

        elif mode == "focus":
            if not query:
                raise ValueError("Query is required for focus mode")
            return engine.generate_focus(query, max_files)

        elif mode == "bundle":
            if not entry_point:
                raise ValueError("Entry point is required for bundle mode")
            return engine.generate_bundle(entry_point, max_depth)

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
