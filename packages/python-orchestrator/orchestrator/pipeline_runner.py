# --- FILE: packages/python-orchestrator/orchestrator/pipeline_runner.py ---

import subprocess
import json
import uuid
import threading
import queue
from pathlib import Path
import sys
import sqlite3
import datetime
import hashlib
import shutil


# Third-party imports
import numpy as np
import faiss
from radon.complexity import cc_visit
from sentence_transformers import SentenceTransformer
from graphlib import TopologicalSorter
from collections import defaultdict

# Local imports
from .parser_client import ParserClient
from .language_adapter import get_adapter
import zipfile

def get_file_hash(file_path: Path) -> str:
    """Computes the SHA256 hash of a file's content."""
    if not file_path.exists(): return ""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

class Pipeline:
    def __init__(self, repo_path: str, output_path: str, daemon_path: str, config=None):
        self.repo_path = Path(repo_path).resolve()
        self.output_path = Path(output_path)
        self.daemon_path = daemon_path
        self.config = config or {}

        # Store cache in a persistent location (project root) instead of inside the repo
        # This prevents cache loss when temp repos are deleted
        project_root = Path(output_path).parent  # output_path is <root>/pack, so parent is root
        repo_name = self.repo_path.name
        self.cache_dir = project_root / ".reposynth_cache" / repo_name

        self.output_path.mkdir(exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.ast_dir = self.output_path / "ast_raw"
        self.ast_dir.mkdir(exist_ok=True)
        self.db_path = self.output_path / "analysis.sqlite"

        # These will be populated by the pipeline stages
        self.name_registry = {}
        self.import_graph = {}
        self.variable_registry = defaultdict(list)
        self.definitions_by_file = defaultdict(list)
        self.top_level_statements = defaultdict(list)  # NEW: Track executable statements

        # Current commit hash for caching
        self.commit_hash = None

    def _get_cache_key(self, stage_name: str, inputs: dict) -> str:
        """Generate a cache key for a pipeline stage based on its inputs."""
        cache_data = {
            "stage": stage_name,
            "commit": self.commit_hash,
            "inputs": inputs
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()

    def _init_db(self, db_conn):
        """Initializes the database schema on a given connection."""
        # CORRECT: Use the passed-in db_conn, not self.db_conn
        cursor = db_conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS complexity (
                id INTEGER PRIMARY KEY,
                file_path TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                complexity INTEGER NOT NULL,
                UNIQUE(file_path, symbol_name)
            )
        """
        )
        db_conn.commit()

    def run(self, config: dict):
        # Determine the commit hash for caching
        self.commit_hash = self._get_commit_hash()

        if config.get("run_parsing"):
            print("--- Running Stage 1: Parsing Repository ---")
            self.run_parsing(commit_hash=self.commit_hash)

        if config.get("build_graphs"):
            print("\n--- Running Stage 2: Building Graphs & Name Registry ---")
            self.build_graphs_and_registry(config=config)

        if config.get("build_variable_registry"):
            print("\n--- Running Stage 2.5 (Hybrid): Building Variable Registry ---")
            self.build_variable_registry(config=config)

        if config.get("run_analysis"):
            print("\n--- Running Stage 3: Static Analysis ---")
            self.run_static_analysis(config=config)

        if config.get("run_embeddings"):
            print("\n--- Running Stage 4: Generating Embeddings ---")
            self.generate_embeddings(config=config)

        if config.get("run_security_scans"):
            print("\n--- Running Stage 4.5: Security Scanning ---")
            self.run_security_scans(config=config)

        # Blast Radius Mode: Generate LLM-ready context instead of pack
        if config.get("pack_mode") == "blast-radius":
            print("\n--- Running Blast Radius Mode ---")

            # Ensure query exists
            query = config.get("query")
            if not query:
                raise ValueError("Blast Radius mode requires a --query parameter")

            # Import and run blast radius calculation
            from .blast_radius import calculate_blast_radius

            max_shockwave = config.get("max_shockwave_files", 50)
            context_md = calculate_blast_radius(
                self.repo_path,
                self.output_path,
                query,
                max_shockwave_files=max_shockwave
            )

            # Save the output
            output_file = self.output_path / "vibecode_prompt.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(context_md)

            print(f"\n✅ VIBECODE READY: {output_file}")
            print(f"📋 Copy this file's content and paste into your LLM chat")
            print("\n--- Pipeline Finished Successfully! ---")
            return  # Skip normal pack assembly

        print("\n--- Running Stage 5: Assembling Final Pack ---")
        self.assemble_pack(config=config)

        print("\n--- Pipeline Finished Successfully! ---")
        print(f"Artifacts saved in: {self.output_path}")
    
    def _get_commit_hash(self) -> str:
        """Gets the current git commit hash of the repository."""
        # Check if it's a git repository first
        git_dir = self.repo_path / ".git"
        if not git_dir.exists():
            print("Warning: Not a git repository. Caching will be less reliable.", file=sys.stderr)
            # Fallback for non-git directories: hash the file structure
            # This is not perfect but better than nothing for local folders.
            file_list_str = "".join(sorted([str(p) for p in self.repo_path.glob("**/*") if p.is_file()]))
            return hashlib.sha256(file_list_str.encode()).hexdigest()

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception as e:
            print(f"Warning: Could not get git commit hash: {e}", file=sys.stderr)
            return "no-git-commit"

    def run_parsing(self, commit_hash: str):
        file_hashes_path = self.cache_dir / f"{commit_hash}_file_hashes.json"
        
        last_hashes = {}
        if file_hashes_path.exists():
            with open(file_hashes_path, 'r') as f:
                last_hashes = json.load(f)
        
        client = ParserClient(daemon_path=self.daemon_path)
        try:
            # Discover all files first
            all_files = client._discover_files(self.repo_path)
            
            files_to_parse = []
            current_hashes = {}
            for file_path in all_files:
                file_hash = get_file_hash(file_path)
                current_hashes[str(file_path)] = file_hash
                # Only parse if file is new or has changed
                if last_hashes.get(str(file_path)) != file_hash:
                    files_to_parse.append(file_path)
            
            print(f"Found {len(all_files)} total files. {len(files_to_parse)} files are new or modified.")
            
            if files_to_parse:
                # This method now needs to be modified to only parse a subset of files
                # And to not create the manifest itself.
                client.parse_files(files_to_parse, output_dir=str(self.ast_dir), repo_path=self.repo_path)

            # Create the manifest based on ALL files, not just parsed ones
            ast_manifest = {}
            for file_path in all_files:
                # Create a stable unique name based on the file path
                relative_path = file_path.relative_to(self.repo_path)
                safe_name = str(relative_path).replace('/', '_').replace('\\', '_')
                unique_ast_filename = f"{safe_name}.jsonl"
                ast_manifest[unique_ast_filename] = str(file_path.resolve())
            
            with open(self.ast_dir / "ast_manifest.json", 'w') as f:
                json.dump(ast_manifest, f, indent=2)

            # Save the new hashes for the next run
            with open(file_hashes_path, 'w') as f:
                json.dump(current_hashes, f, indent=2)

        finally:
            client.shutdown()

    def build_graphs_and_registry(self, config: dict = None):
        config = config or {}

        # Check cache first
        manifest_path = self.ast_dir / "ast_manifest.json"
        if not manifest_path.exists():
            print(
                "Warning: ast_manifest.json not found. No files were parsed. Skipping graph building.",
                file=sys.stderr,
            )
            # Create empty outputs so downstream stages don't crash
            with open(self.output_path / "name_registry.json", "w") as f:
                json.dump({}, f, indent=2)
            with open(self.output_path / "import_graph.json", "w") as f:
                json.dump({}, f, indent=2)
            self.name_registry = {}
            self.import_graph = {}
            self.definitions_by_file = defaultdict(list)
            return

        with open(manifest_path, "r", encoding="utf-8") as f:
            ast_manifest = json.load(f)

        # Create cache key based on AST files
        manifest_hash = hashlib.sha256(json.dumps(ast_manifest, sort_keys=True).encode()).hexdigest()
        cache_key = self._get_cache_key("build_graphs", {"manifest_hash": manifest_hash})

        cache_name_registry = self.cache_dir / f"name_registry_{cache_key}.json"
        cache_import_graph = self.cache_dir / f"import_graph_{cache_key}.json"
        cache_definitions = self.cache_dir / f"definitions_{cache_key}.json"
        cache_top_level_statements = self.cache_dir / f"top_level_statements_{cache_key}.json"

        # Try to load from cache
        if not config.get("no_cache") and cache_name_registry.exists() and cache_import_graph.exists():
            print("⚡ CACHE HIT: Loading graphs from cache...")
            with open(cache_name_registry, "r") as f:
                self.name_registry = json.load(f)
            with open(cache_import_graph, "r") as f:
                self.import_graph = json.load(f)
            if cache_definitions.exists():
                with open(cache_definitions, "r") as f:
                    defs_dict = json.load(f)
                    self.definitions_by_file = defaultdict(list, defs_dict)
            else:
                self.definitions_by_file = defaultdict(list)

            if cache_top_level_statements.exists():
                with open(cache_top_level_statements, "r") as f:
                    stmts_dict = json.load(f)
                    self.top_level_statements = defaultdict(list, stmts_dict)
            else:
                self.top_level_statements = defaultdict(list)

            # Also copy to output directory
            shutil.copy(cache_name_registry, self.output_path / "name_registry.json")
            shutil.copy(cache_import_graph, self.output_path / "import_graph.json")
            if cache_top_level_statements.exists():
                shutil.copy(cache_top_level_statements, self.output_path / "top_level_statements.json")
            print("✓ Graphs loaded from cache (instant)")
            return

        # Cache miss - build from scratch
        self.definitions_by_file = defaultdict(list)
        self.top_level_statements = defaultdict(list)

        for ast_filename, original_filepath_str in ast_manifest.items():
            ast_file = self.ast_dir / ast_filename
            original_file = Path(original_filepath_str)

            if not ast_file.exists():
                continue

            try:
                with open(ast_file, "r", encoding="utf-8") as f:
                    ast_nodes = [json.loads(line) for line in f]

                relative_path = str(original_file.relative_to(self.repo_path))

                # Read with newline='' to preserve original line endings (CRLF on Windows)
                # This ensures byte positions from AST match the source text
                with open(original_file, "r", encoding="utf-8", newline='') as f:
                    source_code = f.read()

                adapter = get_adapter(original_file)
                if not adapter:
                    continue

                definitions = adapter.get_definitions(ast_nodes, source_code)
                self.definitions_by_file[relative_path] = definitions
                for a_def in definitions:
                    fqn = f"{relative_path}:{a_def['name']}"
                    self.name_registry[fqn] = {
                        "kind": a_def["kind"],
                        "file_path": relative_path,
                        "start_byte": a_def["start_byte"],
                        "end_byte": a_def["end_byte"],
                        "is_public": a_def.get("is_public", False),
                    }

                raw_imports = adapter.get_imports(ast_nodes, source_code)
                resolved_imports = []
                current_dir = original_file.parent

                for imp in raw_imports:
                    try:
                        # Handle relative imports (e.g., ".schemas", "..utils")
                        if imp.startswith("."):
                            # Count leading dots and strip them
                            num_dots = len(imp) - len(imp.lstrip('.'))
                            module_name = imp.lstrip('.')

                            # Determine the base directory
                            # . = current directory, .. = parent, ... = grandparent, etc.
                            base_dir = current_dir
                            for _ in range(num_dots - 1):
                                base_dir = base_dir.parent

                            # Resolve to the target file/directory
                            if module_name:
                                target = base_dir / module_name
                            else:
                                # Just dots like "." or ".." - refers to __init__.py
                                target = base_dir

                            # Check if the target is within our repo
                            if self.repo_path in target.parents or target == self.repo_path:
                                # Try to find a file with .ts, .js, .py extensions
                                found = False
                                for ext in [".py", ".ts", ".js", ".tsx", ".jsx"]:
                                    file_path = target.with_suffix(ext)
                                    if file_path.exists():
                                        try:
                                            resolved_imports.append(str(file_path.relative_to(self.repo_path)))
                                            found = True
                                            break
                                        except ValueError:
                                            continue

                                # Try as a package (with __init__.py)
                                if not found:
                                    init_path = target / "__init__.py"
                                    if init_path.exists():
                                        try:
                                            resolved_imports.append(str(init_path.relative_to(self.repo_path)))
                                        except ValueError:
                                            continue
                        else:
                            # Handle absolute imports (e.g., "orchestrator.parser_client")
                            # We need to search for the module in the repository
                            module_parts = imp.split(".")
                            module_filename = module_parts[-1]  # Last part is the module name

                            # Directories to exclude from import resolution
                            exclude_dirs = {".venv", "venv", "node_modules", ".git", "__pycache__", "build", "dist", ".tox", ".pytest_cache"}

                            # Search for files matching the module name
                            found = False
                            for ext in [".py", ".ts", ".js", ".tsx", ".jsx"]:
                                # Use glob to find all files with this name
                                pattern = f"**/{module_filename}{ext}"
                                matching_files = list(self.repo_path.glob(pattern))

                                # Filter out excluded directories
                                matching_files = [
                                    f for f in matching_files
                                    if not any(excluded in f.parts for excluded in exclude_dirs)
                                ]

                                # Try to find the most likely match based on the full module path
                                for match in matching_files:
                                    try:
                                        rel_path = match.relative_to(self.repo_path)
                                        # Check if the module path matches the file path structure
                                        # For "orchestrator.parser_client", look for files ending with "orchestrator/parser_client.py"
                                        expected_suffix = "/".join(module_parts) + ext
                                        if str(rel_path).replace("\\", "/").endswith(expected_suffix):
                                            resolved_imports.append(str(rel_path))
                                            found = True
                                            break
                                    except ValueError:
                                        continue

                                if found:
                                    break

                            # Try as a package (with __init__.py)
                            if not found:
                                pattern = f"**/{module_filename}/__init__.py"
                                matching_files = list(self.repo_path.glob(pattern))

                                # Filter out excluded directories
                                matching_files = [
                                    f for f in matching_files
                                    if not any(excluded in f.parts for excluded in exclude_dirs)
                                ]

                                for match in matching_files:
                                    try:
                                        rel_path = match.relative_to(self.repo_path)
                                        # Verify it matches the module structure
                                        path_parts = str(rel_path).replace("\\", "/").split("/")
                                        # Remove __init__.py from the end
                                        path_parts = path_parts[:-1]
                                        # Check if the module parts appear in order at the end of the path
                                        if len(path_parts) >= len(module_parts):
                                            if path_parts[-len(module_parts):] == module_parts:
                                                resolved_imports.append(str(rel_path))
                                                break
                                    except ValueError:
                                        continue

                    except Exception:
                        continue  # Ignore resolution errors

                self.import_graph[relative_path] = resolved_imports

                # NEW: Capture top-level executable statements
                try:
                    top_level_stmts = adapter.get_top_level_statements(ast_nodes, source_code)
                    if top_level_stmts:
                        self.top_level_statements[relative_path] = top_level_stmts
                except Exception as stmt_error:
                    # Don't fail the whole pipeline if statement extraction fails
                    print(f"Warning: Failed to extract top-level statements for {relative_path}: {stmt_error}", file=sys.stderr)

            except Exception as e:
                print(
                    f"Failed to process graph for {ast_filename} (original: {relative_path}): {e}",
                    file=sys.stderr,
                )

        with open(self.output_path / "name_registry.json", "w") as f:
            json.dump(self.name_registry, f, indent=2)
        with open(self.output_path / "import_graph.json", "w") as f:
            json.dump(self.import_graph, f, indent=2)
        with open(self.output_path / "top_level_statements.json", "w") as f:
            json.dump(dict(self.top_level_statements), f, indent=2)

        # Save to cache for next run
        if not config.get("no_cache"):
            with open(cache_name_registry, "w") as f:
                json.dump(self.name_registry, f, indent=2)
            with open(cache_import_graph, "w") as f:
                json.dump(self.import_graph, f, indent=2)
            with open(cache_definitions, "w") as f:
                json.dump(dict(self.definitions_by_file), f, indent=2)
            with open(cache_top_level_statements, "w") as f:
                json.dump(dict(self.top_level_statements), f, indent=2)

        print("Finished building graphs and name registry.")
        if self.top_level_statements:
            print(f"Captured {sum(len(stmts) for stmts in self.top_level_statements.values())} top-level statements across {len(self.top_level_statements)} files.")

    def build_variable_registry(self, config: dict = None):
        config = config or {}

        manifest_path = self.ast_dir / "ast_manifest.json"
        if not manifest_path.exists():
            print(
                "Warning: ast_manifest.json not found. Skipping variable registry.",
                file=sys.stderr,
            )
            self.variable_registry = defaultdict(list)
            return

        with open(manifest_path, "r", encoding="utf-8") as f:
            ast_manifest = json.load(f)

        # Create cache key based on AST manifest + definitions
        manifest_hash = hashlib.sha256(json.dumps(ast_manifest, sort_keys=True).encode()).hexdigest()
        definitions_hash = hashlib.sha256(json.dumps(dict(self.definitions_by_file), sort_keys=True).encode()).hexdigest()
        cache_key = self._get_cache_key("variable_registry", {
            "manifest_hash": manifest_hash,
            "definitions_hash": definitions_hash
        })

        cache_variable_registry = self.cache_dir / f"variable_registry_{cache_key}.json"

        # Try to load from cache
        if not config.get("no_cache") and cache_variable_registry.exists():
            print("⚡ CACHE HIT: Loading variable registry from cache...")
            with open(cache_variable_registry, "r") as f:
                var_dict = json.load(f)
                self.variable_registry = defaultdict(list, var_dict)
            # Also copy to output directory
            shutil.copy(cache_variable_registry, self.output_path / "variable_registry.json")
            print("✓ Variable registry loaded from cache (instant)")
            return

        # Cache miss - build from scratch
        self.variable_registry = defaultdict(list)

        for ast_filename, original_filepath_str in ast_manifest.items():
            ast_file = self.ast_dir / ast_filename
            original_file = Path(original_filepath_str)

            if not ast_file.exists():
                continue

            relative_path = str(original_file.relative_to(self.repo_path))
            adapter = get_adapter(original_file)
            if not adapter:
                continue

            try:
                with open(ast_file, "r", encoding="utf-8") as f:
                    ast_nodes = [json.loads(line) for line in f]
                with open(original_file, "r", encoding="utf-8", newline='') as f:
                    source_code = f.read()

                # Pass definitions for context
                definitions = self.definitions_by_file.get(relative_path, [])
                variables = adapter.get_variables(ast_nodes, source_code, definitions)
                self.variable_registry[relative_path].extend(variables)

            except Exception as e:
                print(
                    f"Failed to build variable registry for {relative_path}: {e}",
                    file=sys.stderr,
                )

        with open(self.output_path / "variable_registry.json", "w") as f:
            json.dump(self.variable_registry, f, indent=2)

        # Save to cache for next run
        if not config.get("no_cache"):
            with open(cache_variable_registry, "w") as f:
                json.dump(dict(self.variable_registry), f, indent=2)

        print("Variable registry saved.")

    def store_spans(self):
        """
        Creates a comprehensive JSON file containing source code for all public APIs.
        The JSON includes the full source of each file plus extracted spans for each public API.
        NOW INCLUDES: Exported types, interfaces, and enums from variable_registry!
        """
        json_path = self.output_path / "source_spans.json"
        source_spans = {}

        # Group public APIs by file
        files_with_apis = defaultdict(list)

        # Add items from name_registry (functions, classes, constants)
        for fqn, data in self.name_registry.items():
            if data.get("is_public", False):
                files_with_apis[data["file_path"]].append({
                    "fqn": fqn,
                    "name": fqn.split(":")[-1],
                    "kind": data["kind"],
                    "start_byte": data["start_byte"],
                    "end_byte": data["end_byte"]
                })

        # FIXED: Also add exported types/interfaces/enums from variable_registry
        for file_path, variables in self.variable_registry.items():
            for var in variables:
                # Include variables with scope "export" and that have a "kind" field
                if var.get("scope") == "export" and "kind" in var:
                    # Create a FQN for this exported type
                    fqn = f"{file_path}:{var['name']}"
                    files_with_apis[file_path].append({
                        "fqn": fqn,
                        "name": var["name"],
                        "kind": var.get("kind", "exported_variable"),
                        "start_byte": var["start_byte"],
                        "end_byte": var["end_byte"]
                    })

        # Build the complete source spans structure
        for file_path_str, apis in files_with_apis.items():
            try:
                full_path = self.repo_path / file_path_str
                with open(full_path, "r", encoding="utf-8", newline='') as f:
                    source_code = f.read()

                # Extract the actual code span for each API
                api_details = []
                for api in apis:
                    span_code = source_code[api["start_byte"]:api["end_byte"]]
                    api_details.append({
                        "name": api["name"],
                        "fqn": api["fqn"],
                        "kind": api["kind"],
                        "start_byte": api["start_byte"],
                        "end_byte": api["end_byte"],
                        "span": span_code
                    })

                source_spans[file_path_str] = {
                    "file_path": file_path_str,
                    "full_source": source_code,
                    "public_apis": api_details
                }

            except FileNotFoundError:
                print(f"Warning: Could not read file {file_path_str}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"Warning: Error processing {file_path_str}: {e}", file=sys.stderr)
                continue

        # Save the JSON file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(source_spans, f, indent=2)

        print(f"Source spans saved to {json_path}")
        print(f"  - {len(source_spans)} files with public APIs")
        print(f"  - {sum(len(v['public_apis']) for v in source_spans.values())} total public symbols")

        return source_spans

    def run_static_analysis(self, config: dict = None):
        config = config or {}

        # Determine files to analyze based on name_registry
        files_to_analyze = {}
        for fqn, data in self.name_registry.items():
            if "function" in data["kind"] or "method" in data["kind"]:
                if data["file_path"].endswith(".py"):
                    file_path = data["file_path"]
                    if file_path not in files_to_analyze:
                        files_to_analyze[file_path] = []

                    symbol_name = fqn.split(":")[-1]
                    files_to_analyze[file_path].append(symbol_name)

        # Create cache key based on files to analyze
        files_hash = hashlib.sha256(json.dumps(files_to_analyze, sort_keys=True).encode()).hexdigest()
        cache_key = self._get_cache_key("static_analysis", {"files_hash": files_hash})

        cache_db_path = self.cache_dir / f"analysis_{cache_key}.sqlite"

        # Try to load from cache
        if not config.get("no_cache") and cache_db_path.exists():
            print("⚡ CACHE HIT: Loading static analysis from cache...")
            shutil.copy(cache_db_path, self.db_path)
            print("✓ Static analysis loaded from cache (instant)")
            return

        # Cache miss - run analysis from scratch
        if self.db_path.exists():
            try:
                self.db_path.unlink()
            except OSError as e:
                print(f"Warning: Could not delete old database file: {e}", file=sys.stderr)
                return # Exit the stage if we can't clean up

        # 2. Connect to the new, empty database and initialize schema.
        db_conn = sqlite3.connect(self.db_path)
        try:
            self._init_db(db_conn)
            cursor = db_conn.cursor()

        # Use Ruff via subprocess to check McCabe complexity
        # We select only the "mccabe" complexity checker (C901)
        # and set its max_complexity to 1 to report on every function.

            print("Running static analysis with Ruff...")
            for file_path_str, symbol_names in files_to_analyze.items():
                file_path = self.repo_path / file_path_str

                try:
                    # Run Ruff with C901 (McCabe complexity) enabled and max complexity of 1
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "ruff",
                            "check",
                            str(file_path),
                            "--select",
                            "C901",
                            "--config",
                            f"mccabe.max-complexity=1",
                            "--output-format",
                            "json",
                        ],
                        capture_output=True,
                        text=True,
                    )

                    # Parse JSON output from Ruff
                    if result.stdout:
                        diagnostics = json.loads(result.stdout)

                        for diag in diagnostics:
                            # Extract function name and complexity from the message
                            message = diag.get("message", "")
                            try:
                                # Message format: "`function_name` is too complex (10)"
                                func_name = message.split("`")[1]
                                complexity = int(message.split("(")[-1].replace(")", ""))

                                if func_name in symbol_names:
                                    cursor.execute(
                                        "INSERT OR IGNORE INTO complexity (file_path, symbol_name, complexity) VALUES (?, ?, ?)",
                                        (file_path_str, func_name, complexity),
                                    )
                            except (IndexError, ValueError):
                                continue

                except Exception as e:
                    print(f"Ruff analysis failed for {file_path_str}: {e}", file=sys.stderr)
        finally:
            db_conn.commit()
            db_conn.close()

        # Save to cache for next run
        if not config.get("no_cache"):
            shutil.copy(self.db_path, cache_db_path)

        print("Static analysis metrics saved to analysis.sqlite")

    def generate_embeddings(self, config: dict = None):
        config = config or {}

        # 1. Collect public APIs to embed
        public_apis = {}
        for fqn, data in self.name_registry.items():
            if data.get("is_public", False):
                file_path = self.repo_path / data["file_path"]
                try:
                    with open(file_path, "r", encoding="utf-8", newline='') as f:
                        source_code = f.read()

                    # Create a snippet for embedding (e.g., function signature)
                    snippet = source_code[data["start_byte"] : data["end_byte"]].split(
                        "\n"
                    )[0]
                    public_apis[fqn] = snippet
                except Exception:
                    continue  # Skip if file can't be read

        if not public_apis:
            print("No public APIs found to embed. Skipping.")
            return

        # Create cache key based on public APIs
        apis_hash = hashlib.sha256(json.dumps(public_apis, sort_keys=True).encode()).hexdigest()
        cache_key = self._get_cache_key("embeddings", {"apis_hash": apis_hash})

        cache_faiss_path = self.cache_dir / f"vectors_{cache_key}.faiss"
        cache_ids_path = self.cache_dir / f"vector_ids_{cache_key}.json"

        # Try to load from cache
        if not config.get("no_cache") and cache_faiss_path.exists() and cache_ids_path.exists():
            print("⚡ CACHE HIT: Loading embeddings from cache (skipping expensive ML model)...")
            shutil.copy(cache_faiss_path, self.output_path / "vectors.faiss")
            shutil.copy(cache_ids_path, self.output_path / "vector_ids.json")
            print("✓ Embeddings loaded from cache (instant - saved ~10 seconds)")
            return

        # 2. Load model and generate embeddings
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        fqns = list(public_apis.keys())
        snippets = list(public_apis.values())

        print(f"Generating embeddings for {len(fqns)} public symbols...")
        embeddings = model.encode(snippets, show_progress_bar=True)

        # 3. Create and save FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIDMap(index)  # Allows mapping to our own IDs

        ids = np.array(range(len(fqns)))
        index.add_with_ids(embeddings, ids)

        faiss.write_index(index, str(self.output_path / "vectors.faiss"))

        # 4. Save the ID -> FQN mapping
        id_map = {i: fqn for i, fqn in enumerate(fqns)}
        with open(self.output_path / "vector_ids.json", "w") as f:
            json.dump(id_map, f, indent=2)

        # Save to cache for next run
        if not config.get("no_cache"):
            shutil.copy(self.output_path / "vectors.faiss", cache_faiss_path)
            shutil.copy(self.output_path / "vector_ids.json", cache_ids_path)

        print("Embeddings and FAISS index saved successfully.")

    def run_security_scans(self, config: dict = None):
        """
        Stage 4.5: Run security scans to detect potential vulnerabilities.
        Uses detect-secrets to scan for hardcoded secrets, API keys, tokens, etc.
        """
        config = config or {}

        # Create cache key based on commit hash
        cache_key = self._get_cache_key("security_scans", {"commit": self.commit_hash})
        cache_security_report = self.cache_dir / f"security_report_{cache_key}.json"

        # Try to load from cache
        if not config.get("no_cache") and cache_security_report.exists():
            print("⚡ CACHE HIT: Loading security scan results from cache...")
            shutil.copy(cache_security_report, self.output_path / "security_report.json")
            print("✓ Security scan results loaded from cache (instant)")
            return

        # Cache miss - run security scans
        security_report = {
            "scan_timestamp": datetime.datetime.utcnow().isoformat(),
            "repository": str(self.repo_path),
            "commit_hash": self.commit_hash,
            "findings": []
        }

        print("Running security scans with detect-secrets...")

        try:
            # Run detect-secrets scan
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "detect_secrets",
                    "scan",
                    str(self.repo_path),
                    "--baseline",
                    "/dev/null",  # Don't use baseline file
                ],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.stdout:
                try:
                    secrets_data = json.loads(result.stdout)

                    # Parse detect-secrets output
                    if "results" in secrets_data:
                        for file_path, findings in secrets_data["results"].items():
                            for finding in findings:
                                # Make file path relative to repo
                                try:
                                    rel_path = Path(file_path).relative_to(self.repo_path)
                                except (ValueError, Exception):
                                    rel_path = file_path

                                security_report["findings"].append({
                                    "file_path": str(rel_path),
                                    "line_number": finding.get("line_number", 0),
                                    "type": finding.get("type", "unknown"),
                                    "severity": "high" if "key" in finding.get("type", "").lower() else "medium",
                                    "hashed_secret": finding.get("hashed_secret", ""),
                                    "is_verified": finding.get("is_verified", False)
                                })

                    print(f"✓ Security scan completed. Found {len(security_report['findings'])} potential issues.")

                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse detect-secrets output: {e}", file=sys.stderr)
                    security_report["scan_error"] = "Failed to parse scanner output"
            else:
                print("✓ Security scan completed. No issues found.")

        except subprocess.TimeoutExpired:
            print("Warning: Security scan timed out after 5 minutes.", file=sys.stderr)
            security_report["scan_error"] = "Scan timed out"
        except FileNotFoundError:
            print("Warning: detect-secrets not found. Skipping security scan.", file=sys.stderr)
            print("Install with: pip install detect-secrets", file=sys.stderr)
            security_report["scan_error"] = "detect-secrets not installed"
        except Exception as e:
            print(f"Warning: Security scan failed: {e}", file=sys.stderr)
            security_report["scan_error"] = str(e)

        # Add scan summary
        security_report["summary"] = {
            "total_findings": len(security_report["findings"]),
            "high_severity": sum(1 for f in security_report["findings"] if f.get("severity") == "high"),
            "medium_severity": sum(1 for f in security_report["findings"] if f.get("severity") == "medium"),
            "files_scanned": len(set(f["file_path"] for f in security_report["findings"]))
        }

        # Save the report
        report_path = self.output_path / "security_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(security_report, f, indent=2)

        # Save to cache
        if not config.get("no_cache"):
            shutil.copy(report_path, cache_security_report)

        print(f"Security report saved to: {report_path}")
        if security_report["findings"]:
            print(f"⚠️  Warning: Found {len(security_report['findings'])} potential security issues!")
            print("   Review the security_report.json file for details.")

    def generate_deterministic_brief(self, top_n_modules=10, complexity_threshold=10):
        hotspots = []
        if self.db_path.exists():
            db_conn = sqlite3.connect(self.db_path)
            try:
                cursor = db_conn.cursor()
                cursor.execute("SELECT file_path, symbol_name, complexity FROM complexity WHERE complexity >= ? ORDER BY complexity DESC LIMIT 5", (complexity_threshold,))
                hotspots = cursor.fetchall()
            finally:
                db_conn.close()
        else:
            print("Warning: analysis.sqlite not found. Skipping complexity hotspots.", file=sys.stderr)

        brief = []


        repo_name = self.repo_path.name
        brief.append(f"# Architectural Briefing: {repo_name}\n")

        # --- FIX 2: Proper Graph Centrality Calculation ---
        # Filter out test files from the graph for a cleaner summary
        filtered_graph = {
            node: deps
            for node, deps in self.import_graph.items()
            if "test" not in node.lower() and "spec" not in node.lower()
        }

        # Build a reverse graph to easily calculate in-degrees (who imports me?)
        reverse_graph = defaultdict(list)
        for node, deps in filtered_graph.items():
            for dep in deps:
                reverse_graph[dep].append(node)

        # Calculate in-degrees for all nodes in our filtered graph
        inbound_counts = {
            node: len(reverse_graph.get(node, [])) for node in filtered_graph
        }

        sorted_modules = sorted(
            inbound_counts.items(), key=lambda item: item[1], reverse=True
        )
        key_modules = sorted_modules[:top_n_modules]
        # --- END FIX 2 ---

        brief.append("## Key Architectural Modules\n")
        if key_modules:
            brief.append(
                "Based on import centrality, the most important modules are:\n"
            )
            for path, count in key_modules:
                brief.append(f"- `{path}` (Imported by {count} other modules)")
        else:
            brief.append("Could not determine key modules based on internal imports.")
        brief.append("\n---\n")

        if hotspots:
            brief.append("## Complexity Hotspots\n")
            brief.append(
                f"Functions with a cyclomatic complexity of {complexity_threshold} or higher:\n"
            )
            for file_path, symbol_name, complexity in hotspots:
                brief.append(
                    f"- `{file_path}:{symbol_name}` (Complexity: {complexity})"
                )
            brief.append("\n---\n")

        brief.append("## Key Module Details\n")
        for path, _ in key_modules:
            brief.append(f"### Module: `{path}`\n")

            public_apis = []
            for fqn, data in self.name_registry.items():
                if data["file_path"] == path and data.get("is_public", False):
                    try:
                        with open(self.repo_path / path, "r", encoding="utf-8", newline='') as f:
                            source_code = f.read()

                        # Extract the signature properly
                        snippet = source_code[data["start_byte"] : data["end_byte"]]
                        lines = snippet.split("\n")

                        # Find the definition line (skip decorators)
                        signature_lines = []
                        in_signature = False
                        for i, line in enumerate(lines[:20]):  # Check first 20 lines max
                            stripped = line.strip()

                            # Skip empty lines and comments
                            if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                                continue

                            # Skip decorators
                            if stripped.startswith("@"):
                                continue

                            # Check if this is a definition line
                            if not in_signature:
                                if any(keyword in stripped for keyword in ["def ", "class ", "function ", "const ", "let ", "var ", "export function", "export class", "export const"]):
                                    in_signature = True
                                    signature_lines.append(stripped)
                                    # Check if signature ends on this line
                                    if stripped.endswith(":") or stripped.endswith("{") or stripped.endswith(";"):
                                        break
                                    continue
                            else:
                                # We're collecting a multi-line signature
                                signature_lines.append(stripped)
                                if stripped.endswith(":") or stripped.endswith("{") or stripped.endswith(";"):
                                    break

                        if signature_lines:
                            signature = " ".join(signature_lines)
                            # Remove trailing : or { for cleaner display
                            if signature.endswith(":") or signature.endswith("{") or signature.endswith(";"):
                                signature = signature[:-1].strip()
                            # Truncate very long signatures
                            if len(signature) > 120:
                                signature = signature[:117] + "..."
                            public_apis.append(f"`{signature}`")
                    except Exception:
                        continue

            if public_apis:
                brief.append("**Public API:**\n")
                for api in public_apis:
                    brief.append(f"- {api}")
                brief.append("")

            brief.append(
                f"**Dependencies:** Imports `{len(self.import_graph.get(path, []))}` other local modules.\n"
            )

        return "\n".join(brief)

    # --- NEW METHOD ---
    def assemble_pack(self, config: dict):
        """
        Generates the final pack in the appropriate format based on configuration.
        - Semantic mode: Creates repoBrief.md and manifest.json (lightweight)
        - Hybrid mode: Creates a complete .zip archive with all artifacts
        - Full mode: Creates comprehensive .zip with all possible artifacts
        """

        # 1. Generate the brief (always needed)
        brief_content = self.generate_deterministic_brief()

        # 2. Store spans (now for ALL modes - creates source_spans.json)
        source_spans = {}
        if config.get("store_spans"):
            source_spans = self.store_spans()

        # 3. Append source code section to the brief if we have source spans
        if source_spans:
            brief_content += "\n\n---\n\n"
            brief_content += "## Source Code for Public APIs\n\n"
            brief_content += f"This section contains the full source code for all {sum(len(v['public_apis']) for v in source_spans.values())} public APIs across {len(source_spans)} files.\n\n"

            for file_path, file_data in sorted(source_spans.items()):
                brief_content += f"### File: `{file_path}`\n\n"

                for api in file_data["public_apis"]:
                    brief_content += f"#### {api['kind']}: `{api['name']}`\n\n"
                    brief_content += f"**Fully Qualified Name:** `{api['fqn']}`\n\n"
                    brief_content += f"**Location:** bytes {api['start_byte']}-{api['end_byte']}\n\n"
                    brief_content += "```python\n" if file_path.endswith('.py') else "```typescript\n" if file_path.endswith(('.ts', '.tsx')) else "```javascript\n" if file_path.endswith(('.js', '.jsx')) else "```\n"
                    brief_content += api['span']
                    brief_content += "\n```\n\n"

        # Write the complete brief with source code
        with open(self.output_path / "repoBrief.md", "w", encoding="utf-8") as f:
            f.write(brief_content)

        # 4. Generate the manifest with checksums
        manifest = {
            "createdAt": datetime.datetime.utcnow().isoformat(),
            "repoPath": str(self.repo_path),
            "packVersion": "1.0",
            "mode": config.get("pack_mode", "semantic"),
            "commit_hash": self.commit_hash,
            "artifacts": {},
        }

        # Calculate checksums for all generated artifacts
        for artifact_path in self.output_path.glob("*"):
            if artifact_path.is_file() and artifact_path.name != "manifest.json":
                with open(artifact_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                manifest["artifacts"][artifact_path.name] = {
                    "size_bytes": artifact_path.stat().st_size,
                    "sha256": file_hash,
                }

        # Save manifest
        with open(self.output_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print("✓ Manifest and brief generated.")

        # 5. Create pack format based on mode
        pack_mode = config.get("pack_mode", "semantic")

        if pack_mode == "semantic":
            # Semantic mode: Keep files as-is (lightweight, human-readable)
            print("✓ Semantic pack assembled (loose files format)")
            print(f"   - Main summary: {self.output_path / 'repoBrief.md'}")
            print(f"   - Artifacts: {len(manifest['artifacts'])} files")

        elif pack_mode in ["hybrid", "full"]:
            # Hybrid/Full mode: Create a comprehensive zip archive
            pack_name = f"reposynth_{self.repo_path.name}_{pack_mode}.zip"
            zip_path = self.output_path.parent / pack_name

            print(f"Creating {pack_mode} pack archive: {pack_name}...")

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                # Add all artifacts from the pack directory
                for artifact_path in self.output_path.glob("*"):
                    if artifact_path.is_file():
                        arcname = f"pack/{artifact_path.name}"
                        zf.write(artifact_path, arcname=arcname)
                        print(f"   ✓ Added: {artifact_path.name}")
                    elif artifact_path.is_dir() and artifact_path.name == "ast_raw":
                        # Include AST files in full mode
                        for ast_file in artifact_path.glob("*"):
                            if ast_file.is_file():
                                arcname = f"pack/ast_raw/{ast_file.name}"
                                zf.write(ast_file, arcname=arcname)
                        print(f"   ✓ Added: ast_raw/ directory ({len(list(artifact_path.glob('*')))} files)")

                # Add README to the zip
                readme_content = f"""# RepoSynth {pack_mode.capitalize()} Pack

Repository: {self.repo_path.name}
Generated: {manifest['createdAt']}
Commit: {self.commit_hash}
Mode: {pack_mode}

## Contents

This archive contains a complete analysis pack for the repository.

### Key Files:
- **repoBrief.md**: Architectural summary and analysis (includes full source code for public APIs)
- **manifest.json**: Artifact metadata and checksums
- **name_registry.json**: Symbol definitions and locations
- **import_graph.json**: Module dependency graph
- **analysis.sqlite**: Complexity metrics database
- **vectors.faiss**: Semantic embeddings index
- **security_report.json**: Security scan results (if enabled)
"""

                if pack_mode == "hybrid":
                    readme_content += "\n- **source_spans.json**: Source code spans for public APIs (JSON format)\n"
                    readme_content += "- **variable_registry.json**: Variable tracking information\n"
                elif pack_mode == "full":
                    readme_content += "\n- **source_spans.json**: Source code spans for public APIs (JSON format)\n"
                    readme_content += "- **variable_registry.json**: Variable tracking information\n"
                    readme_content += "- **ast_raw/**: Raw AST files for all parsed sources\n"

                zf.writestr("README.md", readme_content)

            print(f"✓ {pack_mode.capitalize()} pack archive created: {zip_path}")
            print(f"   Archive size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
            print(f"   Extract and review the repoBrief.md file for a summary.")

        print("\n✓ Final pack assembly complete!")
