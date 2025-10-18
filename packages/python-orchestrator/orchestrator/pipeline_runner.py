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

        # CORRECT: No cleanup in __init__. Only define paths and create directories.
        self.cache_dir = self.repo_path / ".reposynth_cache" # Corrected typo
        self.output_path.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

        self.ast_dir = self.output_path / "ast_raw"
        self.ast_dir.mkdir(exist_ok=True)
        self.db_path = self.output_path / "analysis.sqlite"

        # These will be populated by the pipeline stages
        self.name_registry = {}
        self.import_graph = {}
        self.variable_registry = defaultdict(list)
        self.definitions_by_file = defaultdict(list)

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
        commit_hash = self._get_commit_hash()
        
        if config.get("run_parsing"):
            print("--- Running Stage 1: Parsing Repository ---")
            self.run_parsing(commit_hash=commit_hash)
        
        if config.get("build_graphs"):
            print("\n--- Running Stage 2: Building Graphs & Name Registry ---")
            self.build_graphs_and_registry()
        
        if config.get("build_variable_registry"):
            print("\n--- Running Stage 2.5 (Hybrid): Building Variable Registry ---")
            self.build_variable_registry()

        if config.get("run_analysis"):
            print("\n--- Running Stage 3: Static Analysis ---")
            self.run_static_analysis()

        if config.get("run_embeddings"):
            print("\n--- Running Stage 4: Generating Embeddings ---")
            self.generate_embeddings()

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

    def build_graphs_and_registry(self):
        self.definitions_by_file = defaultdict(list)
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
            return
        with open(manifest_path, "r", encoding="utf-8") as f:
            ast_manifest = json.load(f)

        for ast_filename, original_filepath_str in ast_manifest.items():
            ast_file = self.ast_dir / ast_filename
            original_file = Path(original_filepath_str)

            if not ast_file.exists():
                continue

            try:
                with open(ast_file, "r", encoding="utf-8") as f:
                    ast_nodes = [json.loads(line) for line in f]

                relative_path = str(original_file.relative_to(self.repo_path))

                with open(original_file, "r", encoding="utf-8") as f:
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
                    # Skip built-in/third-party library imports for now
                    if not imp.startswith("."):
                        continue

                    try:
                        resolved_path = (current_dir / imp).resolve()
                        # Check if the resolved path is within our repo
                        if self.repo_path in resolved_path.parents:
                            # Try to find a file with .ts, .js, .py extensions
                            found = False
                            for ext in [".ts", ".js", ".py", ".tsx", ".jsx"]:
                                if (
                                    resolved_path.parent / f"{resolved_path.name}{ext}"
                                ).exists():
                                    resolved_imports.append(
                                        str(
                                            (
                                                resolved_path.parent
                                                / f"{resolved_path.name}{ext}"
                                            ).relative_to(self.repo_path)
                                        )
                                    )
                                    found = True
                                    break
                            if not found and (resolved_path / "__init__.py").exists():
                                resolved_imports.append(
                                    str(
                                        (resolved_path / "__init__.py").relative_to(
                                            self.repo_path
                                        )
                                    )
                                )

                    except Exception:
                        continue  # Ignore resolution errors

                self.import_graph[relative_path] = resolved_imports

            except Exception as e:
                print(
                    f"Failed to process graph for {ast_filename} (original: {relative_path}): {e}",
                    file=sys.stderr,
                )

        with open(self.output_path / "name_registry.json", "w") as f:
            json.dump(self.name_registry, f, indent=2)
        with open(self.output_path / "import_graph.json", "w") as f:
            json.dump(self.import_graph, f, indent=2)

        print("Finished building graphs and name registry.")

    def build_variable_registry(self):
        manifest_path = self.ast_dir / "ast_manifest.json"
        if not manifest_path.exists():
            print(
                "Warning: ast_manifest.json not found. Skipping variable registry.",
                file=sys.stderr,
            )
            return

        with open(manifest_path, "r", encoding="utf-8") as f:
            ast_manifest = json.load(f)

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
                with open(original_file, "r", encoding="utf-8") as f:
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
        print("Variable registry saved.")

    def store_spans(self):
        zip_path = self.output_path / "spans.zip"
        span_manifest = defaultdict(list)
        files_to_include = set()

        # Identify files and byte ranges for all public APIs
        for fqn, data in self.name_registry.items():
            if data.get("is_public", False):
                file_path = data["file_path"]
                files_to_include.add(file_path)
                span_manifest[file_path].append([data["start_byte"], data["end_byte"]])

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write the manifest to the zip
            zf.writestr("manifest.json", json.dumps(span_manifest, indent=2))

            # Add the raw source files to the zip
            for file_path_str in files_to_include:
                try:
                    full_path = self.repo_path / file_path_str
                    # The arcname is the path inside the zip file
                    zf.write(full_path, arcname=file_path_str)
                except FileNotFoundError:
                    continue

        print(f"Spans and source files saved to {zip_path}")

    def run_static_analysis(self):
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

            files_to_analyze = {}
            for fqn, data in self.name_registry.items():
                if "function" in data["kind"] or "method" in data["kind"]:
                    if data["file_path"].endswith(".py"):
                        file_path = data["file_path"]
                        if file_path not in files_to_analyze:
                            files_to_analyze[file_path] = []

                        symbol_name = fqn.split(":")[-1]
                        files_to_analyze[file_path].append(symbol_name)

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
        
        print("Static analysis metrics saved to analysis.sqlite")

    def generate_embeddings(self):
        # 1. Collect public APIs to embed
        public_apis = {}
        for fqn, data in self.name_registry.items():
            if data.get("is_public", False):
                file_path = self.repo_path / data["file_path"]
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
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

        print("Embeddings and FAISS index saved successfully.")

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
                        with open(self.repo_path / path, "r", encoding="utf-8") as f:
                            source_code = f.read()
                        snippet = source_code[data["start_byte"] : data["end_byte"]]
                        first_line = snippet.split("\n")[0].strip()
                        if first_line:
                            public_apis.append(f"`{first_line}`")
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
        """Generates the final repoBrief.md and manifest.json."""

        # 1. Generate the brief
        brief_content = self.generate_deterministic_brief()
        with open(self.output_path / "repoBrief.md", "w", encoding="utf-8") as f:
            f.write(brief_content)
        if config.get("store_spans"):
            self.store_spans()

        # 2. Generate the manifest
        manifest = {
            "createdAt": datetime.datetime.utcnow().isoformat(),
            "repoPath": str(self.repo_path),
            "packVersion": "1.0",
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

        with open(self.output_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print("Final pack assembled with repoBrief.md and manifest.json.")
