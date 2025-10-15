# --- FILE: packages/python-orchestrator/orchestrator/pipeline_runner.py ---

import subprocess
import json
import uuid
import threading
import queue
from pathlib import Path
import sys
import sqlite3

# Third-party imports
import numpy as np
import faiss
from radon.complexity import cc_visit
from sentence_transformers import SentenceTransformer

# Local imports
from .parser_client import ParserClient
from .language_adapter import get_adapter

class Pipeline:
    def __init__(self, repo_path: str, output_path: str, daemon_path: str):
        self.repo_path = Path(repo_path).resolve()
        self.output_path = Path(output_path)
        self.daemon_path = daemon_path
        
        # Create output directories
        self.ast_dir = self.output_path / "ast_raw"
        self.output_path.mkdir(exist_ok=True)
        self.ast_dir.mkdir(exist_ok=True)

        self.name_registry = {}
        self.import_graph = {}
        self.db_conn = sqlite3.connect(self.output_path / "analysis.sqlite")
        self._init_db()

    def _init_db(self):
        cursor = self.db_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS complexity (
                id INTEGER PRIMARY KEY,
                file_path TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                complexity INTEGER NOT NULL,
                UNIQUE(file_path, symbol_name)
            )
        ''')
        self.db_conn.commit()

    def run(self):
        print("--- Running Stage 1: Parsing Repository ---")
        self.run_parsing()
        
        print("\n--- Running Stage 2: Building Graphs & Name Registry ---")
        self.build_graphs_and_registry()
        
        print("\n--- Running Stage 3: Static Analysis ---")
        self.run_static_analysis()

        print("\n--- Running Stage 4: Generating Embeddings ---")
        self.generate_embeddings()

        print("\n--- Pipeline Finished Successfully! ---")
        print(f"Artifacts saved in: {self.output_path}")

    def run_parsing(self):
        client = ParserClient(daemon_path=self.daemon_path)
        try:
            client.parse_repository(repo_path=str(self.repo_path), output_dir=str(self.ast_dir))
        finally:
            client.shutdown()

    def build_graphs_and_registry(self):
        for ast_file in self.ast_dir.glob("*.jsonl"):
            try:
                with open(ast_file, 'r') as f:
                    ast_nodes = [json.loads(line) for line in f]
                
                # Infer original file path from a manifest or filename convention
                # For now, a bit of a hack to get the original path
                original_file_name = ast_file.name.split('_')[0]
                original_file = next(self.repo_path.glob(f"**/{original_file_name}"))
                relative_path = str(original_file.relative_to(self.repo_path))

                with open(original_file, 'r', encoding='utf-8') as f:
                    source_code = f.read()

                adapter = get_adapter(original_file)
                if not adapter:
                    continue

                # Get definitions and add to name registry
                definitions = adapter.get_definitions(ast_nodes, source_code)
                for a_def in definitions:
                    fqn = f"{relative_path}:{a_def['name']}"
                    self.name_registry[fqn] = {
                        "kind": a_def['kind'],
                        "file_path": relative_path,
                        "start_byte": a_def['start_byte'],
                        "end_byte": a_def['end_byte'],
                        "is_public": a_def.get('is_public', False),
                    }

                # Get imports for the graph
                imports = adapter.get_imports(ast_nodes, source_code)
                self.import_graph[relative_path] = imports
                print(f"Processed graphs for {relative_path}")

            except Exception as e:
                print(f"Failed to process graph for {ast_file.name}: {e}", file=sys.stderr)
        
        with open(self.output_path / "name_registry.json", "w") as f:
            json.dump(self.name_registry, f, indent=2)
        with open(self.output_path / "import_graph.json", "w") as f:
            json.dump(self.import_graph, f, indent=2)

    def run_static_analysis(self):
        cursor = self.db_conn.cursor()
        for fqn, data in self.name_registry.items():
            if data['kind'] != 'function_definition' or not data['file_path'].endswith('.py'):
                continue
            
            file_path = self.repo_path / data['file_path']
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # Get complexity for just the function's code block
                func_code = source_code[data['start_byte']:data['end_byte']]
                complexity_scores = cc_visit(func_code)
                
                if complexity_scores:
                    # cc_visit returns a list, we take the first (and only) for the block
                    complexity = complexity_scores[0].complexity
                    symbol_name = fqn.split(':')[-1]
                    cursor.execute(
                        "INSERT OR IGNORE INTO complexity (file_path, symbol_name, complexity) VALUES (?, ?, ?)",
                        (data['file_path'], symbol_name, complexity)
                    )
            except Exception as e:
                print(f"Radon analysis failed for {fqn}: {e}", file=sys.stderr)
        
        self.db_conn.commit()
        print("Static analysis metrics saved to analysis.sqlite")

    def generate_embeddings(self):
        # 1. Collect public APIs to embed
        public_apis = {}
        for fqn, data in self.name_registry.items():
            if data.get('is_public', False):
                file_path = self.repo_path / data['file_path']
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        source_code = f.read()
                    
                    # Create a snippet for embedding (e.g., function signature)
                    snippet = source_code[data['start_byte']:data['end_byte']].split('\n')[0]
                    public_apis[fqn] = snippet
                except Exception:
                    continue # Skip if file can't be read

        if not public_apis:
            print("No public APIs found to embed. Skipping.")
            return

        # 2. Load model and generate embeddings
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        fqns = list(public_apis.keys())
        snippets = list(public_apis.values())
        
        print(f"Generating embeddings for {len(fqns)} public symbols...")
        embeddings = model.encode(snippets, show_progress_bar=True)
        
        # 3. Create and save FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIDMap(index) # Allows mapping to our own IDs
        
        ids = np.array(range(len(fqns)))
        index.add_with_ids(embeddings, ids)
        
        faiss.write_index(index, str(self.output_path / "vectors.faiss"))
        
        # 4. Save the ID -> FQN mapping
        id_map = {i: fqn for i, fqn in enumerate(fqns)}
        with open(self.output_path / "vector_ids.json", "w") as f:
            json.dump(id_map, f, indent=2)

        print("Embeddings and FAISS index saved successfully.")