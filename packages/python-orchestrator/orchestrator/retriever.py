import json
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- 1. Core Keyword Search Logic ---
def retrieve_relevant_symbols(query: str, name_registry: Dict[str, Any], max_items: int = 5) -> List[Dict[str, Any]]:
    """
    Scans file paths and symbol names for keyword matches.
    """
    query_lower = query.lower()
    # Split by spaces to get keywords
    query_words = [w for w in query_lower.split() if len(w) > 2] # Ignore small words
    
    matches = []
    seen_files = set()

    for symbol_fqn, symbol_data in name_registry.items():
        file_path = symbol_data.get('file_path', '')
        # Extract name from fqn if not present
        symbol_name = symbol_data.get('name', symbol_fqn.split('.')[-1]).lower()

        score = 0

        # Check for matches
        for word in query_words:
            if word in file_path.lower(): score += 3  # File name match is high value
            if word in symbol_name: score += 2        # Function name match
            if word in symbol_fqn.lower(): score += 2  # FQN match

        if score > 0:
            # We want to return files, so we deduplicate by file path
            if file_path not in seen_files:
                data = symbol_data.copy()  # Copy to avoid mutating registry
                data['fqn'] = symbol_fqn  # Include the fully qualified name
                data['name'] = symbol_fqn.split('.')[-1]  # Extract simple name
                matches.append((score, data))
                seen_files.add(file_path)

    # Sort by score descending
    matches.sort(reverse=True, key=lambda x: x[0])
    return [m[1] for m in matches[:max_items]]

# --- 2. AI Semantic Search Logic ---
_EMBEDDING_MODEL = None

def get_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Use a small, fast model
            _EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            print("Warning: sentence-transformers not installed.", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Warning: Failed to load AI model: {e}", file=sys.stderr)
            return None
    return _EMBEDDING_MODEL

def semantic_search(query: str, pack_dir: Path, max_items: int = 5):
    index_path = pack_dir / "vectors.faiss"
    ids_path = pack_dir / "vector_ids.json"
    registry_path = pack_dir / "name_registry.json"

    if not (index_path.exists() and ids_path.exists()): return []

    try:
        import faiss
        model = get_model()
        if not model: return []
        
        index = faiss.read_index(str(index_path))
        with open(ids_path, 'r') as f: id_map = json.load(f)
        with open(registry_path, 'r') as f: registry = json.load(f)

        vec = model.encode([query])
        distances, indices = index.search(np.array(vec).astype('float32'), max_items)
        
        results = []
        seen = set()
        for idx in indices[0]:
            if str(idx) in id_map:
                fqn = id_map[str(idx)]
                if fqn in registry:
                    data = registry[fqn].copy()  # Copy to avoid mutating registry
                    data['fqn'] = fqn  # Include the fully qualified name
                    data['name'] = fqn.split('.')[-1]  # Extract simple name
                    fpath = data.get('file_path')
                    if fpath and fpath not in seen:
                        results.append(data)
                        seen.add(fpath)
        return results
    except Exception as e:
        print(f"Semantic search error: {e}", file=sys.stderr)
        return []

# --- 3. The Panic Fallback (New!) ---
def get_fallback_files(name_registry: Dict[str, Any], max_items: int = 5) -> List[Dict[str, Any]]:
    """
    If search finds nothing, return the most important files 
    (usually index, main, schema, etc.)
    """
    important_keywords = ['index', 'main', 'app', 'schema', 'types', 'server']
    results = []
    seen = set()
    
    # First pass: look for important filenames
    for fqn, data in name_registry.items():
        fpath = data.get('file_path', '')
        if fpath in seen: continue
        
        for kw in important_keywords:
            if kw in fpath.lower():
                results.append(data)
                seen.add(fpath)
                break
        if len(results) >= max_items: return results

    # Second pass: just take the first few files
    for fqn, data in name_registry.items():
        fpath = data.get('file_path', '')
        if fpath not in seen:
            results.append(data)
            seen.add(fpath)
        if len(results) >= max_items: return results
        
    return results

# --- 4. The Main Hybrid Search Function ---
def hybrid_search(query: str, pack_dir: Path, max_items: int = 5, registry: Optional[Dict[str, Any]] = None):
    """
    Hybrid search with SEMANTIC-FIRST priority.

    Order:
    1. Semantic search (FAISS) - primary, understands meaning
    2. Keyword search - fallback/boost for exact matches
    3. Important files fallback - panic mode if nothing found
    """
    print(f"DEBUG: Searching for '{query}'", file=sys.stderr)

    # Load Registry if not provided
    if registry is None:
        try:
            if pack_dir.is_file() and pack_dir.suffix == '.json':
                with open(pack_dir, 'r') as f:
                    data = json.load(f)
                    registry = data.get('name_registry', {})
            else:
                with open(pack_dir / "name_registry.json", 'r') as f:
                    registry = json.load(f)
        except Exception as e:
            print(f"DEBUG: Failed to load registry: {e}", file=sys.stderr)
            return []

    results = []
    seen_paths = set()

    # 1. SEMANTIC SEARCH FIRST (primary - understands meaning)
    if pack_dir and pack_dir.is_dir():
        try:
            semantic = semantic_search(query, pack_dir, max_items * 2)  # Get extra for merging
            print(f"DEBUG: Semantic found {len(semantic)}", file=sys.stderr)

            for item in semantic:
                fpath = item.get('file_path')
                if fpath and fpath not in seen_paths:
                    results.append(item)
                    seen_paths.add(fpath)
        except Exception as e:
            print(f"DEBUG: Semantic search failed: {e}", file=sys.stderr)
    else:
        print("DEBUG: Skipping semantic search (pack_dir is not a directory)", file=sys.stderr)

    # 2. KEYWORD SEARCH (fallback/boost - for exact matches)
    # Only use if semantic found nothing, or to boost with exact matches
    if len(results) < max_items:
        keyword_results = retrieve_relevant_symbols(query, registry, max_items)
        print(f"DEBUG: Keyword found {len(keyword_results)}", file=sys.stderr)

        for item in keyword_results:
            fpath = item.get('file_path')
            if fpath and fpath not in seen_paths:
                results.append(item)
                seen_paths.add(fpath)

    # 3. FALLBACK: If still nothing, return important files
    if len(results) == 0:
        print("DEBUG: No results found. Using fallback.", file=sys.stderr)
        results = get_fallback_files(registry, max_items)

    return results[:max_items]