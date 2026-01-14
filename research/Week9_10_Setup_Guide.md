# Week 9-10: Full Experiments Setup Guide

This guide walks through setting up real repository data for the CCE evaluation experiments.

---

## Prerequisites

1. **RepoSynth pipeline built**: Ensure the Rust daemon is compiled
   ```bash
   cd packages/rust-parser-daemon
   cargo build --release
   ```

2. **Python environment**: Orchestrator package installed
   ```bash
   cd packages/python-orchestrator
   pip install -e .
   ```

---

## Step 1: Clone Target Repositories

Clone three well-known Python repositories for evaluation:

```bash
cd C:\Users\rajka\reposynth\research\repos

# Flask - Lightweight WSGI web framework
git clone --depth 1 https://github.com/pallets/flask.git

# FastAPI - Modern async web framework
git clone --depth 1 https://github.com/tiangolo/fastapi.git

# Requests - HTTP library
git clone --depth 1 https://github.com/psf/requests.git
```

**Note**: Using `--depth 1` for shallow clone (faster, smaller).

---

## Step 2: Index Each Repository

Run the RepoSynth pipeline on each repository to create semantic packs:

```bash
cd C:\Users\rajka\reposynth

# Index Flask
python -m orchestrator --repo research/repos/flask --mode semantic
move pack research\packs\flask_pack

# Index FastAPI
python -m orchestrator --repo research/repos/fastapi --mode semantic
move pack research\packs\fastapi_pack

# Index Requests
python -m orchestrator --repo research/repos/requests --mode semantic
move pack research\packs\requests_pack
```

**Alternative (from Git URL directly)**:
```bash
python -m orchestrator --repo https://github.com/pallets/flask.git --mode semantic
```

---

## Step 3: Verify Pack Contents

Each pack should contain these files:

```
research/packs/<repo>_pack/
├── name_registry.json    # Symbol definitions (functions, classes, etc.)
├── import_graph.json     # Module dependency graph
├── vectors.faiss         # FAISS semantic search index
├── vector_ids.json       # Mapping of vector IDs to symbols
├── repoBrief.md          # Architectural summary with source code
├── manifest.json         # Pack metadata and checksums
├── source_spans.json     # Source code spans for public APIs
└── README.md             # Pack documentation
```

**Verify a pack**:
```bash
ls research/packs/flask_pack/
```

---

## Step 4: Validate Semantic Search

Quick test that semantic search works:

```python
import faiss
import json
from sentence_transformers import SentenceTransformer

pack_path = "research/packs/flask_pack"

# Load index
index = faiss.read_index(f"{pack_path}/vectors.faiss")
with open(f"{pack_path}/vector_ids.json") as f:
    vector_ids = json.load(f)

# Test query
model = SentenceTransformer('all-MiniLM-L6-v2')
query_vec = model.encode(["How does Flask handle routing?"])
D, I = index.search(query_vec, k=5)

print("Top 5 results:")
for i, idx in enumerate(I[0]):
    print(f"  {i+1}. {vector_ids[str(idx)]}")
```

---

## Step 5: Run Experiment Notebook

Open the experiment notebook in Google Colab or Jupyter:

```bash
# Local Jupyter
jupyter notebook research/Week9_10_Full_Experiments.ipynb

# Or upload to Colab
# Upload the notebook and pack folders to Google Drive
```

---

## Expected Timeline

| Step | Duration | Notes |
|------|----------|-------|
| Clone repos | ~2 min | Shallow clones |
| Index Flask | ~3-5 min | ~60 Python files |
| Index FastAPI | ~5-8 min | ~150 Python files |
| Index Requests | ~2-3 min | ~40 Python files |
| Run experiments | ~30-60 min | Depends on GPU |

---

## Troubleshooting

### "Daemon executable not found"
```bash
cd packages/rust-parser-daemon
cargo build --release
```

### "FAISS not installed"
```bash
pip install faiss-cpu  # or faiss-gpu for GPU support
```

### "SentenceTransformers error"
```bash
pip install sentence-transformers
```

### Pack directory not created
Check that the pipeline completed successfully. Look for errors in the console output.

---

## Pack Sizes (Approximate)

| Repository | Files | Symbols | Pack Size |
|------------|-------|---------|-----------|
| Flask | ~60 | ~500 | ~5 MB |
| FastAPI | ~150 | ~1200 | ~15 MB |
| Requests | ~40 | ~300 | ~3 MB |

---

## Next Steps

After creating packs:
1. Update `research/benchmarks/benchmark_v1.json` with real file paths from `name_registry.json`
2. Run `Week9_10_Full_Experiments.ipynb` with the real packs
3. Generate statistical results and paper figures
