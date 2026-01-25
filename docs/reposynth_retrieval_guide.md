# RepoSynth Retrieval Guide

A step-by-step guide for using RepoSynth's retrieval system in notebooks and experiments.

---

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start)
2. [Step-by-Step Setup](#step-by-step-setup)
3. [Using in Notebooks](#using-in-notebooks)
4. [API Reference](#api-reference)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

**Goal:** Get RepoSynth retrieval working in under 5 minutes.

```bash
# Step 1: Navigate to reposynth directory
cd C:\Users\rajka\reposynth

# Step 2: Create a pack for your repository (example: cerberus)
python -m orchestrator ./cerberus --with-embeddings --store-spans

# Step 3: Use in Python
python -c "
from orchestrator.retrieval import RepoSynthRetriever
retriever = RepoSynthRetriever('./output/cerberus/pack')
results = retriever.retrieve('validation', top_k=3)
print(f'Found {len(results)} files')
"
```

---

## Step-by-Step Setup

### Step 1: Clone a Repository (if needed)

```bash
# Example: Clone the Cerberus validation library
git clone https://github.com/pyeve/cerberus.git --depth 1
```

### Step 2: Run the RepoSynth Pipeline

```bash
# IMPORTANT: Use BOTH flags for best results
python -m orchestrator ./cerberus --with-embeddings --store-spans
```

**What each flag does:**

| Flag | Purpose | Required? |
|------|---------|-----------|
| `--with-embeddings` | Creates FAISS index for semantic search | Yes |
| `--store-spans` | Stores full source code for retrieval | Yes (strongly recommended) |

### Step 3: Verify the Pack Was Created

After running the pipeline, check that these files exist:

```
output/cerberus/pack/
    name_registry.json    <-- Symbol metadata (REQUIRED)
    vectors.faiss         <-- FAISS embeddings (REQUIRED)
    vector_ids.json       <-- Maps FAISS IDs to symbols
    source_spans.json     <-- Full source code (REQUIRED for quality)
```

**Quick verification:**

```bash
# Check pack exists
ls ./output/cerberus/pack/

# Should show: name_registry.json, vectors.faiss, source_spans.json, etc.
```

---

## Using in Notebooks

### Option A: Week9_CCE_Ablation_Study.ipynb (Already Integrated)

The notebook has been updated to automatically use RepoSynth when available.

**Steps:**

1. **Create the pack first** (see Step-by-Step Setup above)

2. **Open the notebook** and find Cell A2 (RepoSynth Integration)

3. **Set the pack path:**
   ```python
   # Cell A2 - Update this path to your pack location
   REPOSYNTH_PACK_DIR = '../output/cerberus/pack'
   ```

4. **Run all cells** - The notebook will automatically:
   - Detect the pack exists
   - Import `RepoSynthRetriever`
   - Use hybrid search instead of pure embedding search

**What you'll see when RepoSynth is active:**
```
======================================================================
REPOSYNTH MODE: Using pre-computed embeddings
======================================================================
Pack directory: ../output/cerberus/pack
RepoSynthRetriever: Loaded 14 files with full source
RepoSynthRetriever initialized with 318 symbols
```

**What you'll see if pack is missing (fallback mode):**
```
======================================================================
NOTEBOOK MODE: Using runtime EmbeddingRetriever
======================================================================
To use RepoSynth, create a pack first:
  python -m orchestrator ./cerberus --with-embeddings --store-spans
```

### Option B: Any Custom Notebook

Add this code to use RepoSynth retrieval:

```python
# Cell 1: Setup
import sys
sys.path.insert(0, '../packages/python-orchestrator')

from orchestrator.retrieval import RepoSynthRetriever

# Cell 2: Initialize retriever
pack_dir = '../output/cerberus/pack'  # <-- UPDATE THIS PATH
retriever = RepoSynthRetriever(pack_dir)

# Cell 3: Retrieve code
results = retriever.retrieve("How does validation work?", top_k=3)

for r in results:
    print(f"File: {r['source']}")
    print(f"Content: {r['content'][:200]}...")
    print("---")
```

### Option C: With CCE Adaptive Generator

For full CCE-triggered retrieval during generation:

```python
from orchestrator.generation import create_reposynth_generator

# Assumes model and tokenizer are already loaded
generator = create_reposynth_generator(
    model=model,
    tokenizer=tokenizer,
    pack_dir='../output/cerberus/pack',
    uncertainty_method='cce',
    threshold=0.3,
    max_retrievals=3
)

result = generator.generate("Implement input validation")
print(result.code)
```

---

## API Reference

### RepoSynthRetriever

```python
from orchestrator.retrieval import RepoSynthRetriever

retriever = RepoSynthRetriever(pack_dir: str)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `retrieve(query, top_k=3)` | Find relevant files for query | `List[Dict]` with `source`, `content`, `score` |
| `reset()` | Clear deduplication tracking | None |
| `get_file_content(path)` | Get full source for one file | `str` or `None` |
| `get_file_list()` | List all files in pack | `List[str]` |
| `get_file_list_context()` | Formatted file list for prompts | `str` |

**Example return value from `retrieve()`:**

```python
[
    {
        'source': 'cerberus/validator.py',
        'content': '"""Cerberus validator module..."""\n\nclass Validator:\n    ...',
        'score': 1.0
    },
    {
        'source': 'cerberus/schema.py',
        'content': '"""Schema definitions..."""\n\ndef schema():\n    ...',
        'score': 1.0
    }
]
```

---

## How RepoSynth Retrieval Works

```
Query: "How does validation work?"
         │
         ▼
┌─────────────────────────────────────────┐
│     SEMANTIC-FIRST HYBRID SEARCH         │
├─────────────────────────────────────────┤
│ 1. Semantic Search (FAISS) - PRIMARY     │
│    - Embeds query with sentence-transformer│
│    - Finds similar code by meaning       │
│    - Catches synonyms/related concepts   │
│    - Best for natural language queries   │
│                                          │
│ 2. Keyword Search - FALLBACK/BOOST       │
│    - Only if semantic found < max_items  │
│    - Scans file paths and symbol names   │
│    - Catches exact matches semantic missed│
│                                          │
│ 3. Important Files - PANIC FALLBACK      │
│    - Only if both found nothing          │
│    - Returns index, main, schema, etc.   │
│                                          │
│ 4. Enrich with Full Source               │
│    - Loads complete file content         │
│    - From source_spans.json              │
└─────────────────────────────────────────┘
         │
         ▼
    Results: [{'source': 'validator.py', 'content': '...full code...'}]
```

---

## Troubleshooting

### Problem: "Pack directory not found"

```
ValueError: Pack directory not found: ./output/cerberus/pack
```

**Solution:** Create the pack first:
```bash
python -m orchestrator ./cerberus --with-embeddings --store-spans
```

---

### Problem: "No source_spans.json found" warning

```
WARNING: No source_spans.json found!
Retrieval quality will be degraded (only symbol metadata available).
```

**Solution:** Re-run pipeline with `--store-spans`:
```bash
python -m orchestrator ./cerberus --with-embeddings --store-spans
```

---

### Problem: "No results returned"

**Check these things:**

1. **Pack has files:**
   ```python
   retriever = RepoSynthRetriever('./output/cerberus/pack')
   print(f"Files in pack: {len(retriever.get_file_list())}")
   # Should be > 0
   ```

2. **Query has meaningful words:**
   ```python
   # BAD - words too short (filtered out)
   retriever.retrieve("a b c")

   # GOOD - meaningful keywords
   retriever.retrieve("validation schema")
   ```

3. **FAISS index exists:**
   ```bash
   ls ./output/cerberus/pack/vectors.faiss
   # Should exist
   ```

---

### Problem: Import errors

```
ModuleNotFoundError: No module named 'orchestrator'
```

**Solution:** Add orchestrator to Python path:
```python
import sys
sys.path.insert(0, '../packages/python-orchestrator')  # Adjust path as needed

from orchestrator.retrieval import RepoSynthRetriever
```

---

### Problem: Notebook uses EmbeddingRetriever instead of RepoSynth

**Check Cell A2 in the notebook:**

1. Is `REPOSYNTH_PACK_DIR` set correctly?
2. Does the pack directory exist?
3. Does it contain `name_registry.json`?

```python
# Debug in notebook
import os
print(f"Pack exists: {os.path.exists(REPOSYNTH_PACK_DIR)}")
print(f"Registry exists: {os.path.exists(f'{REPOSYNTH_PACK_DIR}/name_registry.json')}")
```

---

## Comparison: RepoSynth vs Notebook Retriever

| Feature | Notebook EmbeddingRetriever | RepoSynth Retriever |
|---------|----------------------------|---------------------|
| Setup time | None (runtime) | One-time pipeline run |
| Embedding speed | Slow (computes each run) | Fast (pre-computed) |
| Search type | Semantic only | Hybrid (keyword + semantic) |
| Fallback | None | Returns important files |
| Storage | Memory only | Persistent on disk |
| Reproducibility | Varies | Consistent |

**When to use each:**

- **RepoSynth**: Production, experiments, when you'll query the same codebase multiple times
- **Notebook**: Quick prototyping, one-off analysis, when you don't want to run the pipeline

---

## Complete Example: End-to-End

```bash
# Terminal: Create pack
cd C:\Users\rajka\reposynth
git clone https://github.com/pyeve/cerberus.git --depth 1
python -m orchestrator ./cerberus --with-embeddings --store-spans
```

```python
# Notebook: Use pack
import sys
sys.path.insert(0, '../packages/python-orchestrator')

from orchestrator.retrieval import RepoSynthRetriever

# Initialize
retriever = RepoSynthRetriever('../output/cerberus/pack')
print(f"Loaded {len(retriever.get_file_list())} files")

# Search
results = retriever.retrieve("validate input data", top_k=3)

# Display
for i, r in enumerate(results, 1):
    print(f"\n--- Result {i}: {r['source']} ---")
    print(r['content'][:500])
```

**Expected output:**
```
RepoSynthRetriever: Loaded 14 files with full source
RepoSynthRetriever initialized with 318 symbols
Loaded 14 files

--- Result 1: cerberus/validator.py ---
"""
    Extensible validation for Python dictionaries.
    ...
"""

class Validator:
    def validate(self, document, schema=None, ...):
        ...
```
