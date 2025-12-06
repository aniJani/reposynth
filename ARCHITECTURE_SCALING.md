# ContextLoom: Architecture & Scaling Reference

This document catalogs all architectural choices, model sizes, embedding dimensions, and scaling parameters that affect system capacity, performance, and resource requirements.

---

## Table of Contents

1. [Embedding Models & Dimensions](#1-embedding-models--dimensions)
2. [Language Model Architecture](#2-language-model-architecture)
3. [Vector Index Architecture](#3-vector-index-architecture)
4. [Token Classification Architecture](#4-token-classification-architecture)
5. [Retrieval System Architecture](#5-retrieval-system-architecture)
6. [Context Window Architecture](#6-context-window-architecture)
7. [Entropy Computation Architecture](#7-entropy-computation-architecture)
8. [Caching Architecture](#8-caching-architecture)
9. [Batch Processing & Parallelism](#9-batch-processing--parallelism)
10. [Memory Architecture](#10-memory-architecture)
11. [Storage Architecture](#11-storage-architecture)
12. [Network Architecture](#12-network-architecture)
13. [Scaling Dimensions Summary](#13-scaling-dimensions-summary)
14. [Resource Requirements Matrix](#14-resource-requirements-matrix)
15. [Scaling Recommendations](#15-scaling-recommendations)

---

## 1. Embedding Models & Dimensions

### 1.1 Sentence Transformer Models

| Model | Dimensions | Parameters | Speed | Quality | Use Case |
|-------|------------|------------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | 22M | Fast | Good | Default, development |
| `all-MiniLM-L12-v2` | 384 | 33M | Medium | Better | Balanced |
| `all-mpnet-base-v2` | 768 | 110M | Slow | Best | Production quality |
| `all-distilroberta-v1` | 768 | 82M | Medium | Good | Alternative |
| `paraphrase-MiniLM-L6-v2` | 384 | 22M | Fast | Good | Paraphrase-focused |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | 22M | Fast | Good | QA-optimized |

### 1.2 Code-Specific Embedding Models

| Model | Dimensions | Parameters | Languages | Use Case |
|-------|------------|------------|-----------|----------|
| `krlvi/sentence-t5-base-nlpl-code` | 768 | 110M | Multi | Code search |
| `flax-sentence-embeddings/st-codesearch-distilroberta-base` | 768 | 82M | Multi | Code search |
| `microsoft/codebert-base` | 768 | 125M | 6 langs | Code understanding |
| `microsoft/graphcodebert-base` | 768 | 125M | 6 langs | Code + data flow |
| `Salesforce/codet5-base` | 768 | 220M | 8 langs | Code generation |
| `bigcode/starencoder` | 1024 | 125M | 80+ langs | Code embeddings |

### 1.3 Embedding Dimension Choices

| Dimension | Memory/Vector | Index Size (1M vectors) | Search Speed | Quality |
|-----------|---------------|-------------------------|--------------|---------|
| 128 | 512 bytes | ~512 MB | Fastest | Lower |
| 256 | 1 KB | ~1 GB | Fast | Moderate |
| 384 | 1.5 KB | ~1.5 GB | Fast | Good |
| 512 | 2 KB | ~2 GB | Medium | Good |
| 768 | 3 KB | ~3 GB | Medium | Better |
| 1024 | 4 KB | ~4 GB | Slower | Best |
| 1536 | 6 KB | ~6 GB | Slow | Highest |
| 4096 | 16 KB | ~16 GB | Slowest | Maximum |

### 1.4 Embedding Configuration

```yaml
embedding_config:
  # Model selection
  model_name: "all-MiniLM-L6-v2"  # or path to custom model
  model_type: "sentence_transformer"  # or "huggingface", "openai", "custom"

  # Dimensions
  embedding_dim: 384  # Must match model output
  max_seq_length: 512  # Maximum input tokens

  # Normalization
  normalize_embeddings: true  # L2 normalize for cosine similarity

  # Pooling strategy
  pooling_strategy: "mean"  # "mean", "cls", "max", "weighted_mean"

  # Quantization (for efficiency)
  quantize: false
  quantization_bits: 8  # 4, 8, or 16

  # Batching
  encode_batch_size: 32

  # Device
  device: "cuda"  # "cuda", "cpu", "mps"
  half_precision: true  # Use float16
```

### 1.5 Dimensionality Reduction Options

| Method | Output Dim | Speed | Quality Loss | Use Case |
|--------|------------|-------|--------------|----------|
| PCA | Any | Fast | Low | General reduction |
| UMAP | Any | Medium | Very Low | Visualization + search |
| Random Projection | Any | Fastest | Medium | Speed-critical |
| Autoencoder | Any | Slow (train) | Low | Custom reduction |
| Matryoshka (truncation) | 64-768 | None | Varies | Flexible models |

```yaml
dimensionality_reduction:
  enabled: false
  method: "pca"  # "pca", "umap", "random_projection", "autoencoder"
  target_dim: 256

  # PCA specific
  pca_components: 256
  pca_whiten: false

  # UMAP specific
  umap_n_neighbors: 15
  umap_min_dist: 0.1
  umap_metric: "cosine"
```

---

## 2. Language Model Architecture

### 2.1 Model Size Options

| Model | Parameters | Layers | Hidden Dim | Heads | Context | VRAM (fp16) |
|-------|------------|--------|------------|-------|---------|-------------|
| CodeLlama-7B | 7B | 32 | 4096 | 32 | 16K | ~14 GB |
| CodeLlama-13B | 13B | 40 | 5120 | 40 | 16K | ~26 GB |
| CodeLlama-34B | 34B | 48 | 8192 | 64 | 16K | ~68 GB |
| CodeLlama-70B | 70B | 80 | 8192 | 64 | 16K | ~140 GB |
| DeepSeek-Coder-1.3B | 1.3B | 24 | 2048 | 16 | 16K | ~3 GB |
| DeepSeek-Coder-6.7B | 6.7B | 32 | 4096 | 32 | 16K | ~14 GB |
| DeepSeek-Coder-33B | 33B | 62 | 7168 | 56 | 16K | ~66 GB |
| StarCoder2-3B | 3B | 30 | 2560 | 20 | 16K | ~6 GB |
| StarCoder2-7B | 7B | 32 | 4096 | 32 | 16K | ~14 GB |
| StarCoder2-15B | 15B | 40 | 6144 | 48 | 16K | ~30 GB |
| Qwen2.5-Coder-7B | 7B | 28 | 3584 | 28 | 128K | ~14 GB |
| Qwen2.5-Coder-14B | 14B | 48 | 5120 | 40 | 128K | ~28 GB |
| Qwen2.5-Coder-32B | 32B | 64 | 5120 | 40 | 128K | ~64 GB |

### 2.2 Precision Options

| Precision | Bits | Memory | Speed | Quality | Use Case |
|-----------|------|--------|-------|---------|----------|
| float32 | 32 | 1x | Baseline | Best | Training |
| float16 | 16 | 0.5x | ~1.5x | Near-best | Default inference |
| bfloat16 | 16 | 0.5x | ~1.5x | Near-best | Training/inference |
| int8 | 8 | 0.25x | ~2x | Good | Memory-constrained |
| int4 | 4 | 0.125x | ~2x | Moderate | Edge deployment |
| GPTQ-4bit | 4 | 0.125x | ~1.5x | Good | Quantized models |
| AWQ-4bit | 4 | 0.125x | ~1.8x | Better | Optimized quant |
| GGUF-Q4_K_M | ~4.5 | 0.14x | ~1.5x | Good | llama.cpp |

### 2.3 Model Configuration

```yaml
model_config:
  # Model selection
  model_name: "codellama/CodeLlama-13b-Instruct-hf"
  revision: "main"  # or specific commit hash

  # Architecture
  architecture: "llama"  # "llama", "mistral", "qwen", "starcoder"

  # Precision
  torch_dtype: "float16"  # "float32", "float16", "bfloat16"
  load_in_8bit: false
  load_in_4bit: false

  # Quantization config (if using 4-bit)
  quantization:
    enabled: false
    method: "bitsandbytes"  # "bitsandbytes", "gptq", "awq"
    bits: 4
    group_size: 128
    double_quant: true
    quant_type: "nf4"  # "nf4", "fp4"

  # Memory optimization
  device_map: "auto"  # "auto", "balanced", "sequential", or explicit
  max_memory:
    0: "22GB"  # GPU 0
    1: "22GB"  # GPU 1
    cpu: "64GB"

  # Attention
  attn_implementation: "flash_attention_2"  # "eager", "sdpa", "flash_attention_2"
  use_cache: true

  # Context length
  max_position_embeddings: 16384  # or 32768, 65536, 131072
  rope_scaling:
    type: "dynamic"  # "linear", "dynamic"
    factor: 2.0  # Extend context by this factor
```

### 2.4 Vocabulary Size Impact

| Vocab Size | Embedding Memory | Output Layer Memory | Models |
|------------|------------------|---------------------|--------|
| 32,000 | ~62 MB (fp16) | ~62 MB (fp16) | Llama 1/2 |
| 32,256 | ~63 MB | ~63 MB | CodeLlama |
| 49,152 | ~96 MB | ~96 MB | StarCoder |
| 100,000 | ~195 MB | ~195 MB | Mistral |
| 128,256 | ~250 MB | ~250 MB | Llama 3 |
| 151,936 | ~297 MB | ~297 MB | Qwen2 |

---

## 3. Vector Index Architecture

### 3.1 FAISS Index Types

| Index Type | Build Time | Search Time | Memory | Accuracy | Use Case |
|------------|------------|-------------|--------|----------|----------|
| `Flat` | O(1) | O(n) | 1x | 100% | Small datasets (<100K) |
| `IVF` | O(n) | O(n/k) | 1x | ~95-99% | Medium (100K-1M) |
| `HNSW` | O(n log n) | O(log n) | 1.3x | ~95-99% | Fast search |
| `IVF-PQ` | O(n) | O(n/k) | 0.1-0.3x | ~90-95% | Large + memory constrained |
| `IVF-HNSW` | O(n log n) | O(log n) | 1.3x | ~95-99% | Large + fast search |
| `ScaNN` | O(n) | O(1) | 0.5x | ~95% | Production scale |

### 3.2 Index Configuration

```yaml
faiss_config:
  # Index type
  index_type: "IVF"  # "Flat", "IVF", "HNSW", "IVF-PQ", "IVF-HNSW"

  # Flat index (exact search)
  flat:
    metric: "IP"  # "IP" (inner product) or "L2"

  # IVF (Inverted File)
  ivf:
    nlist: 100  # Number of clusters (sqrt(n) to 4*sqrt(n))
    nprobe: 10  # Clusters to search (higher = slower but more accurate)
    metric: "IP"

  # HNSW (Hierarchical Navigable Small World)
  hnsw:
    M: 32  # Number of connections per layer (16-64)
    ef_construction: 200  # Build-time search width (100-500)
    ef_search: 50  # Search-time search width (10-100)

  # Product Quantization
  pq:
    m: 8  # Number of subquantizers (embedding_dim must be divisible by m)
    nbits: 8  # Bits per subquantizer (typically 8)

  # GPU acceleration
  use_gpu: true
  gpu_id: 0

  # Memory mapping (for large indices)
  use_mmap: false

  # Training
  train_size: 100000  # Vectors to use for training IVF/PQ
```

### 3.3 Index Sizing

| Vectors | Flat (384d) | IVF (384d) | HNSW (384d) | IVF-PQ (384d, m=8) |
|---------|-------------|------------|-------------|---------------------|
| 10K | 15 MB | 15 MB | 20 MB | 1.5 MB |
| 100K | 150 MB | 150 MB | 200 MB | 15 MB |
| 1M | 1.5 GB | 1.5 GB | 2 GB | 150 MB |
| 10M | 15 GB | 15 GB | 20 GB | 1.5 GB |
| 100M | 150 GB | 150 GB | 200 GB | 15 GB |

---

## 4. Token Classification Architecture

### 4.1 Vocabulary-Based Classification

| Approach | Tokens Classified | Memory | Speed | Accuracy |
|----------|-------------------|--------|-------|----------|
| Word List (Small) | ~500 | <1 MB | Fastest | Lower coverage |
| Word List (Medium) | ~2,000 | ~2 MB | Fast | Good coverage |
| Word List (Large) | ~10,000 | ~10 MB | Fast | Best coverage |
| Regex Patterns | Dynamic | ~1 MB | Medium | Pattern-dependent |
| Embedding Similarity | Full vocab | ~100 MB | Slow | Semantic-based |
| Trained Classifier | Full vocab | ~50 MB | Medium | Learned |

### 4.2 Classification Configuration

```yaml
token_classification:
  # Method
  method: "word_list"  # "word_list", "regex", "embedding", "classifier"

  # Word list approach
  word_list:
    code_words_file: "data/code_words.txt"
    language_words_file: "data/language_words.txt"
    include_variations: true  # lowercase, uppercase, capitalized
    include_space_prefix: true  # " word" for GPT tokenizers

  # Regex approach
  regex:
    code_patterns:
      - "[A-Z][a-zA-Z]+[A-Z]"  # camelCase
      - "[a-z]+_[a-z]+"  # snake_case
      - "[A-Z][a-zA-Z]+"  # PascalCase
    language_patterns:
      - "^(very|quite|really|extremely)$"
      - "^(however|therefore|moreover)$"

  # Embedding similarity approach
  embedding:
    model: "all-MiniLM-L6-v2"
    code_anchors: ["function", "class", "variable", "database", "API"]
    language_anchors: ["however", "therefore", "elegant", "simple"]
    similarity_threshold: 0.6

  # Trained classifier
  classifier:
    model_path: "models/token_classifier.pt"
    architecture: "mlp"  # "mlp", "lstm", "transformer"
    hidden_dim: 256
    num_layers: 2

  # Caching
  cache_classifications: true
  cache_size: 100000  # Token IDs to cache
```

### 4.3 Token Set Sizes

| Configuration | Code Tokens | Language Tokens | Total | Memory |
|---------------|-------------|-----------------|-------|--------|
| Minimal | ~200 | ~100 | ~300 | <1 MB |
| Standard | ~1,000 | ~500 | ~1,500 | ~2 MB |
| Extended | ~5,000 | ~2,000 | ~7,000 | ~10 MB |
| Comprehensive | ~20,000 | ~10,000 | ~30,000 | ~40 MB |

---

## 5. Retrieval System Architecture

### 5.1 Retrieval Pipeline Stages

```
Query → [Embedding] → [Index Search] → [Reranking] → [Filtering] → Results
         ↓              ↓                ↓             ↓
       384-dim        Top-100          Top-20        Top-5
```

### 5.2 Stage Configuration

```yaml
retrieval_pipeline:
  # Stage 1: Query Embedding
  query_embedding:
    model: "all-MiniLM-L6-v2"
    max_query_length: 256

  # Stage 2: Initial Retrieval (vector search)
  initial_retrieval:
    index_type: "HNSW"
    top_k: 100  # Retrieve more for reranking

  # Stage 3: Reranking (optional, more expensive)
  reranking:
    enabled: true
    model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: 20  # Rerank top 20
    batch_size: 16

  # Stage 4: Filtering
  filtering:
    enabled: true
    min_score: 0.5
    max_results: 5
    deduplicate: true
    dedup_threshold: 0.9

  # Hybrid search (combine with BM25)
  hybrid:
    enabled: true
    bm25_weight: 0.3
    vector_weight: 0.7
    fusion_method: "rrf"  # "rrf" (reciprocal rank fusion) or "linear"
```

### 5.3 Reranker Models

| Model | Parameters | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | Fast | Good | Default |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 33M | Medium | Better | Production |
| `BAAI/bge-reranker-base` | 110M | Slow | Best | High quality |
| `BAAI/bge-reranker-large` | 335M | Slowest | Highest | Maximum quality |
| `Cohere rerank-english-v2.0` | API | Medium | Excellent | API-based |

### 5.4 Chunking Strategy

| Strategy | Chunk Size | Overlap | Chunks/File | Use Case |
|----------|------------|---------|-------------|----------|
| Fixed | 512 tokens | 50 | Many | Simple baseline |
| Sentence | Varies | 0 | Variable | Natural boundaries |
| Paragraph | Varies | 0 | Fewer | Document structure |
| Function | Varies | 0 | Per function | Code-aware |
| AST-based | Varies | 0 | Per node | Semantic code |
| Sliding Window | 512 | 256 | Many | Maximum recall |

```yaml
chunking:
  strategy: "function"  # "fixed", "sentence", "paragraph", "function", "ast"

  # Fixed chunking
  fixed:
    chunk_size: 512  # tokens
    overlap: 50  # tokens

  # Sliding window
  sliding:
    window_size: 512
    stride: 256

  # Code-aware chunking
  code_aware:
    chunk_by: "function"  # "function", "class", "file"
    max_chunk_size: 1024
    min_chunk_size: 64
    include_docstrings: true
    include_imports: true
```

---

## 6. Context Window Architecture

### 6.1 Context Window Sizes

| Model | Default Context | Extended | Maximum | Tokens/$ (input) |
|-------|-----------------|----------|---------|------------------|
| GPT-4 | 8K | 32K | 128K | ~$0.03/1K |
| GPT-4o | 128K | - | 128K | ~$0.005/1K |
| Claude 3 Sonnet | 200K | - | 200K | ~$0.003/1K |
| Claude 3 Opus | 200K | - | 200K | ~$0.015/1K |
| Gemini 1.5 Pro | 128K | 1M | 2M | ~$0.00125/1K |
| CodeLlama | 16K | 100K+ | 100K+ | Local |
| Qwen2.5-Coder | 128K | - | 128K | Local |

### 6.2 Context Allocation Strategy

```yaml
context_allocation:
  total_budget: 8192  # Total available tokens

  # Reserved allocations
  reserved:
    system_prompt: 256
    user_query: 512
    response: 1024
    formatting: 128

  # Available for retrieved context
  available_for_context: 6272  # total - reserved

  # Allocation ratios
  allocation:
    initial_context: 0.4   # 40% for initial retrieval
    adaptive_context: 0.4  # 40% for adaptive retrieval
    buffer: 0.2            # 20% buffer for overflow

  # Per-retrieval limits
  per_retrieval:
    max_files: 3
    max_tokens: 2000

  # Priority-based allocation
  priority_weights:
    seed_files: 1.0
    direct_imports: 0.8
    adaptive_retrievals: 0.7
    secondary_imports: 0.5
```

### 6.3 Context Compression Options

| Method | Compression Ratio | Quality Loss | Speed | Use Case |
|--------|-------------------|--------------|-------|----------|
| None | 1x | 0% | Fastest | Small context |
| Truncation | 2-10x | High | Fast | Simple reduction |
| Summarization | 3-10x | Medium | Slow | Preserve meaning |
| TOON Format | 2-5x | Low | Fast | Structured data |
| Selective Inclusion | 2-5x | Low | Medium | Important parts only |
| LLMLingua | 2-10x | Low | Medium | Neural compression |

```yaml
context_compression:
  enabled: true
  method: "toon"  # "none", "truncation", "summarization", "toon", "selective", "llmlingua"

  # Truncation
  truncation:
    strategy: "end"  # "end", "middle", "start"
    keep_ratio: 0.5

  # Summarization
  summarization:
    model: "gpt-3.5-turbo"
    max_summary_length: 256

  # TOON format
  toon:
    include_symbols: true
    include_imports: true
    include_docstrings: false
    max_rows_per_table: 100

  # LLMLingua
  llmlingua:
    model: "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    compression_rate: 0.5
    force_tokens: ["def", "class", "import", "return"]
```

---

## 7. Entropy Computation Architecture

### 7.1 Computation Precision

| Precision | Memory | Speed | Numerical Stability | Use Case |
|-----------|--------|-------|---------------------|----------|
| float64 | 8 bytes | Slow | Best | High precision |
| float32 | 4 bytes | Medium | Good | Default |
| float16 | 2 bytes | Fast | Lower | GPU-optimized |
| bfloat16 | 2 bytes | Fast | Good | Training |

### 7.2 Entropy Computation Options

```yaml
entropy_computation:
  # Precision
  dtype: "float32"  # "float64", "float32", "float16"

  # Numerical stability
  epsilon: 1.0e-10  # Added to prevent log(0)
  clip_probs: true  # Clip probabilities to [epsilon, 1-epsilon]

  # Computation method
  method: "standard"  # "standard", "streaming", "approximate"

  # Streaming (for very long sequences)
  streaming:
    enabled: false
    window_size: 100

  # Approximate (for speed)
  approximate:
    enabled: false
    sample_size: 1000  # Sample from vocab for approximation

  # GPU acceleration
  use_gpu: true

  # Batch computation
  batch_entropy: true
  batch_size: 32
```

### 7.3 Softmax Computation

| Method | Speed | Stability | Memory | Use Case |
|--------|-------|-----------|--------|----------|
| Naive | Fast | Poor | Low | Never use |
| Log-sum-exp | Medium | Good | Low | CPU |
| Flash Softmax | Fastest | Good | Lowest | GPU |
| Chunked | Medium | Good | Configurable | Long sequences |

---

## 8. Caching Architecture

### 8.1 Cache Levels

```
┌─────────────────────────────────────────────────────────────┐
│                      CACHE HIERARCHY                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  L1: In-Memory (Python dict)                                │
│      - Token classifications                                 │
│      - Recent entropy values                                 │
│      - Hot embeddings                                        │
│      Size: 100 MB - 1 GB                                    │
│                                                              │
│  L2: Local Cache (Redis/SQLite)                             │
│      - File embeddings                                       │
│      - Symbol registry                                       │
│      - AST cache                                            │
│      Size: 1 GB - 10 GB                                     │
│                                                              │
│  L3: Persistent Storage (Disk)                              │
│      - FAISS indices                                         │
│      - Full embeddings                                       │
│      - Analysis results                                      │
│      Size: 10 GB - 100 GB                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Cache Configuration

```yaml
caching:
  # L1: In-memory cache
  memory_cache:
    enabled: true
    max_size_mb: 512
    eviction_policy: "lru"  # "lru", "lfu", "fifo"
    ttl_seconds: 3600

    # What to cache
    cache_token_classifications: true
    cache_entropy_values: true
    cache_top_tokens: true

  # L2: Local cache (Redis)
  redis_cache:
    enabled: true
    host: "localhost"
    port: 6379
    db: 0
    max_memory: "2gb"
    eviction_policy: "allkeys-lru"

  # L2: Alternative (SQLite)
  sqlite_cache:
    enabled: false
    path: "cache/embeddings.db"
    max_size_mb: 5000

  # L3: Disk cache
  disk_cache:
    enabled: true
    path: "cache/"
    max_size_gb: 50
    compression: "lz4"  # "none", "gzip", "lz4", "zstd"

  # KV cache for LLM (if managing manually)
  kv_cache:
    enabled: true
    max_length: 16384
    dtype: "float16"
```

### 8.3 Cache Sizes by Component

| Component | Entries | Size/Entry | Total Size | TTL |
|-----------|---------|------------|------------|-----|
| Token Classifications | 100K | 8 bytes | 800 KB | Forever |
| Entropy History | 10K | 4 bytes | 40 KB | 1 hour |
| File Embeddings | 10K | 1.5 KB | 15 MB | 24 hours |
| AST Cache | 1K | 10 KB | 10 MB | On change |
| Symbol Registry | 100K | 200 bytes | 20 MB | On change |
| FAISS Index | 1M | 1.5 KB | 1.5 GB | On rebuild |

---

## 9. Batch Processing & Parallelism

### 9.1 Batch Sizes

| Component | Min Batch | Default Batch | Max Batch | Memory Impact |
|-----------|-----------|---------------|-----------|---------------|
| Embedding Generation | 1 | 32 | 256 | Linear |
| FAISS Search | 1 | 16 | 1024 | Linear |
| Entropy Computation | 1 | 64 | 512 | Linear |
| Token Classification | 1 | 1000 | 10000 | Linear |
| Reranking | 1 | 16 | 64 | Linear |
| AST Parsing | 1 | 10 | 100 | Variable |

### 9.2 Parallelism Configuration

```yaml
parallelism:
  # Process-level
  num_workers: 4  # For data loading, AST parsing

  # Thread-level
  num_threads: 8  # For CPU operations

  # GPU parallelism
  tensor_parallel_size: 1  # For model parallelism
  pipeline_parallel_size: 1  # For pipeline parallelism

  # Async operations
  async_embedding: true
  async_retrieval: true
  max_concurrent_requests: 10

  # Batch processing
  batch_sizes:
    embedding: 32
    search: 16
    reranking: 16
    classification: 1000

  # Prefetching
  prefetch:
    enabled: true
    prefetch_factor: 2  # Batches to prefetch
```

### 9.3 Multi-GPU Configuration

```yaml
multi_gpu:
  enabled: true
  strategy: "data_parallel"  # "data_parallel", "tensor_parallel", "pipeline_parallel"

  # Data parallel
  data_parallel:
    gpu_ids: [0, 1]

  # Tensor parallel (split model across GPUs)
  tensor_parallel:
    size: 2

  # Pipeline parallel (split layers across GPUs)
  pipeline_parallel:
    size: 2
    chunks: 4

  # Memory allocation
  memory_per_gpu: "22GB"
  offload_to_cpu: false
```

---

## 10. Memory Architecture

### 10.1 Memory Budget Breakdown

```yaml
memory_budget:
  total_gpu_memory: "24GB"  # Available GPU memory

  # Allocations
  model_weights: "12GB"     # Base model
  kv_cache: "4GB"           # KV cache for context
  activations: "2GB"        # Forward pass activations
  optimizer: "0GB"          # Not training
  embeddings: "1GB"         # Embedding model
  faiss_index: "2GB"        # Vector index (GPU)
  buffer: "3GB"             # Safety buffer

  # CPU memory
  total_cpu_memory: "64GB"
  cpu_offload: "16GB"       # Model offload if needed
  cache: "8GB"              # Various caches
  data: "4GB"               # Loaded data
```

### 10.2 Memory Optimization Techniques

| Technique | Memory Savings | Speed Impact | Implementation |
|-----------|----------------|--------------|----------------|
| Gradient Checkpointing | 60-70% activations | -20% speed | `torch.checkpoint` |
| Mixed Precision | 50% weights | +20% speed | `torch.cuda.amp` |
| 8-bit Quantization | 50% weights | ~0% speed | `bitsandbytes` |
| 4-bit Quantization | 75% weights | -10% speed | `bitsandbytes` |
| Flash Attention | 50% attention | +20% speed | `flash_attn` |
| Paged Attention | Variable | +10% speed | `vllm` |
| CPU Offload | All weights | -50% speed | `accelerate` |

### 10.3 Memory Configuration

```yaml
memory_optimization:
  # Quantization
  quantization:
    enabled: true
    bits: 8  # 4, 8, or 16

  # Attention optimization
  attention:
    use_flash_attention: true
    use_memory_efficient_attention: true

  # Offloading
  offload:
    enabled: false
    offload_folder: "offload/"
    offload_state_dict: true

  # Garbage collection
  gc:
    enabled: true
    gc_interval: 100  # Steps between GC
    empty_cuda_cache: true
```

---

## 11. Storage Architecture

### 11.1 Storage Requirements by Component

| Component | Size per 1K Files | Size per 10K Files | Size per 100K Files |
|-----------|-------------------|--------------------|--------------------|
| Source Code | ~50 MB | ~500 MB | ~5 GB |
| AST Cache | ~100 MB | ~1 GB | ~10 GB |
| Embeddings (384d) | ~1.5 MB | ~15 MB | ~150 MB |
| FAISS Index | ~2 MB | ~20 MB | ~200 MB |
| Symbol Registry | ~10 MB | ~100 MB | ~1 GB |
| Import Graph | ~1 MB | ~10 MB | ~100 MB |
| Entropy Logs | ~5 MB | ~50 MB | ~500 MB |

### 11.2 Storage Configuration

```yaml
storage:
  # Base paths
  data_dir: "data/"
  cache_dir: "cache/"
  index_dir: "indices/"
  logs_dir: "logs/"

  # File formats
  embedding_format: "npy"  # "npy", "hdf5", "parquet"
  index_format: "faiss"    # "faiss", "annoy", "hnswlib"
  metadata_format: "json"  # "json", "sqlite", "parquet"

  # Compression
  compression:
    embeddings: "lz4"  # "none", "gzip", "lz4", "zstd"
    indices: "none"    # Usually not compressed
    logs: "gzip"

  # Sharding (for large datasets)
  sharding:
    enabled: false
    shard_size: 100000  # Entries per shard

  # Backup
  backup:
    enabled: true
    interval: "daily"
    retention_days: 7
```

---

## 12. Network Architecture

### 12.1 API Configuration (if using external models)

```yaml
api_config:
  # OpenAI
  openai:
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    timeout: 60
    max_retries: 3

  # Anthropic
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    timeout: 60

  # Rate limiting
  rate_limiting:
    requests_per_minute: 60
    tokens_per_minute: 100000

  # Connection pooling
  connection_pool:
    max_connections: 10
    keepalive_timeout: 30
```

### 12.2 Distributed Architecture

```yaml
distributed:
  # Mode
  mode: "single"  # "single", "distributed"

  # Distributed settings
  master_addr: "localhost"
  master_port: 29500
  world_size: 2
  rank: 0

  # Backend
  backend: "nccl"  # "nccl" (GPU), "gloo" (CPU), "mpi"

  # Sharding
  model_sharding: "full"  # "full", "zero2", "zero3"
```

---

## 13. Scaling Dimensions Summary

### 13.1 Horizontal Scaling (More of the same)

| Dimension | Scale By | Effect | Cost |
|-----------|----------|--------|------|
| More GPUs | Add GPUs | Faster inference, larger models | Linear |
| More Nodes | Add servers | Higher throughput | Linear |
| More Shards | Split index | Larger index capacity | Sub-linear |
| More Workers | Add processes | Faster preprocessing | Linear |

### 13.2 Vertical Scaling (Bigger components)

| Dimension | Scale By | Effect | Cost |
|-----------|----------|--------|------|
| Larger Model | More params | Better quality | Quadratic (memory) |
| Larger Embeddings | More dims | Better retrieval | Linear |
| Larger Context | More tokens | More context | Linear (with flash attn) |
| Larger Index | More vectors | More documents | Linear |
| Larger Cache | More memory | Faster repeated queries | Linear |

### 13.3 Quality Scaling

| Dimension | Scale By | Effect | Cost |
|-----------|----------|--------|------|
| Better Embeddings | Better model | Better retrieval | Model dependent |
| Reranking | Add reranker | Better precision | 10-100x per query |
| Ensemble | Multiple models | More robust | Linear with models |
| Larger Token Sets | More tokens | Better classification | Minimal |

---

## 14. Resource Requirements Matrix

### 14.1 Configuration Profiles

| Profile | GPU | VRAM | CPU | RAM | Storage | Use Case |
|---------|-----|------|-----|-----|---------|----------|
| **Minimal** | Optional | 0-4 GB | 4 cores | 8 GB | 10 GB | Development |
| **Standard** | 1x RTX 3080 | 10 GB | 8 cores | 32 GB | 50 GB | Single repo |
| **Production** | 1x A100 | 40 GB | 16 cores | 64 GB | 200 GB | Multiple repos |
| **Enterprise** | 4x A100 | 160 GB | 64 cores | 256 GB | 1 TB | Organization-wide |

### 14.2 Component Resource Usage

| Component | CPU | GPU | RAM | VRAM | Disk |
|-----------|-----|-----|-----|------|------|
| Embedding (MiniLM) | Low | Medium | 1 GB | 0.5 GB | 100 MB |
| LLM (7B, fp16) | Low | High | 2 GB | 14 GB | 14 GB |
| LLM (13B, fp16) | Low | High | 4 GB | 26 GB | 26 GB |
| LLM (7B, 4-bit) | Low | Medium | 2 GB | 4 GB | 4 GB |
| FAISS (1M vectors) | Medium | Optional | 2 GB | 2 GB | 2 GB |
| AST Parsing | High | None | 4 GB | 0 | 1 GB |
| Entropy Computation | Low | Low | 0.5 GB | 0.1 GB | 0 |

---

## 15. Scaling Recommendations

### 15.1 Development (Start Here)

```yaml
development:
  model: "codellama/CodeLlama-7b-Instruct-hf"
  precision: "4bit"
  embedding_model: "all-MiniLM-L6-v2"
  embedding_dim: 384
  index_type: "Flat"
  context_window: 4096
  batch_size: 8
  cache_size_mb: 256
```

### 15.2 Production (Single Repo)

```yaml
production_single:
  model: "codellama/CodeLlama-13b-Instruct-hf"
  precision: "float16"
  embedding_model: "all-mpnet-base-v2"
  embedding_dim: 768
  index_type: "HNSW"
  reranking: true
  context_window: 16384
  batch_size: 32
  cache_size_mb: 2048
```

### 15.3 Production (Multiple Repos)

```yaml
production_multi:
  model: "codellama/CodeLlama-34b-Instruct-hf"
  precision: "float16"
  tensor_parallel: 2
  embedding_model: "bigcode/starencoder"
  embedding_dim: 1024
  index_type: "IVF-HNSW"
  reranking: true
  context_window: 32768
  batch_size: 64
  cache_size_mb: 8192
  redis_cache: true
```

### 15.4 Research (Maximum Quality)

```yaml
research:
  model: "Qwen/Qwen2.5-Coder-32B-Instruct"
  precision: "float16"
  tensor_parallel: 4
  embedding_model: "BAAI/bge-large-en-v1.5"
  embedding_dim: 1024
  index_type: "Flat"  # Exact search for research
  reranking: true
  reranker_model: "BAAI/bge-reranker-large"
  context_window: 65536
  batch_size: 16
  full_logging: true
```

---

## Appendix A: Scaling Formulas

### Memory Estimation

```
Model Memory (fp16) ≈ 2 * num_parameters bytes
KV Cache Memory ≈ 2 * num_layers * 2 * hidden_dim * context_length * batch_size * 2 bytes
Embedding Memory ≈ num_vectors * embedding_dim * 4 bytes
Index Memory (HNSW) ≈ 1.3 * num_vectors * embedding_dim * 4 bytes
```

### Throughput Estimation

```
Tokens/Second ≈ batch_size * model_speed_factor / (context_length / 1024)
Queries/Second ≈ 1 / (embedding_time + search_time + rerank_time + generation_time)
```

### Cost Estimation

```
GPU Hours ≈ num_examples * avg_generation_time / 3600
API Cost ≈ (input_tokens + output_tokens) * cost_per_token
Storage Cost ≈ total_embeddings * embedding_dim * 4 / 1e9 * cost_per_gb
```

---

*Document created: December 2024*
*Last updated: December 2024*
