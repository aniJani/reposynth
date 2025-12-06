# ContextLoom: Complete Parameter Reference

This document catalogs every configurable parameter, hyperparameter, and design choice in the research implementation. Use this for ablation studies, sensitivity analysis, and experiment configuration.

---

## Table of Contents

1. [Entropy Calculation Parameters](#1-entropy-calculation-parameters)
2. [Token Classification Parameters](#2-token-classification-parameters)
3. [Contrastive Code Entropy (CCE) Parameters](#3-contrastive-code-entropy-cce-parameters)
4. [Uncertainty Monitoring Parameters](#4-uncertainty-monitoring-parameters)
5. [Spike Detection Parameters](#5-spike-detection-parameters)
6. [Measurement Strategy Parameters](#6-measurement-strategy-parameters)
7. [Adaptive Retrieval Parameters](#7-adaptive-retrieval-parameters)
8. [Context Management Parameters](#8-context-management-parameters)
9. [Topic Inference Parameters](#9-topic-inference-parameters)
10. [Generation Parameters](#10-generation-parameters)
11. [Model Selection Parameters](#11-model-selection-parameters)
12. [Evaluation Parameters](#12-evaluation-parameters)
13. [Baseline Configuration Parameters](#13-baseline-configuration-parameters)
14. [Visualization Parameters](#14-visualization-parameters)
15. [Experimental Design Choices](#15-experimental-design-choices)

---

## 1. Entropy Calculation Parameters

### 1.1 Basic Entropy

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `temperature` | float | 1.0 | [0.1, 2.0] | Softmax temperature for probability calculation. Lower = sharper distribution, higher = flatter. |
| `epsilon` | float | 1e-10 | [1e-12, 1e-6] | Small constant added to prevent log(0). |
| `log_base` | str | "natural" | ["natural", "2", "10"] | Logarithm base for entropy calculation. Natural log gives nats, log2 gives bits. |

### 1.2 Normalized Entropy

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `normalization_method` | str | "max_entropy" | ["max_entropy", "vocab_percentile", "empirical"] | How to normalize entropy to [0,1]. |
| `vocab_percentile` | float | 0.99 | [0.9, 1.0] | If using percentile normalization, what percentile of vocab to consider. |
| `empirical_max` | float | None | [0, 15] | If using empirical normalization, the observed maximum entropy. |

### 1.3 Probability Differential (UnCert-CoT Style)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `top_k_for_diff` | int | 2 | [2, 10] | Number of top probabilities to consider. Default is top-2 difference. |
| `diff_aggregation` | str | "gap" | ["gap", "ratio", "kl"] | How to compute the differential. "gap" = p1-p2, "ratio" = p1/p2, "kl" = KL divergence. |

---

## 2. Token Classification Parameters

### 2.1 Code Token Definition

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `code_word_list` | List[str] | (see below) | Words considered "code-related". |
| `include_camel_case` | bool | True | Include camelCase patterns as code tokens. |
| `include_snake_case` | bool | True | Include snake_case patterns as code tokens. |
| `include_pascal_case` | bool | True | Include PascalCase patterns as code tokens. |
| `include_all_caps` | bool | True | Include ALL_CAPS patterns (constants) as code tokens. |
| `include_dotted` | bool | True | Include .method patterns as code tokens. |
| `min_identifier_length` | int | 2 | Minimum length for identifier patterns. |

**Default Code Word Categories:**

```yaml
databases:
  - redis, Redis, REDIS
  - postgresql, PostgreSQL, postgres
  - mysql, MySQL
  - mongodb, MongoDB, mongo
  - sqlite, SQLite
  - database, db, DB

authentication:
  - jwt, JWT
  - oauth, OAuth, OAuth2
  - token, Token
  - session, Session
  - cookie, Cookie
  - auth, Auth, authentication
  - password, hash, bcrypt

programming_constructs:
  - function, Function, func
  - class, Class
  - method, Method
  - variable, var, let, const
  - async, await, Async, Await
  - promise, Promise
  - callback, Callback
  - import, export, require

web_api:
  - api, API
  - endpoint, Endpoint
  - route, Route
  - http, HTTP, https, HTTPS
  - request, response, Request, Response
  - middleware, Middleware
  - controller, Controller
  - rest, REST, graphql, GraphQL

data_types:
  - array, Array
  - object, Object
  - string, String
  - number, Number, int, integer, float
  - boolean, Boolean, bool
  - null, None, undefined

error_handling:
  - error, Error
  - exception, Exception
  - throw, catch, try
  - finally

configuration:
  - config, Config, configuration
  - env, ENV, environment
  - secret, Secret
  - key, Key
```

### 2.2 Language Token Definition

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language_word_list` | List[str] | (see below) | Words considered "language/style-related". |
| `include_common_articles` | bool | True | Include "the", "a", "an". |
| `include_prepositions` | bool | True | Include "in", "on", "at", etc. |
| `include_conjunctions` | bool | True | Include "and", "or", "but", etc. |

**Default Language Word Categories:**

```yaml
adjectives_quality:
  - simple, complex, sophisticated, elegant
  - robust, efficient, effective, powerful
  - basic, advanced, modern, traditional
  - standard, custom, flexible, scalable

adjectives_size:
  - large, small, big, tiny
  - huge, massive, minimal

adverbs_degree:
  - very, quite, really, extremely
  - highly, fairly, somewhat, rather

adverbs_manner:
  - basically, essentially, fundamentally
  - primarily, mainly, mostly
  - generally, typically, usually
  - often, sometimes, rarely

transitions:
  - however, therefore, moreover
  - furthermore, additionally, consequently
  - although, nevertheless, thus
  - hence, accordingly, meanwhile

hedging:
  - probably, possibly, perhaps
  - likely, unlikely, certainly
  - maybe, might, could

filler:
  - actually, literally, definitely
  - obviously, clearly, apparently

articles_prepositions:
  - the, a, an
  - in, on, at, by, for, with, to
  - of, from, into, through

conjunctions:
  - and, or, but, nor
  - yet, so, because, although
```

### 2.3 Token Set Construction

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `include_variations` | bool | True | - | Include lowercase, uppercase, capitalized variations. |
| `include_space_prefix` | bool | True | - | Include " word" (space-prefixed) variations for GPT-style tokenizers. |
| `include_subwords` | bool | False | - | Include subword tokens (e.g., "Redis" might tokenize to "Red" + "is"). |
| `max_tokens_per_word` | int | 3 | [1, 5] | Maximum subword tokens to include per word. |
| `cache_token_sets` | bool | True | - | Cache computed token sets for efficiency. |

---

## 3. Contrastive Code Entropy (CCE) Parameters

### 3.1 Core CCE Calculation

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `cce_threshold` | float | 0.3 | [0.1, 0.7] | Threshold for triggering context retrieval. CCE > threshold → retrieve. |
| `code_weight` | float | 1.0 | [0.5, 2.0] | Weight multiplier for code entropy component. |
| `language_weight` | float | 1.0 | [0.5, 2.0] | Weight multiplier for language entropy component. |
| `normalization` | str | "category" | ["category", "global", "none"] | How to normalize category entropies. |

### 3.2 Uncertainty Type Classification

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `code_uncertainty_threshold` | float | 0.2 | [0.1, 0.4] | CCE > this → classified as "code" uncertainty. |
| `language_uncertainty_threshold` | float | -0.2 | [-0.4, -0.1] | CCE < this → classified as "language" uncertainty. |
| `mixed_zone` | tuple | (-0.2, 0.2) | - | Range where uncertainty is "mixed". |

### 3.3 Confidence Calculation

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `confidence_scale` | float | 0.5 | [0.3, 1.0] | Scaling factor for confidence. confidence = min(|CCE| / scale, 1.0). |
| `min_confidence_for_action` | float | 0.3 | [0.1, 0.5] | Minimum confidence required to trust the CCE decision. |

---

## 4. Uncertainty Monitoring Parameters

### 4.1 Method Selection

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `uncertainty_method` | str | "cce" | ["raw_entropy", "normalized_entropy", "prob_differential", "cce", "ensemble"] | Which uncertainty method to use. |
| `ensemble_weights` | dict | {"cce": 0.5, "prob_diff": 0.3, "entropy": 0.2} | - | Weights if using ensemble method. |

### 4.2 Monitoring Frequency

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `monitor_every_n_tokens` | int | 1 | [1, 20] | Compute uncertainty every N tokens. 1 = every token. |
| `batch_monitoring` | bool | False | - | Compute uncertainty in batches for efficiency. |
| `batch_size` | int | 10 | [5, 50] | If batching, how many tokens per batch. |

### 4.3 History Tracking

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `history_window` | int | 50 | [10, 200] | Number of past entropy values to keep for analysis. |
| `track_top_tokens` | bool | True | - | Store top-k tokens at each step for debugging. |
| `top_k_to_track` | int | 5 | [3, 20] | How many top tokens to track. |

---

## 5. Spike Detection Parameters

### 5.1 Detection Method

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `spike_method` | str | "relative" | ["fixed", "relative", "statistical", "adaptive"] | How to detect entropy spikes. |

### 5.2 Fixed Threshold Method

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `fixed_threshold` | float | 0.6 | [0.3, 0.9] | Fixed entropy value above which is considered a spike. |

### 5.3 Relative Threshold Method

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `baseline_window` | int | 10 | [5, 30] | Number of initial tokens to establish baseline. |
| `relative_factor` | float | 1.5 | [1.2, 3.0] | Spike if entropy > baseline * factor. |
| `baseline_method` | str | "mean" | ["mean", "median", "min"] | How to compute baseline from window. |

### 5.4 Statistical Method

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `z_score_threshold` | float | 2.0 | [1.5, 3.0] | Spike if entropy > mean + z*std. |
| `rolling_window` | int | 20 | [10, 50] | Window for computing rolling mean/std. |
| `min_samples_for_stats` | int | 5 | [3, 10] | Minimum samples before using statistical method. |

### 5.5 Spike Filtering

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `min_spike_gap` | int | 5 | [1, 20] | Minimum tokens between spike detections. |
| `spike_confirmation` | int | 1 | [1, 3] | Number of consecutive high values to confirm spike. |
| `ignore_first_n` | int | 3 | [0, 10] | Ignore spikes in first N tokens (often noisy). |

---

## 6. Measurement Strategy Parameters

### 6.1 Strategy Selection

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `measurement_strategy` | str | "semantic_boundary" | ["every_token", "every_n", "line_boundary", "semantic_boundary", "adaptive"] | When to measure entropy. |

### 6.2 Every-N Strategy

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `n_tokens` | int | 10 | [1, 50] | Measure every N tokens. |

### 6.3 Line Boundary Strategy (UnCert-CoT Style)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `line_start_triggers` | List[str] | ["\n", "{", "}", ":"] | Tokens that indicate line/block start. |
| `measure_after_newline` | bool | True | Measure at first token after newline. |
| `measure_after_brace` | bool | True | Measure after { or }. |

### 6.4 Semantic Boundary Strategy (Novel)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `semantic_triggers` | List[str] | (see below) | Patterns indicating semantic boundaries. |
| `use_ast_boundaries` | bool | False | Use AST parsing to find boundaries (expensive). |
| `function_call_pattern` | bool | True | Measure at function calls (e.g., "foo("). |
| `import_pattern` | bool | True | Measure at import statements. |
| `assignment_pattern` | bool | True | Measure at variable assignments. |

**Default Semantic Triggers:**

```yaml
function_calls:
  - pattern: "\\w+\\("
  - description: Identifier followed by opening paren

imports:
  - pattern: "^(import|from|require)"
  - description: Import statements

assignments:
  - pattern: "(let|const|var|=)"
  - description: Variable assignments

api_patterns:
  - pattern: "(get|post|put|delete|fetch)\\("
  - description: API calls

database_patterns:
  - pattern: "(query|find|insert|update|delete|select)\\("
  - description: Database operations
```

### 6.5 Adaptive Strategy

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `initial_frequency` | int | 5 | [1, 20] | Initial measurement frequency. |
| `increase_on_spike` | bool | True | Measure more frequently after spike. |
| `post_spike_frequency` | int | 1 | [1, 5] | Frequency after spike detected. |
| `decay_rate` | float | 0.9 | [0.5, 0.99] | How quickly to return to normal frequency. |

---

## 7. Adaptive Retrieval Parameters

### 7.1 Retrieval Triggering

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `retrieval_threshold` | float | 0.3 | [0.1, 0.7] | CCE/uncertainty threshold to trigger retrieval. |
| `require_code_uncertainty` | bool | True | - | Only retrieve if uncertainty_type == "code". |
| `min_tokens_before_retrieval` | int | 10 | [0, 50] | Don't retrieve before generating this many tokens. |
| `max_retrievals` | int | 3 | [1, 10] | Maximum retrieval operations per generation. |

### 7.2 Retrieval Cooldown

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `retrieval_cooldown` | int | 20 | [5, 50] | Minimum tokens between retrievals. |
| `cooldown_after_success` | int | 30 | [10, 100] | Cooldown after successful retrieval (entropy dropped). |
| `cooldown_after_failure` | int | 10 | [5, 30] | Cooldown after failed retrieval (entropy stayed high). |

### 7.3 Retrieval Scope

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `max_files_per_retrieval` | int | 3 | [1, 10] | Maximum files to retrieve at once. |
| `max_tokens_per_retrieval` | int | 2000 | [500, 5000] | Maximum tokens to add per retrieval. |
| `retrieval_depth` | int | 2 | [1, 5] | How many hops in dependency graph to consider. |

---

## 8. Context Management Parameters

### 8.1 Token Budget

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `total_token_budget` | int | 8192 | [2048, 128000] | Total context window budget. |
| `reserved_for_response` | int | 1024 | [256, 4096] | Tokens reserved for model response. |
| `reserved_for_prompt` | int | 512 | [128, 2048] | Tokens reserved for user query. |
| `available_for_context` | int | auto | - | Computed: total - reserved_response - reserved_prompt. |

### 8.2 Context Priorities

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `seed_file_priority` | float | 1.0 | [0.5, 1.0] | Priority for initially retrieved files. |
| `adaptive_retrieval_priority` | float | 0.8 | [0.3, 1.0] | Priority for adaptively retrieved files. |
| `priority_decay` | float | 0.9 | [0.5, 1.0] | Priority multiplier for older context. |

### 8.3 Eviction Policy

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `eviction_policy` | str | "priority" | ["fifo", "lru", "priority", "relevance"] | How to evict context when budget exceeded. |
| `eviction_batch_size` | int | 1 | [1, 5] | Number of files to evict at once. |
| `min_context_retention` | float | 0.5 | [0.2, 0.8] | Minimum fraction of original context to keep. |

### 8.4 Deduplication

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deduplicate_files` | bool | True | Don't add same file twice. |
| `deduplicate_symbols` | bool | True | Don't add overlapping symbol definitions. |
| `merge_overlapping` | bool | True | Merge overlapping code ranges. |

---

## 9. Topic Inference Parameters

### 9.1 Inference Method

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `topic_inference_method` | str | "top_tokens" | ["top_tokens", "attention", "semantic", "hybrid"] | How to infer topic from uncertainty. |

### 9.2 Top Tokens Method

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `top_k_for_topic` | int | 5 | [3, 20] | Number of top tokens to use for topic. |
| `filter_stopwords` | bool | True | - | Remove common stopwords from topic. |
| `min_token_length` | int | 3 | [2, 5] | Minimum token length to include. |
| `prefer_code_tokens` | bool | True | - | Prioritize code-classified tokens. |

### 9.3 Semantic Method

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_model` | str | "all-MiniLM-L6-v2" | Sentence transformer model for embeddings. |
| `similarity_threshold` | float | 0.7 | Minimum similarity to consider relevant. |
| `max_candidates` | int | 10 | Maximum candidate topics to consider. |

### 9.4 Search Query Construction

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `max_query_tokens` | int | 10 | [3, 20] | Maximum tokens in constructed search query. |
| `include_context_keywords` | bool | True | - | Include keywords from surrounding context. |
| `context_keyword_window` | int | 20 | [5, 50] | Tokens of context to scan for keywords. |

---

## 10. Generation Parameters

### 10.1 Model Generation

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `max_new_tokens` | int | 512 | [64, 2048] | Maximum tokens to generate. |
| `generation_temperature` | float | 0.7 | [0.0, 2.0] | Sampling temperature for generation. |
| `top_p` | float | 0.9 | [0.5, 1.0] | Nucleus sampling threshold. |
| `top_k` | int | 50 | [1, 100] | Top-k sampling parameter. |
| `do_sample` | bool | True | - | Whether to sample (vs greedy). |

### 10.2 Generation Loop

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `check_uncertainty_every` | int | 1 | [1, 10] | Check uncertainty every N generated tokens. |
| `pause_on_retrieval` | bool | True | - | Pause generation during retrieval. |
| `restart_on_retrieval` | bool | False | - | Restart generation from scratch with new context. |
| `continue_from_checkpoint` | bool | True | - | Continue from where we paused. |

### 10.3 Early Stopping

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stop_on_low_entropy_streak` | bool | False | Stop if entropy stays low (confident). |
| `low_entropy_threshold` | float | 0.2 | What counts as "low" entropy. |
| `low_entropy_streak_length` | int | 20 | How many tokens of low entropy to stop. |

---

## 11. Model Selection Parameters

### 11.1 Primary Model

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `model_name` | str | "codellama/CodeLlama-7b-Instruct-hf" | See below | Model for generation. |
| `model_precision` | str | "float16" | ["float32", "float16", "bfloat16", "int8", "int4"] | Model precision. |
| `device` | str | "cuda" | ["cuda", "cpu", "mps"] | Device for inference. |
| `device_map` | str | "auto" | ["auto", "balanced", "sequential"] | Multi-GPU strategy. |

**Supported Models:**

```yaml
code_llama:
  - codellama/CodeLlama-7b-Instruct-hf
  - codellama/CodeLlama-13b-Instruct-hf
  - codellama/CodeLlama-34b-Instruct-hf

deepseek:
  - deepseek-ai/deepseek-coder-6.7b-instruct
  - deepseek-ai/deepseek-coder-33b-instruct

starcoder:
  - bigcode/starcoder2-7b
  - bigcode/starcoder2-15b

qwen:
  - Qwen/Qwen2.5-Coder-7B-Instruct
  - Qwen/Qwen2.5-Coder-14B-Instruct
```

### 11.2 Tokenizer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokenizer_name` | str | None | Tokenizer (defaults to model_name). |
| `add_special_tokens` | bool | True | Add BOS/EOS tokens. |
| `padding_side` | str | "left" | Padding side for batched inputs. |

---

## 12. Evaluation Parameters

### 12.1 Dataset

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `num_examples` | int | 100 | [50, 500] | Number of evaluation examples. |
| `difficulty_distribution` | dict | {"easy": 0.3, "medium": 0.5, "hard": 0.2} | - | Distribution of difficulty levels. |
| `category_distribution` | dict | {"comprehension": 0.4, "debugging": 0.3, "architecture": 0.3} | - | Distribution of question types. |

### 12.2 Quality Metrics

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_llm_judge` | bool | True | Use LLM-as-judge for answer quality. |
| `judge_model` | str | "gpt-4" | Model for LLM-as-judge. |
| `judge_temperature` | float | 0.0 | Temperature for judge (0 = deterministic). |
| `human_eval_subset` | int | 20 | Number of examples for human evaluation. |

### 12.3 Efficiency Metrics

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `count_input_tokens` | bool | True | Include input tokens in efficiency calculation. |
| `count_retrieved_tokens` | bool | True | Include retrieved tokens. |
| `count_output_tokens` | bool | True | Include output tokens. |

### 12.4 Uncertainty Metrics

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `spike_iou_threshold` | float | 0.5 | [0.3, 0.8] | IoU threshold for spike detection evaluation. |
| `context_relevance_threshold` | float | 0.7 | [0.5, 0.9] | Similarity threshold for relevant context. |

### 12.5 Statistical Analysis

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `significance_level` | float | 0.05 | [0.01, 0.10] | P-value threshold for significance. |
| `confidence_level` | float | 0.95 | [0.90, 0.99] | Confidence level for intervals. |
| `num_bootstrap_samples` | int | 1000 | [100, 10000] | Bootstrap samples for CI estimation. |

---

## 13. Baseline Configuration Parameters

### 13.1 No Context Baseline

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_query_only` | bool | True | Only include the user query, no code. |

### 13.2 Full Context Baseline

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `max_full_context_tokens` | int | 8000 | [2000, 32000] | Maximum tokens for full context. |
| `full_context_strategy` | str | "bfs" | ["bfs", "dfs", "random", "alphabetical"] | Order to include files. |

### 13.3 BM25 Baseline

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `bm25_k1` | float | 1.5 | [0.5, 2.5] | BM25 term frequency saturation. |
| `bm25_b` | float | 0.75 | [0.0, 1.0] | BM25 length normalization. |
| `bm25_top_k` | int | 5 | [1, 20] | Number of documents to retrieve. |

### 13.4 Embedding Baseline

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_model` | str | "all-MiniLM-L6-v2" | Sentence transformer model. |
| `embedding_top_k` | int | 5 | Number of chunks to retrieve. |
| `chunk_size` | int | 500 | Tokens per chunk. |
| `chunk_overlap` | int | 50 | Overlap between chunks. |

### 13.5 UnCert-CoT Baseline

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `uncert_method` | str | "prob_diff" | "entropy" or "prob_diff". |
| `uncert_threshold` | float | 0.25 | Threshold from UnCert-CoT paper. |
| `measure_at_line_start` | bool | True | UnCert-CoT style measurement. |

---

## 14. Visualization Parameters

### 14.1 Entropy Chart

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chart_width` | int | 800 | Chart width in pixels. |
| `chart_height` | int | 400 | Chart height in pixels. |
| `show_threshold_line` | bool | True | Show horizontal threshold line. |
| `show_baseline_line` | bool | True | Show baseline entropy line. |
| `highlight_spikes` | bool | True | Highlight spike regions. |
| `spike_color` | str | "#ff6b6b" | Color for spike highlighting. |

### 14.2 Timeline

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show_retrieval_events` | bool | True | Show retrieval markers on timeline. |
| `show_context_growth` | bool | True | Show cumulative context size. |
| `retrieval_marker_style` | str | "vertical_line" | Style for retrieval markers. |

### 14.3 Export

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `export_format` | str | "png" | ["png", "svg", "pdf", "html"]. |
| `dpi` | int | 150 | Resolution for raster formats. |
| `include_data_table` | bool | True | Include raw data with visualization. |

---

## 15. Experimental Design Choices

### 15.1 Ablation Variables

| Ablation | Values to Test | Description |
|----------|----------------|-------------|
| CCE vs Raw Entropy | {cce, normalized, raw, prob_diff} | Which uncertainty method is best? |
| Token Classification | {with_classification, without_classification} | Does code/language filtering help? |
| Measurement Points | {every_token, every_10, line_boundary, semantic_boundary} | Where to measure? |
| Threshold Values | {0.1, 0.2, 0.3, 0.4, 0.5} | What's the optimal threshold? |
| Code Token Set Size | {minimal, standard, extended} | How many code tokens to include? |
| Retrieval Depth | {1, 2, 3} | How many dependency hops? |

### 15.2 Independent Variables

| Variable | Type | Description |
|----------|------|-------------|
| Method | categorical | Uncertainty detection method |
| Threshold | continuous | Retrieval trigger threshold |
| Measurement Strategy | categorical | When to measure entropy |
| Model Size | categorical | 7B, 13B, 34B |
| Question Difficulty | categorical | easy, medium, hard |

### 15.3 Dependent Variables

| Variable | Type | Description |
|----------|------|-------------|
| Answer Correctness | continuous [0,1] | LLM-judged quality |
| Tokens Used | integer | Total tokens consumed |
| Retrieval Count | integer | Number of retrievals |
| Spike Precision | continuous [0,1] | Precision of spike detection |
| Spike Recall | continuous [0,1] | Recall of spike detection |
| Latency | continuous (ms) | Time to generate answer |

### 15.4 Control Variables

| Variable | Held Constant At | Description |
|----------|------------------|-------------|
| Temperature | 0.7 | Generation temperature |
| Max Tokens | 512 | Maximum response length |
| Random Seed | 42 | For reproducibility |
| Model | CodeLlama-13B | Unless varied |

---

## Appendix A: Recommended Configurations

### A.1 Quick Experiment (Development)

```yaml
model: codellama/CodeLlama-7b-Instruct-hf
uncertainty_method: prob_differential
measurement_strategy: every_10
num_examples: 20
max_new_tokens: 256
```

### A.2 Full Experiment (Paper Results)

```yaml
model: codellama/CodeLlama-13b-Instruct-hf
uncertainty_method: cce
measurement_strategy: semantic_boundary
num_examples: 100
max_new_tokens: 512
num_bootstrap_samples: 1000
```

### A.3 Ablation Study

```yaml
models:
  - codellama/CodeLlama-7b-Instruct-hf
  - codellama/CodeLlama-13b-Instruct-hf

uncertainty_methods:
  - raw_entropy
  - normalized_entropy
  - prob_differential
  - cce

thresholds: [0.1, 0.2, 0.3, 0.4, 0.5]

measurement_strategies:
  - every_token
  - line_boundary
  - semantic_boundary
```

---

## Appendix B: Sensitivity Analysis Ranges

| Parameter | Low | Default | High | Priority |
|-----------|-----|---------|------|----------|
| cce_threshold | 0.1 | 0.3 | 0.7 | Critical |
| baseline_window | 5 | 10 | 30 | High |
| relative_factor | 1.2 | 1.5 | 3.0 | High |
| min_spike_gap | 1 | 5 | 20 | Medium |
| retrieval_cooldown | 5 | 20 | 50 | Medium |
| max_files_per_retrieval | 1 | 3 | 10 | Medium |
| code_weight | 0.5 | 1.0 | 2.0 | Low |
| language_weight | 0.5 | 1.0 | 2.0 | Low |

---

*Document created: December 2024*
*Last updated: December 2024*
