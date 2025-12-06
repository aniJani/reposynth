# ContextLoom Research Implementation Plan

## Project Title
**"Contrastive Code Entropy: Uncertainty-Guided Adaptive Context Retrieval for LLM Code Understanding"**

---

## Executive Summary

This document outlines a phased implementation plan to transform RepoSynth into a research contribution. The core innovation is **Contrastive Code Entropy (CCE)** - a novel uncertainty metric that distinguishes between linguistic uncertainty (word choice) and knowledge uncertainty (missing code context), enabling targeted context retrieval only when the model lacks technical knowledge.

---

## Research Contributions (Claimed Novelty)

| # | Contribution | Status |
|---|--------------|--------|
| 1 | **Contrastive Code Entropy (CCE)** - Novel metric distinguishing code vs language uncertainty | New |
| 2 | **Adaptive Context Retrieval** - Entropy-triggered retrieval during generation | New application |
| 3 | **Code Token Taxonomy** - Classification of vocabulary into code/language categories | New |
| 4 | **Evaluation Framework** - Benchmarks for context-aware code Q&A | New |

### What We Cite (Not Claiming as Novel)
- TOON format for serialization [toon-format/spec]
- Entropy-based uncertainty detection [ARPO, UnCert-CoT]
- Knapsack optimization for context selection [known technique]
- Tree-sitter for AST parsing [existing tool]
- Sentence-transformers for embeddings [existing tool]

---

## Phase 0: Research Foundation (Week 1)

### 0.1 Literature Review
- [ ] Read and summarize ARPO paper (arXiv:2507.19849)
- [ ] Read and summarize UnCert-CoT paper (arXiv:2503.15341)
- [ ] Read and summarize FlowForge paper (arXiv:2507.15559)
- [ ] Survey existing code RAG systems (Cursor, Continue, Cody, etc.)
- [ ] Document gaps in existing approaches

### 0.2 Research Questions
1. **RQ1**: Can entropy-based uncertainty detection identify when an LLM lacks code context?
2. **RQ2**: Does Contrastive Code Entropy outperform raw entropy for triggering context retrieval?
3. **RQ3**: Does adaptive context retrieval improve answer quality while reducing token usage?
4. **RQ4**: What are the optimal thresholds and measurement points for code Q&A tasks?

### 0.3 Hypotheses
- **H1**: CCE > 0.3 indicates missing code context with >80% precision
- **H2**: Adaptive retrieval achieves same answer quality with 40% fewer tokens
- **H3**: Measuring at semantic boundaries (function calls, imports) outperforms line boundaries

### Deliverables
- [ ] Literature review document (2-3 pages)
- [ ] Research questions and hypotheses document
- [ ] Related work section draft

---

## Phase 1: Core Entropy Implementation (Weeks 2-3)

### 1.1 Basic Entropy Calculation Module

**File**: `packages/python-orchestrator/orchestrator/entropy/calculator.py`

```python
# Implement these functions:
- compute_token_entropy(logits) -> float
- compute_normalized_entropy(logits, vocab_size) -> float  # [0,1]
- compute_prob_differential(logits) -> float  # UnCert-CoT style
- compute_sequence_entropy(all_logits) -> List[float]
```

### 1.2 Token Classification Module

**File**: `packages/python-orchestrator/orchestrator/entropy/token_classifier.py`

```python
# Implement these:
- build_code_token_set(tokenizer) -> Set[int]
- build_language_token_set(tokenizer) -> Set[int]
- classify_token(token_id, tokenizer) -> Literal["code", "language", "other"]
- TokenClassifier class with pre-computed sets
```

### 1.3 Contrastive Code Entropy Module

**File**: `packages/python-orchestrator/orchestrator/entropy/cce.py`

```python
# Core novel contribution:
- compute_code_entropy(logits, code_token_ids) -> float
- compute_language_entropy(logits, lang_token_ids) -> float
- compute_cce(logits, code_ids, lang_ids) -> CCEResult
- CCEResult dataclass with:
  - code_entropy: float
  - language_entropy: float
  - contrastive_entropy: float
  - uncertainty_type: Literal["code", "language", "mixed"]
  - should_retrieve: bool
```

### 1.4 Unit Tests

**File**: `tests/test_entropy.py`

- [ ] Test entropy calculation correctness
- [ ] Test CCE with synthetic logits (known distributions)
- [ ] Test token classification coverage
- [ ] Test edge cases (empty, uniform, peaked distributions)

### Deliverables
- [ ] Entropy calculation module with tests
- [ ] Token classifier with code/language taxonomy
- [ ] CCE implementation with documentation
- [ ] Validation against known distributions

---

## Phase 2: Uncertainty Monitor Integration (Weeks 4-5)

### 2.1 Uncertainty Monitor Class

**File**: `packages/python-orchestrator/orchestrator/entropy/monitor.py`

```python
class UncertaintyMonitor:
    """
    Monitors LLM generation and detects uncertainty.

    Supports multiple methods:
    - raw_entropy: Standard Shannon entropy
    - normalized_entropy: [0,1] normalized
    - prob_differential: UnCert-CoT style
    - cce: Contrastive Code Entropy (novel)
    """

    def __init__(self, method, threshold, tokenizer, measure_at)
    def should_measure(self, token, position) -> bool
    def compute_uncertainty(self, logits) -> UncertaintyResult
    def detect_spike(self, history) -> Optional[SpikeInfo]
    def should_request_context(self, result) -> bool
```

### 2.2 Measurement Point Strategies

**File**: `packages/python-orchestrator/orchestrator/entropy/measurement.py`

```python
# Different strategies for when to measure entropy:

class MeasurementStrategy(ABC):
    def should_measure(self, token, position, context) -> bool

class EveryTokenStrategy(MeasurementStrategy): ...
class LineStartStrategy(MeasurementStrategy): ...  # UnCert-CoT style
class SemanticBoundaryStrategy(MeasurementStrategy): ...  # Novel
class AdaptiveStrategy(MeasurementStrategy): ...  # Novel
```

Semantic boundaries to detect:
- Function/method calls
- Import statements
- Variable assignments referencing external modules
- API endpoint references
- Database queries

### 2.3 Spike Detection Algorithm

**File**: `packages/python-orchestrator/orchestrator/entropy/spike_detector.py`

```python
class SpikeDetector:
    """
    Detects entropy spikes indicating uncertainty.

    Methods:
    - threshold: Simple fixed threshold
    - relative: Spike if > baseline * factor
    - statistical: Spike if > mean + k*std
    - adaptive: Learned threshold (future RL work)
    """

    def __init__(self, method, params)
    def update(self, entropy_value)
    def is_spike(self) -> bool
    def get_spike_info(self) -> SpikeInfo
```

### 2.4 Integration Tests

- [ ] Test monitor with real model (CodeLlama-7B or similar)
- [ ] Test different measurement strategies
- [ ] Test spike detection accuracy
- [ ] Benchmark latency overhead

### Deliverables
- [ ] UncertaintyMonitor class
- [ ] Multiple measurement strategies
- [ ] Spike detection with multiple methods
- [ ] Integration tests with real models

---

## Phase 3: Adaptive Context Retrieval (Weeks 6-7)

### 3.1 Topic Inference from Uncertainty

**File**: `packages/python-orchestrator/orchestrator/retrieval/topic_inference.py`

```python
class TopicInferrer:
    """
    Infers what topic/concept the model is uncertain about.

    Methods:
    - top_tokens: Use top-k predicted tokens as search terms
    - attention: Use attention weights to identify relevant context
    - semantic: Embed uncertain region and find similar code
    """

    def infer_topic(self, logits, context, position) -> str
    def extract_code_identifiers(self, top_tokens) -> List[str]
    def find_related_symbols(self, topic, symbol_registry) -> List[str]
```

### 3.2 Adaptive Retriever

**File**: `packages/python-orchestrator/orchestrator/retrieval/adaptive.py`

```python
class AdaptiveContextRetriever:
    """
    Retrieves additional context when uncertainty is detected.

    Integrates with existing RepoSynth retriever.
    """

    def __init__(self, base_retriever, uncertainty_monitor, topic_inferrer)

    def retrieve_on_uncertainty(
        self,
        uncertainty_result: UncertaintyResult,
        current_context: str,
        query: str
    ) -> RetrievalResult

    def should_stop_retrieving(self, iterations, token_budget) -> bool
```

### 3.3 Generation Loop with Adaptive Retrieval

**File**: `packages/python-orchestrator/orchestrator/generation/adaptive_generator.py`

```python
class AdaptiveGenerator:
    """
    Generates responses with uncertainty-triggered context retrieval.
    """

    def generate(
        self,
        query: str,
        initial_context: str,
        max_tokens: int,
        max_retrievals: int = 3
    ) -> GenerationResult

    # GenerationResult includes:
    # - response: str
    # - entropy_trace: List[float]
    # - cce_trace: List[CCEResult]
    # - context_requests: List[ContextRequest]
    # - final_context: str
    # - total_tokens_used: int
```

### 3.4 Context Management

**File**: `packages/python-orchestrator/orchestrator/retrieval/context_manager.py`

```python
class ContextManager:
    """
    Manages context window during adaptive retrieval.

    Handles:
    - Token budget tracking
    - Context deduplication
    - Priority-based eviction (remove low-value context when full)
    - Context provenance tracking
    """

    def add_context(self, content, source, priority) -> bool
    def evict_lowest_priority(self, tokens_needed) -> List[str]
    def get_current_context(self) -> str
    def get_provenance(self) -> Dict[str, str]
```

### Deliverables
- [ ] Topic inference from uncertainty
- [ ] Adaptive retriever integrated with RepoSynth
- [ ] Generation loop with retrieval
- [ ] Context management with eviction

---

## Phase 4: Visualization System (Week 8)

### 4.1 Data Structures for Visualization

**File**: `packages/python-orchestrator/orchestrator/visualization/data.py`

```python
@dataclass
class EntropyVisualizationData:
    # Token-level data
    tokens: List[str]
    positions: List[int]
    entropy_values: List[float]
    cce_values: List[CCEResult]

    # Spike data
    spike_positions: List[int]
    spike_reasons: List[str]

    # Context retrieval data
    retrieval_events: List[RetrievalEvent]

    # Thresholds
    entropy_threshold: float
    cce_threshold: float
    baseline_entropy: float

    def to_json(self) -> str
    def to_chart_data(self) -> Dict
```

### 4.2 Visualization API

**File**: `packages/python-orchestrator/orchestrator/visualization/api.py`

```python
# FastAPI endpoints for visualization

@app.get("/entropy-trace/{session_id}")
def get_entropy_trace(session_id: str) -> EntropyVisualizationData

@app.get("/retrieval-timeline/{session_id}")
def get_retrieval_timeline(session_id: str) -> List[RetrievalEvent]

@app.websocket("/entropy-stream/{session_id}")
async def stream_entropy(websocket: WebSocket, session_id: str)
```

### 4.3 Frontend Visualization (Optional)

**File**: `packages/frontend/src/components/EntropyChart.tsx`

- Line chart showing entropy over tokens
- Highlighted spike regions
- Context retrieval markers
- Code vs language entropy comparison
- Interactive tooltip with token details

### Deliverables
- [ ] Visualization data structures
- [ ] API endpoints for visualization data
- [ ] Basic frontend chart (optional, can use notebooks)
- [ ] Jupyter notebook for analysis visualization

---

## Phase 5: Evaluation Framework (Weeks 9-10)

### 5.1 Benchmark Dataset Creation

**File**: `research/benchmarks/dataset.py`

Create a benchmark dataset for code Q&A with:

```python
@dataclass
class CodeQAExample:
    query: str                      # User question
    repository: str                 # Source repository
    ground_truth_files: List[str]   # Files needed to answer
    ground_truth_answer: str        # Correct answer
    difficulty: str                 # easy/medium/hard
    category: str                   # comprehension/debugging/architecture
```

Dataset sources:
- [ ] Curate 100 examples from open-source repos (TypeScript, Python)
- [ ] Include varying difficulty levels
- [ ] Include different question types
- [ ] Document ground truth context requirements

### 5.2 Evaluation Metrics

**File**: `research/evaluation/metrics.py`

```python
# Quality Metrics
def answer_correctness(predicted, ground_truth) -> float  # LLM-as-judge
def answer_completeness(predicted, ground_truth) -> float
def hallucination_rate(predicted, context_provided) -> float

# Efficiency Metrics
def context_precision(retrieved_files, ground_truth_files) -> float
def context_recall(retrieved_files, ground_truth_files) -> float
def token_efficiency(tokens_used, baseline_tokens) -> float

# Uncertainty Metrics
def spike_precision(detected_spikes, actual_missing_context) -> float
def spike_recall(detected_spikes, actual_missing_context) -> float
def cce_correlation(cce_values, context_needed) -> float
```

### 5.3 Baseline Implementations

**File**: `research/baselines/`

```
baselines/
├── no_context.py        # Answer without any context
├── full_context.py      # Include entire codebase (up to limit)
├── random_context.py    # Random file selection
├── bm25_context.py      # BM25 keyword retrieval
├── embedding_context.py # Pure embedding-based retrieval
├── reposynth_base.py    # RepoSynth without entropy monitoring
└── uncert_cot.py        # UnCert-CoT style (line boundary only)
```

### 5.4 Experiment Runner

**File**: `research/experiments/runner.py`

```python
class ExperimentRunner:
    def run_experiment(
        self,
        method: str,           # "cce", "raw_entropy", "prob_diff", "baseline"
        dataset: List[CodeQAExample],
        config: ExperimentConfig
    ) -> ExperimentResults

    def compare_methods(
        self,
        methods: List[str],
        dataset: List[CodeQAExample]
    ) -> ComparisonResults
```

### 5.5 Statistical Analysis

**File**: `research/analysis/statistics.py`

```python
# Statistical tests for significance
def paired_t_test(method_a_scores, method_b_scores) -> PValue
def wilcoxon_signed_rank(method_a_scores, method_b_scores) -> PValue
def effect_size_cohens_d(method_a_scores, method_b_scores) -> float
def confidence_interval(scores, confidence=0.95) -> Tuple[float, float]
```

### Deliverables
- [ ] Benchmark dataset (100 examples)
- [ ] Evaluation metrics implementation
- [ ] Baseline implementations
- [ ] Experiment runner
- [ ] Statistical analysis tools

---

## Phase 6: Experiments (Weeks 11-12)

### 6.1 Experiment 1: CCE vs Raw Entropy

**Research Question**: Does CCE outperform raw entropy for detecting missing code context?

**Setup**:
- Dataset: 100 code Q&A examples
- Models: CodeLlama-7B, CodeLlama-13B
- Metrics: Spike precision, spike recall, F1

**Conditions**:
1. Raw entropy (threshold τ = 0.3)
2. Normalized entropy (threshold τ = 0.25)
3. Probability differential (threshold τ = 0.25)
4. CCE (threshold τ = 0.3)

### 6.2 Experiment 2: Adaptive Retrieval Effectiveness

**Research Question**: Does adaptive retrieval improve answer quality?

**Setup**:
- Dataset: 100 code Q&A examples
- Model: CodeLlama-13B
- Metrics: Answer correctness, token efficiency, hallucination rate

**Conditions**:
1. No context (baseline)
2. Full context (upper bound)
3. Static retrieval (RepoSynth without entropy)
4. Adaptive retrieval (CCE-triggered)

### 6.3 Experiment 3: Measurement Point Strategies

**Research Question**: Where should entropy be measured?

**Setup**:
- Dataset: 50 code Q&A examples
- Model: CodeLlama-7B
- Metrics: Latency, retrieval precision, answer quality

**Conditions**:
1. Every token
2. Every 10 tokens
3. Line boundaries (UnCert-CoT style)
4. Semantic boundaries (novel)

### 6.4 Experiment 4: Threshold Sensitivity

**Research Question**: What are optimal thresholds for CCE?

**Setup**:
- Dataset: 100 examples
- Model: CodeLlama-7B
- Thresholds: τ ∈ {0.1, 0.2, 0.3, 0.4, 0.5}

**Analysis**:
- Precision-recall curves
- F1 vs threshold
- Optimal operating point

### 6.5 Experiment 5: Ablation Study

**Research Question**: Which components of CCE contribute most?

**Ablations**:
1. CCE without code token filtering
2. CCE without language token filtering
3. CCE with different token taxonomies
4. CCE with different normalization

### Deliverables
- [ ] Experiment 1 results and analysis
- [ ] Experiment 2 results and analysis
- [ ] Experiment 3 results and analysis
- [ ] Experiment 4 results and analysis
- [ ] Experiment 5 results and analysis
- [ ] Results tables and figures

---

## Phase 7: Paper Writing (Weeks 13-14)

### 7.1 Paper Structure

```
1. Abstract (200 words)

2. Introduction (1.5 pages)
   - Problem: LLMs lack context awareness during generation
   - Limitation: Existing methods can't distinguish code vs language uncertainty
   - Contribution: CCE + adaptive retrieval
   - Results summary

3. Related Work (1 page)
   - Uncertainty in LLMs (entropy-based methods)
   - Code retrieval and RAG
   - Chain-of-thought and adaptive reasoning

4. Method (2.5 pages)
   4.1 Problem Formulation
   4.2 Contrastive Code Entropy
   4.3 Token Classification
   4.4 Adaptive Context Retrieval
   4.5 Implementation Details

5. Experiments (2 pages)
   5.1 Experimental Setup
   5.2 RQ1: CCE vs Raw Entropy
   5.3 RQ2: Adaptive Retrieval Effectiveness
   5.4 RQ3: Measurement Strategies
   5.5 Ablation Studies

6. Results and Discussion (1.5 pages)
   - Main findings
   - Limitations
   - Failure cases

7. Conclusion (0.5 pages)

References
Appendix (supplementary material)
```

### 7.2 Figures to Create

1. **Figure 1**: System architecture diagram
2. **Figure 2**: CCE vs raw entropy visualization (the key insight)
3. **Figure 3**: Entropy trace with retrieval events
4. **Figure 4**: Precision-recall curves for spike detection
5. **Figure 5**: Answer quality vs token usage tradeoff
6. **Table 1**: Main results comparison
7. **Table 2**: Ablation study results

### 7.3 Writing Schedule

| Day | Section | Status |
|-----|---------|--------|
| 1-2 | Method section draft | [ ] |
| 3-4 | Experiments section draft | [ ] |
| 5 | Results section draft | [ ] |
| 6 | Introduction draft | [ ] |
| 7 | Related work draft | [ ] |
| 8 | Abstract + conclusion | [ ] |
| 9-10 | Revision and polish | [ ] |
| 11-12 | Figures and tables | [ ] |
| 13-14 | Final revision | [ ] |

### Deliverables
- [ ] Complete paper draft
- [ ] All figures and tables
- [ ] Supplementary material
- [ ] Code repository cleaned for release

---

## Phase 8: Submission Preparation (Week 15)

### 8.1 Target Venues

| Venue | Deadline | Focus |
|-------|----------|-------|
| ICSE 2026 | TBD | Software Engineering |
| FSE 2026 | TBD | Software Engineering |
| ACL 2026 | TBD | NLP/Computational Linguistics |
| EMNLP 2025 | TBD | NLP |
| NeurIPS 2025 | May 2025 | ML |
| ICLR 2026 | Sep 2025 | ML |

### 8.2 Submission Checklist

- [ ] Paper formatted for venue
- [ ] Supplementary material prepared
- [ ] Code repository cleaned
- [ ] Anonymous submission (if required)
- [ ] All co-authors approved
- [ ] Ethics statement (if required)

### 8.3 Artifact Preparation

```
artifact/
├── README.md           # Installation and usage
├── requirements.txt    # Dependencies
├── setup.py           # Package installation
├── data/
│   └── benchmark/     # Evaluation dataset
├── src/
│   └── cce/          # Core CCE implementation
├── experiments/
│   ├── configs/      # Experiment configurations
│   └── scripts/      # Reproduction scripts
├── results/
│   └── figures/      # Generated figures
└── notebooks/
    └── analysis.ipynb # Analysis notebooks
```

---

## Resource Requirements

### Compute
- GPU: 1x A100 (40GB) or 2x RTX 3090 for experiments
- Estimated GPU hours: 100-200 hours total

### Models
- CodeLlama-7B-Instruct
- CodeLlama-13B-Instruct
- (Optional) DeepSeek-Coder, StarCoder2

### Data
- Open-source repositories for benchmark
- Existing benchmarks: HumanEval, MBPP (for reference)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| CCE doesn't outperform baselines | Pivot to analysis of when/why entropy methods fail |
| Compute limitations | Use smaller models, reduce dataset size |
| Evaluation subjectivity | Use multiple LLM judges, human evaluation subset |
| Time constraints | Prioritize core experiments (1, 2), defer ablations |

---

## Timeline Summary

| Phase | Duration | Key Milestone |
|-------|----------|---------------|
| 0: Foundation | Week 1 | Literature review complete |
| 1: Entropy Implementation | Weeks 2-3 | CCE module working |
| 2: Uncertainty Monitor | Weeks 4-5 | Monitor integrated |
| 3: Adaptive Retrieval | Weeks 6-7 | End-to-end system working |
| 4: Visualization | Week 8 | Visualization ready |
| 5: Evaluation Framework | Weeks 9-10 | Benchmarks ready |
| 6: Experiments | Weeks 11-12 | Results collected |
| 7: Paper Writing | Weeks 13-14 | Draft complete |
| 8: Submission | Week 15 | Paper submitted |

**Total: ~15 weeks (4 months)**

---

## Success Criteria

1. **Technical**: CCE achieves >10% improvement in spike detection F1 over raw entropy
2. **Efficiency**: Adaptive retrieval uses 30% fewer tokens with equal answer quality
3. **Publication**: Paper accepted at top-tier venue (ICSE, FSE, ACL, or equivalent)
4. **Artifact**: Open-source release with documentation

---

## References

1. ARPO: Agentic Reinforced Policy Optimization (arXiv:2507.19849)
2. UnCert-CoT: Uncertainty-Guided Chain-of-Thought (arXiv:2503.15341)
3. FlowForge: Multi-agent Workflow Design (arXiv:2507.15559)
4. TOON: Token-Oriented Object Notation (github.com/toon-format/spec)

---

*Document created: December 2024*
*Last updated: December 2024*
*Authors: [Your Names]*
