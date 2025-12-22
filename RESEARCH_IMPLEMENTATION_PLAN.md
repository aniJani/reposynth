# RepoSynth Research Implementation Plan
## Contrastive Code Entropy (CCE) - Actionable Roadmap

**Goal**: Transform RepoSynth into a publishable research contribution
**Timeline**: 15 weeks (4 months)
**Target Venues**: ICSE 2026, FSE 2026, ACL 2026

---

## PHASE 0: Foundation & Validation (Week 1)
**Goal**: Validate core hypothesis before full implementation

### Week 1 - Tasks

#### Literature Review (Days 1-2)
- [ ] Read ARPO paper (arXiv:2507.19849) - focus on entropy-based uncertainty
- [ ] Read UnCert-CoT (arXiv:2503.15341) - line boundary measurement approach
- [ ] Read FlowForge (arXiv:2507.15559) - multi-agent workflow context
- [ ] Survey 3 existing code RAG tools (Cursor, Continue, Cody)
- [ ] **Deliverable**: `research/literature_review.md` with key takeaways

#### Research Questions & Hypotheses (Day 3)
- [ ] Document 4 research questions (RQ1-RQ4)
- [ ] Formalize 3 hypotheses with measurable targets
- [ ] **Deliverable**: `research/research_questions.md`

#### Proof-of-Concept Experiment (Days 4-5)
**Quick validation before committing to full implementation**

```bash
# Goal: Manually verify that entropy differs between code/language uncertainty
# Setup: Run CodeLlama-7B on 10 hand-crafted examples
# Expected: See higher entropy on missing API vs synonym choice
```

Tasks:
- [ ] Create 10 test examples (5 missing code context, 5 language choice)
- [ ] Write script to extract logits from model
- [ ] Calculate raw entropy at decision points
- [ ] Manually inspect top-k token predictions
- [ ] **Validation**: Does entropy correlate with missing context?

**Decision Point**: If POC fails, pivot plan before investing in full implementation

---

## PHASE 1: Core Entropy Implementation (Weeks 2-3)
**Goal**: Build and validate entropy calculation modules

### Week 2 - Basic Entropy Calculator

#### Day 1-2: Entropy Calculation Module
**File**: `packages/python-orchestrator/orchestrator/entropy/calculator.py`

```python
class EntropyCalculator:
    """Compute various entropy metrics from logits."""

    @staticmethod
    def shannon_entropy(logits: np.ndarray) -> float:
        """Standard Shannon entropy: H = -Σ p(x) log p(x)"""

    @staticmethod
    def normalized_entropy(logits: np.ndarray) -> float:
        """Normalize to [0, 1]: H_norm = H / log(V)"""

    @staticmethod
    def probability_differential(logits: np.ndarray) -> float:
        """UnCert-CoT style: 1 - max(P)"""

    @staticmethod
    def top_k_entropy(logits: np.ndarray, k: int = 10) -> float:
        """Entropy over top-k tokens only"""
```

Tasks:
- [ ] Implement 4 entropy functions
- [ ] Add input validation and edge cases
- [ ] Write unit tests with known distributions
- [ ] Benchmark performance (should be <1ms per call)

#### Day 3-4: Token Classification
**File**: `packages/python-orchestrator/orchestrator/entropy/token_classifier.py`

**Strategy**: Start simple, iterate later
1. **Code tokens**: Programming keywords, operators, brackets
2. **Language tokens**: Common English words (use NLTK word list)
3. **Other**: Everything else (domain-specific, variables)

```python
class TokenClassifier:
    def __init__(self, tokenizer):
        self.code_tokens = self._build_code_token_set(tokenizer)
        self.language_tokens = self._build_language_token_set(tokenizer)

    def classify(self, token_id: int) -> Literal["code", "language", "other"]

    def _build_code_token_set(self, tokenizer) -> Set[int]:
        # Keywords from Python, JS, TS, Java, Go, Rust
        # Operators: +, -, =, ==, !=, &&, ||, etc.
        # Brackets: (, ), [, ], {, }

    def _build_language_token_set(self, tokenizer) -> Set[int]:
        # Top 5000 English words from NLTK
        # Common function words: the, is, are, was, were, etc.
```

Tasks:
- [ ] Create programming keywords list (all major languages)
- [ ] Load English word list (NLTK or similar)
- [ ] Map strings to token IDs for model tokenizer
- [ ] Test coverage: what % of vocab is classified?
- [ ] **Target**: >70% of tokens classified as code or language

#### Day 5: Contrastive Code Entropy (CCE)
**File**: `packages/python-orchestrator/orchestrator/entropy/cce.py`

```python
@dataclass
class CCEResult:
    code_entropy: float           # H over code tokens only
    language_entropy: float       # H over language tokens only
    contrastive_entropy: float    # code_H - language_H
    total_entropy: float          # Standard H over all tokens
    uncertainty_type: str         # "code", "language", "mixed", "low"
    should_retrieve: bool         # Decision flag

class CCECalculator:
    def __init__(self, token_classifier: TokenClassifier, threshold: float = 0.3):
        self.classifier = token_classifier
        self.threshold = threshold

    def compute_cce(self, logits: np.ndarray) -> CCEResult:
        """
        Core CCE algorithm:
        1. Partition logits into code_logits and lang_logits
        2. Compute H_code and H_lang separately
        3. CCE = H_code - H_lang
        4. If CCE > threshold => code uncertainty => retrieve
        """
```

Tasks:
- [ ] Implement CCE algorithm
- [ ] Test with synthetic logits (known distributions)
- [ ] Verify edge cases (all code, all language, uniform)
- [ ] **Validation**: Manual test on 10 POC examples

### Week 3 - Testing & Validation

#### Day 1-2: Unit Tests
**File**: `tests/entropy/test_cce.py`

Test scenarios:
- [ ] Peaked distribution (low entropy)
- [ ] Uniform distribution (high entropy)
- [ ] Code-heavy distribution (high code_H, low lang_H)
- [ ] Language-heavy distribution (low code_H, high lang_H)
- [ ] Empty/invalid inputs

#### Day 3-4: Integration with Real Model
- [ ] Load CodeLlama-7B model
- [ ] Write script to extract logits during generation
- [ ] Run on 20 examples (10 code uncertain, 10 language uncertain)
- [ ] Plot entropy traces
- [ ] **Validation**: CCE visually separates the two cases

#### Day 5: Documentation & Cleanup
- [ ] Write docstrings for all modules
- [ ] Create `entropy/README.md` explaining CCE
- [ ] Add usage examples
- [ ] **Deliverable**: Working CCE module with tests

---

## PHASE 2: Uncertainty Monitoring (Weeks 4-5)
**Goal**: Detect uncertainty during generation in real-time

### Week 4 - Uncertainty Monitor

#### Day 1-3: Monitor Core Class
**File**: `packages/python-orchestrator/orchestrator/entropy/monitor.py`

```python
class UncertaintyMonitor:
    """
    Monitors generation and detects when to retrieve context.

    Supports multiple methods:
    - raw_entropy: Simple Shannon entropy
    - normalized_entropy: [0,1] normalized
    - prob_differential: UnCert-CoT style
    - cce: Contrastive Code Entropy (our method)
    """

    def __init__(
        self,
        method: str = "cce",
        threshold: float = 0.3,
        tokenizer = None,
        measurement_strategy: str = "semantic_boundary"
    ):
        self.method = method
        self.threshold = threshold
        self.measurement_strategy = self._create_strategy(measurement_strategy)
        self.cce_calculator = CCECalculator(TokenClassifier(tokenizer))
        self.history = []  # Store entropy trace

    def should_measure(self, token: str, position: int, context: str) -> bool:
        """Check if we should measure entropy at this position."""
        return self.measurement_strategy.should_measure(token, position, context)

    def measure_uncertainty(self, logits: np.ndarray) -> UncertaintyResult:
        """Compute uncertainty metric based on method."""
        if self.method == "cce":
            cce_result = self.cce_calculator.compute_cce(logits)
            self.history.append(cce_result)
            return UncertaintyResult(
                value=cce_result.contrastive_entropy,
                should_retrieve=cce_result.should_retrieve,
                details=cce_result
            )
        elif self.method == "raw_entropy":
            # ... other methods

    def detect_spike(self) -> Optional[SpikeInfo]:
        """Detect if current entropy is a spike vs baseline."""
```

Tasks:
- [ ] Implement UncertaintyMonitor class
- [ ] Support 4 methods (raw, normalized, prob_diff, cce)
- [ ] Implement history tracking
- [ ] Write tests for each method

#### Day 4-5: Measurement Strategies
**File**: `packages/python-orchestrator/orchestrator/entropy/measurement.py`

```python
class MeasurementStrategy(ABC):
    @abstractmethod
    def should_measure(self, token: str, position: int, context: str) -> bool:
        pass

class EveryTokenStrategy(MeasurementStrategy):
    def should_measure(self, token, position, context) -> bool:
        return True

class LineStartStrategy(MeasurementStrategy):
    """UnCert-CoT: Measure at line boundaries."""
    def should_measure(self, token, position, context) -> bool:
        return token == "\n" or position == 0

class SemanticBoundaryStrategy(MeasurementStrategy):
    """Our approach: Measure at code semantic boundaries."""

    def should_measure(self, token, position, context) -> bool:
        # Detect:
        # - Function calls: foo(
        # - Imports: import, from
        # - Assignments: variable =
        # - Method access: object.
        # - Type annotations: : Type

        triggers = ["(", "import ", "from ", " = ", ".", ":"]
        return any(context.endswith(trigger) for trigger in triggers)
```

Tasks:
- [ ] Implement 4 measurement strategies
- [ ] Test on sample code snippets
- [ ] Measure overhead (latency impact)
- [ ] **Target**: Semantic strategy measures 10-20% of tokens

### Week 5 - Spike Detection & Integration

#### Day 1-2: Spike Detector
**File**: `packages/python-orchestrator/orchestrator/entropy/spike_detector.py`

```python
class SpikeDetector:
    """Detects entropy spikes indicating uncertainty."""

    def __init__(self, method: str = "threshold", params: dict = None):
        self.method = method
        self.baseline = None
        self.history = deque(maxlen=100)  # Rolling window

    def update(self, entropy_value: float):
        """Update detector with new entropy value."""
        self.history.append(entropy_value)
        if self.baseline is None and len(self.history) >= 10:
            self.baseline = np.mean(self.history)

    def is_spike(self, current_value: float) -> bool:
        """Determine if current value is a spike."""
        if self.method == "threshold":
            return current_value > self.threshold
        elif self.method == "relative":
            return current_value > self.baseline * self.factor
        elif self.method == "statistical":
            mean = np.mean(self.history)
            std = np.std(self.history)
            return current_value > mean + self.k_sigma * std
```

Tasks:
- [ ] Implement 3 spike detection methods
- [ ] Test with synthetic spike data
- [ ] Tune parameters on POC examples
- [ ] Compare precision/recall of methods

#### Day 3-5: End-to-End Integration Test
- [ ] Integrate monitor with model generation loop
- [ ] Test on 30 examples (code Q&A)
- [ ] Visualize entropy traces
- [ ] Measure false positive/negative rates
- [ ] **Deliverable**: Working uncertainty monitor

---

## PHASE 3: Adaptive Context Retrieval (Weeks 6-7)
**Goal**: Retrieve context when uncertainty is detected

### Week 6 - Topic Inference & Retrieval

#### Day 1-2: Topic Inference
**File**: `packages/python-orchestrator/orchestrator/retrieval/topic_inference.py`

```python
class TopicInferrer:
    """Infer what the model is uncertain about."""

    def infer_topic(
        self,
        logits: np.ndarray,
        context: str,
        position: int,
        tokenizer
    ) -> str:
        """
        Extract search topic from uncertainty.

        Methods:
        1. Top-k tokens: Use top predictions as search terms
        2. Context extraction: Parse last statement
        3. Symbol extraction: Identify identifiers being referenced
        """
        # Get top-10 predicted tokens
        top_indices = np.argsort(logits)[-10:]
        top_tokens = [tokenizer.decode([idx]) for idx in top_indices]

        # Filter to code-looking tokens
        code_tokens = [t for t in top_tokens if self._is_code_like(t)]

        # Extract identifiers from recent context
        recent_context = context[-200:]  # Last 200 chars
        identifiers = self._extract_identifiers(recent_context)

        # Combine into search query
        return " ".join(code_tokens + identifiers)

    def _extract_identifiers(self, text: str) -> List[str]:
        """Extract variable/function names using regex."""
        # Match: foo.bar, baz(), module.Class, etc.
```

Tasks:
- [ ] Implement topic inference
- [ ] Test on examples with known missing context
- [ ] Validate: Does inferred topic match actual missing info?
- [ ] **Target**: >70% relevance on manual inspection

#### Day 3-5: Adaptive Retriever
**File**: `packages/python-orchestrator/orchestrator/retrieval/adaptive.py`

```python
class AdaptiveContextRetriever:
    """Retrieves context when uncertainty is detected."""

    def __init__(
        self,
        base_retriever,  # Existing RepoSynth retriever
        uncertainty_monitor: UncertaintyMonitor,
        topic_inferrer: TopicInferrer,
        max_retrievals: int = 3
    ):
        self.base_retriever = base_retriever
        self.monitor = uncertainty_monitor
        self.inferrer = topic_inferrer
        self.max_retrievals = max_retrievals
        self.retrieval_count = 0

    def retrieve_on_uncertainty(
        self,
        uncertainty_result: UncertaintyResult,
        current_context: str,
        generated_so_far: str
    ) -> Optional[str]:
        """
        Retrieve additional context when uncertain.

        Returns:
            New context to add, or None if budget exhausted
        """
        if not uncertainty_result.should_retrieve:
            return None

        if self.retrieval_count >= self.max_retrievals:
            return None

        # Infer what we're uncertain about
        topic = self.inferrer.infer_topic(
            uncertainty_result.logits,
            generated_so_far,
            uncertainty_result.position
        )

        # Retrieve relevant context
        retrieved = self.base_retriever.retrieve(
            query=topic,
            top_k=3
        )

        self.retrieval_count += 1
        return retrieved
```

Tasks:
- [ ] Implement adaptive retriever
- [ ] Integrate with existing RepoSynth retriever
- [ ] Test on 20 examples
- [ ] Track: How many retrievals per question?

### Week 7 - Generation Loop & Context Management

#### Day 1-3: Adaptive Generator
**File**: `packages/python-orchestrator/orchestrator/generation/adaptive_generator.py`

```python
class AdaptiveGenerator:
    """Generate responses with uncertainty-triggered retrieval."""

    def generate(
        self,
        query: str,
        initial_context: str,
        model,
        max_tokens: int = 512,
        max_retrievals: int = 3
    ) -> GenerationResult:
        """
        Main generation loop with adaptive retrieval.

        Algorithm:
        1. Start generating with initial context
        2. At each measurement point:
           a. Compute uncertainty (CCE)
           b. If uncertain, retrieve more context
           c. Continue generation with expanded context
        3. Return final response + metadata
        """
        context = initial_context
        generated = ""
        entropy_trace = []
        retrieval_events = []

        for i in range(max_tokens):
            # Generate next token
            logits = model.forward(context + generated)

            # Measure uncertainty
            if self.monitor.should_measure(generated, i, context):
                uncertainty = self.monitor.measure_uncertainty(logits)
                entropy_trace.append(uncertainty)

                # Retrieve if uncertain
                if uncertainty.should_retrieve:
                    new_context = self.retriever.retrieve_on_uncertainty(
                        uncertainty, context, generated
                    )
                    if new_context:
                        context += "\n" + new_context
                        retrieval_events.append({
                            "position": i,
                            "topic": self.inferrer.last_topic,
                            "context": new_context
                        })

            # Sample token
            token = sample(logits)
            generated += token

            if token == "<|endoftext|>":
                break

        return GenerationResult(
            response=generated,
            entropy_trace=entropy_trace,
            retrieval_events=retrieval_events,
            total_context_tokens=len(tokenize(context)),
            total_retrievals=len(retrieval_events)
        )
```

Tasks:
- [ ] Implement adaptive generation loop
- [ ] Handle context window limits
- [ ] Add logging/debugging
- [ ] Test end-to-end on 10 examples

#### Day 4-5: Context Manager
**File**: `packages/python-orchestrator/orchestrator/retrieval/context_manager.py`

```python
class ContextManager:
    """Manages context window during adaptive retrieval."""

    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self.segments = []  # List of (content, source, priority, tokens)
        self.current_tokens = 0

    def add_context(
        self,
        content: str,
        source: str,
        priority: int = 1
    ) -> bool:
        """
        Add context segment.

        Returns:
            True if added, False if budget exceeded
        """
        tokens = len(tokenize(content))

        # Check if we need to evict
        if self.current_tokens + tokens > self.max_tokens:
            evicted = self.evict_lowest_priority(tokens)
            if not evicted:
                return False  # Can't fit even after eviction

        self.segments.append((content, source, priority, tokens))
        self.current_tokens += tokens
        return True

    def evict_lowest_priority(self, tokens_needed: int) -> bool:
        """Remove lowest priority segments until we have space."""
        # Sort by priority (ascending)
        self.segments.sort(key=lambda x: x[2])

        freed = 0
        removed = []
        for i, (content, source, priority, tokens) in enumerate(self.segments):
            if freed >= tokens_needed:
                break
            removed.append(i)
            freed += tokens

        # Remove in reverse order
        for i in reversed(removed):
            self.segments.pop(i)

        self.current_tokens -= freed
        return freed >= tokens_needed
```

Tasks:
- [ ] Implement context manager
- [ ] Test eviction logic
- [ ] Add provenance tracking
- [ ] **Target**: Stay within 4K token budget

---

## PHASE 4: Evaluation Framework (Weeks 8-9)
**Goal**: Build benchmark and evaluation metrics

### Week 8 - Benchmark Dataset

#### Day 1-3: Dataset Curation
**File**: `research/benchmarks/dataset.json`

Dataset structure:
```json
{
  "examples": [
    {
      "id": "example_001",
      "query": "How does user authentication work in this codebase?",
      "repository": "sample-web-app",
      "ground_truth_files": [
        "src/auth/login.ts",
        "src/auth/middleware.ts",
        "src/models/user.ts"
      ],
      "ground_truth_answer": "Authentication uses JWT tokens...",
      "difficulty": "medium",
      "category": "architecture"
    }
  ]
}
```

Tasks:
- [ ] Select 3-5 open-source repos (TypeScript, Python)
- [ ] Create 100 questions:
  - 30 comprehension ("How does X work?")
  - 30 debugging ("Why does X fail?")
  - 40 architecture ("Where is X implemented?")
- [ ] Document ground truth files for each
- [ ] Write reference answers
- [ ] **Deliverable**: `benchmark_dataset_v1.json`

#### Day 4-5: Evaluation Metrics
**File**: `research/evaluation/metrics.py`

```python
# Quality Metrics

def answer_correctness_llm_judge(
    predicted: str,
    ground_truth: str,
    context: str
) -> float:
    """
    Use GPT-4 as judge to rate answer correctness [0-1].

    Prompt:
    "Rate how well the predicted answer matches the ground truth.
     Consider: correctness, completeness, accuracy.
     Return score 0.0-1.0"
    """

def answer_completeness(predicted: str, ground_truth: str) -> float:
    """Measure what fraction of ground truth is covered."""
    # Use sentence embeddings + cosine similarity
    # Or keyword overlap (precision/recall)

def hallucination_rate(predicted: str, provided_context: str) -> float:
    """
    Measure how much of the answer is not grounded in context.

    Method:
    1. Extract claims from predicted answer
    2. Check if each claim is supported by context
    3. Return fraction of unsupported claims
    """

# Efficiency Metrics

def context_precision(retrieved_files: List[str], ground_truth_files: List[str]) -> float:
    """Precision of retrieved files: |retrieved ∩ ground_truth| / |retrieved|"""

def context_recall(retrieved_files: List[str], ground_truth_files: List[str]) -> float:
    """Recall of retrieved files: |retrieved ∩ ground_truth| / |ground_truth|"""

def token_efficiency(tokens_used: int, baseline_tokens: int) -> float:
    """Relative token usage: 1 - (tokens_used / baseline_tokens)"""

# Uncertainty Metrics

def spike_precision(detected_spikes: List[int], ground_truth_missing: List[int]) -> float:
    """Precision of uncertainty detection."""

def spike_recall(detected_spikes: List[int], ground_truth_missing: List[int]) -> float:
    """Recall of uncertainty detection."""
```

Tasks:
- [ ] Implement 8 evaluation metrics
- [ ] Test metrics on sample data
- [ ] Set up GPT-4 API for LLM-as-judge
- [ ] Validate inter-rater reliability (compare GPT-4 with human ratings on 20 examples)

### Week 9 - Baselines & Experiment Runner

#### Day 1-2: Baseline Implementations
**File**: `research/baselines/`

Baselines to implement:
1. **No Context**: Answer without any retrieval
2. **Full Context**: Include maximum allowed context
3. **Random Context**: Random file selection
4. **BM25 Retrieval**: Keyword-based retrieval
5. **Embedding Retrieval**: Pure semantic similarity
6. **RepoSynth Base**: Existing RepoSynth (no entropy monitoring)
7. **UnCert-CoT**: Line boundary entropy measurement

Tasks:
- [ ] Implement 7 baseline methods
- [ ] Ensure fair comparison (same model, same token budget)
- [ ] Test each baseline on 10 examples
- [ ] **Deliverable**: Baselines ready to run

#### Day 3-5: Experiment Runner
**File**: `research/experiments/runner.py`

```python
class ExperimentRunner:
    def run_experiment(
        self,
        method_name: str,
        dataset: List[CodeQAExample],
        config: ExperimentConfig,
        output_dir: str
    ) -> ExperimentResults:
        """
        Run one experiment configuration.

        For each example:
        1. Generate answer using method
        2. Compute all evaluation metrics
        3. Log results and traces

        Returns aggregated results.
        """
        results = []

        for example in tqdm(dataset):
            # Generate answer
            if method_name == "cce":
                result = self.run_cce_method(example, config)
            elif method_name == "baseline_bm25":
                result = self.run_bm25_baseline(example, config)
            # ... other methods

            # Evaluate
            metrics = self.evaluate(result, example)

            # Save
            results.append({
                "example_id": example.id,
                "predicted_answer": result.answer,
                "metrics": metrics,
                "entropy_trace": result.entropy_trace,
                "retrieval_events": result.retrieval_events
            })

        # Aggregate
        return ExperimentResults(
            method=method_name,
            results=results,
            aggregated_metrics=self.aggregate_metrics(results)
        )

    def compare_methods(
        self,
        methods: List[str],
        dataset: List[CodeQAExample]
    ) -> ComparisonResults:
        """Run multiple methods and compare."""
```

Tasks:
- [ ] Implement experiment runner
- [ ] Add logging and checkpointing (resume interrupted runs)
- [ ] Create visualization scripts
- [ ] Test on 10 examples
- [ ] **Deliverable**: Ready to run experiments

---

## PHASE 5: Experiments (Weeks 10-11)
**Goal**: Collect experimental results

### Week 10 - Main Experiments

#### Experiment 1: CCE vs Raw Entropy (2 days)
**RQ1**: Does CCE outperform raw entropy for detecting missing code context?

```bash
# Run experiment
python research/experiments/runner.py \
  --methods raw_entropy,normalized_entropy,prob_diff,cce \
  --dataset benchmark_dataset_v1.json \
  --output results/exp1_entropy_comparison/
```

Analysis:
- [ ] Compare spike detection precision/recall
- [ ] Compute F1 scores
- [ ] Statistical significance test (paired t-test)
- [ ] **Target**: CCE F1 > raw entropy F1 by >10%

#### Experiment 2: Adaptive Retrieval Effectiveness (2 days)
**RQ2**: Does adaptive retrieval improve answer quality?

```bash
python research/experiments/runner.py \
  --methods no_context,full_context,static_retrieval,adaptive_cce \
  --dataset benchmark_dataset_v1.json \
  --output results/exp2_adaptive_retrieval/
```

Analysis:
- [ ] Compare answer correctness
- [ ] Compare token efficiency
- [ ] Plot quality vs token usage tradeoff
- [ ] **Target**: Same quality with 30-40% fewer tokens

#### Experiment 3: Measurement Strategies (1 day)
**RQ3**: Where should entropy be measured?

```bash
python research/experiments/runner.py \
  --methods cce_every_token,cce_line_boundary,cce_semantic_boundary \
  --dataset benchmark_dataset_v1.json \
  --output results/exp3_measurement_strategies/
```

Analysis:
- [ ] Compare latency overhead
- [ ] Compare retrieval precision
- [ ] Compare answer quality
- [ ] **Target**: Semantic boundaries = best tradeoff

### Week 11 - Additional Experiments

#### Experiment 4: Threshold Sensitivity (1 day)
**RQ4**: What are optimal thresholds?

```bash
for threshold in 0.1 0.2 0.3 0.4 0.5; do
  python research/experiments/runner.py \
    --method cce \
    --threshold $threshold \
    --dataset benchmark_dataset_v1.json \
    --output results/exp4_thresholds/threshold_$threshold/
done
```

Analysis:
- [ ] Plot precision-recall curves
- [ ] Find optimal F1 operating point
- [ ] Analyze false positive/negative tradeoffs

#### Experiment 5: Ablation Study (2 days)
**Question**: Which components contribute most?

Ablations:
1. CCE without code token filtering
2. CCE without language token filtering
3. CCE with different token taxonomies
4. CCE with different normalizations

Tasks:
- [ ] Implement 4 ablations
- [ ] Run on full dataset
- [ ] Measure performance degradation
- [ ] **Deliverable**: Ablation results table

#### Statistical Analysis (1 day)
**File**: `research/analysis/statistics.py`

Tasks:
- [ ] Run paired t-tests (CCE vs baselines)
- [ ] Compute effect sizes (Cohen's d)
- [ ] Calculate 95% confidence intervals
- [ ] Test for statistical significance (p < 0.05)
- [ ] **Deliverable**: `results/statistical_analysis.json`

---

## PHASE 6: Paper Writing (Weeks 12-13)
**Goal**: Write research paper

### Week 12 - Draft Sections

#### Day 1-2: Method Section
**Section 4: Method (2.5 pages)**

Outline:
- 4.1 Problem Formulation
  - Define code Q&A task
  - Formalize uncertainty detection problem
- 4.2 Contrastive Code Entropy
  - Algorithm description
  - Mathematical formulation
  - Intuition and examples
- 4.3 Token Classification
  - Code vs language taxonomy
  - Implementation details
- 4.4 Adaptive Context Retrieval
  - Retrieval triggering
  - Topic inference
  - Context management
- 4.5 Implementation
  - System architecture
  - Integration with RepoSynth

Tasks:
- [ ] Write algorithm pseudocode
- [ ] Create system architecture diagram (Figure 1)
- [ ] Write method description (2000 words)
- [ ] Add mathematical notation

#### Day 3-4: Experiments Section
**Section 5: Experiments (2 pages)**

Outline:
- 5.1 Experimental Setup
  - Dataset description
  - Baselines
  - Evaluation metrics
  - Hyperparameters
- 5.2 RQ1: CCE vs Raw Entropy
- 5.3 RQ2: Adaptive Retrieval
- 5.4 RQ3: Measurement Strategies
- 5.5 Ablation Studies

Tasks:
- [ ] Write experimental setup
- [ ] Describe each experiment
- [ ] Create results tables
- [ ] **Deliverable**: Experiments section draft

#### Day 5: Results Section
**Section 6: Results and Discussion (1.5 pages)**

Outline:
- Main findings summary
- Answer to each RQ
- Qualitative analysis
- Failure cases
- Limitations

Tasks:
- [ ] Write results summary
- [ ] Select 2-3 qualitative examples
- [ ] Discuss limitations honestly
- [ ] **Deliverable**: Results section draft

### Week 13 - Complete Draft

#### Day 1: Introduction & Related Work
**Section 1: Introduction (1.5 pages)**
- Problem motivation
- Limitations of existing work
- Our contributions (4 bullet points)
- Results summary

**Section 2: Related Work (1 page)**
- Uncertainty in LLMs
- Code retrieval and RAG
- Chain-of-thought reasoning

Tasks:
- [ ] Write introduction
- [ ] Write related work
- [ ] Position our work clearly

#### Day 2: Abstract & Conclusion
**Abstract (200 words)**
- Problem (1-2 sentences)
- Method (2-3 sentences)
- Results (2-3 sentences)

**Section 7: Conclusion (0.5 pages)**
- Summary of contributions
- Impact and future work

Tasks:
- [ ] Write abstract
- [ ] Write conclusion
- [ ] Ensure consistency across paper

#### Day 3-4: Figures & Tables
Create all figures:
- [ ] **Figure 1**: System architecture
- [ ] **Figure 2**: CCE visualization (key figure!)
- [ ] **Figure 3**: Entropy trace with retrievals
- [ ] **Figure 4**: Precision-recall curves
- [ ] **Figure 5**: Quality vs token usage
- [ ] **Table 1**: Main results comparison
- [ ] **Table 2**: Ablation study

#### Day 5: Revision
- [ ] Read entire paper for flow
- [ ] Check math notation consistency
- [ ] Proofread for typos
- [ ] Ensure all claims are supported by results
- [ ] **Deliverable**: Complete draft v1

---

## PHASE 7: Submission Preparation (Week 14)
**Goal**: Prepare for submission

### Target Venues
1. **ICSE 2026** (International Conference on Software Engineering)
   - Research track
   - Deadline: ~August 2025
   - 11 pages + references

2. **FSE 2026** (Foundations of Software Engineering)
   - Research track
   - Deadline: ~February 2025
   - 12 pages + references

3. **ACL 2026** (Association for Computational Linguistics)
   - Main conference
   - Deadline: ~January 2026
   - 8 pages + references

### Week 14 - Submission

#### Day 1-2: Format Paper
- [ ] Choose target venue
- [ ] Download LaTeX template
- [ ] Reformat paper to venue requirements
- [ ] Check page limits

#### Day 3: Supplementary Material
Create supplementary PDF with:
- [ ] Extended results tables
- [ ] Additional ablations
- [ ] Hyperparameter details
- [ ] Qualitative examples
- [ ] Error analysis

#### Day 4: Code & Artifact
Prepare code release:
- [ ] Clean repository
- [ ] Write comprehensive README
- [ ] Add installation instructions
- [ ] Create reproduction scripts
- [ ] Add license (MIT or Apache 2.0)
- [ ] Upload to GitHub
- [ ] Create Zenodo archive for DOI

#### Day 5: Submit
- [ ] Final proofread
- [ ] Check all co-authors approved
- [ ] Anonymous submission (if required)
- [ ] Submit to venue
- [ ] Celebrate! 🎉

---

## PHASE 8: Contingency & Iteration (Week 15)
**Buffer week for unexpected issues**

### Potential Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| CCE doesn't outperform baselines | Medium | Pivot to analysis paper: "When does entropy fail?" |
| Compute limitations | Low | Use smaller models (CodeLlama-7B only) |
| Benchmark too small | Medium | Reduce to 50 examples, deeper analysis |
| Results not significant | Medium | Focus on qualitative insights, user study |
| Time constraints | High | Prioritize Exp 1-2, defer ablations |

### Backup Plans
1. **If CCE fails**: Analyze failure modes, contribute diagnostic framework
2. **If compute limited**: Focus on efficiency analysis, smaller scale
3. **If results marginal**: Add human evaluation, user study
4. **If timeline slips**: Submit to later venue (EMNLP, NeurIPS)

---

## Success Criteria

### Technical Milestones
- [ ] CCE achieves >10% improvement in F1 vs raw entropy (Exp 1)
- [ ] Adaptive retrieval uses <70% tokens with ≥95% quality (Exp 2)
- [ ] System runs in <2x latency overhead (Exp 3)
- [ ] Results statistically significant (p < 0.05)

### Publication Milestones
- [ ] Complete paper draft (8-12 pages)
- [ ] All experiments complete with positive results
- [ ] Code released on GitHub
- [ ] Paper submitted to top-tier venue

### Artifact Milestones
- [ ] Open-source release with documentation
- [ ] Reproduction scripts for all experiments
- [ ] Benchmark dataset publicly available
- [ ] >100 GitHub stars within 6 months (aspirational)

---

## Resource Checklist

### Compute
- [ ] Access to GPU (A100 40GB or 2x RTX 3090)
- [ ] Estimated 100-200 GPU hours
- [ ] Cloud credits (AWS, GCP) or local cluster

### Models
- [ ] CodeLlama-7B-Instruct (✓ can download)
- [ ] CodeLlama-13B-Instruct (✓ can download)
- [ ] GPT-4 API access (for LLM-as-judge)

### Data
- [ ] 3-5 open-source repos for benchmarks
- [ ] Existing benchmarks for reference (HumanEval, MBPP)

### Tools
- [ ] Python 3.10+
- [ ] PyTorch 2.0+
- [ ] Transformers library
- [ ] Tree-sitter (AST parsing)
- [ ] Sentence-transformers (embeddings)

---

## Timeline Gantt Chart

```
Week 1:  [Foundation & POC          ]
Week 2:  [                Entropy Implementation                    ]
Week 3:  [                Entropy Implementation                    ]
Week 4:  [                           Monitoring System              ]
Week 5:  [                           Monitoring System              ]
Week 6:  [                                       Adaptive Retrieval ]
Week 7:  [                                       Adaptive Retrieval ]
Week 8:  [                                             Evaluation   ]
Week 9:  [                                             Evaluation   ]
Week 10: [                                                Experiments]
Week 11: [                                                Experiments]
Week 12: [                                                      Paper]
Week 13: [                                                      Paper]
Week 14: [                                                 Submission]
Week 15: [Buffer/Contingency                                        ]
```

---

## Next Steps (Start Now!)

### Immediate Actions (This Week)
1. [ ] **Read ARPO paper** - Understand entropy-based uncertainty detection
2. [ ] **Read UnCert-CoT paper** - Understand measurement strategies
3. [ ] **Set up development environment**:
   ```bash
   # Create research branch
   git checkout -b research/cce-implementation

   # Install dependencies
   pip install torch transformers tree-sitter sentence-transformers

   # Download CodeLlama-7B
   huggingface-cli download codellama/CodeLlama-7b-Instruct-hf
   ```
4. [ ] **Run POC experiment** - Validate core hypothesis (5 examples by hand)
5. [ ] **Document research questions** - Write `research/research_questions.md`

### Week 2 Start
- [ ] Begin Phase 1: Implement `entropy/calculator.py`
- [ ] Set up testing framework
- [ ] Create project structure:
  ```
  packages/python-orchestrator/orchestrator/
  ├── entropy/
  │   ├── __init__.py
  │   ├── calculator.py
  │   ├── token_classifier.py
  │   └── cce.py
  ├── retrieval/
  │   ├── adaptive.py
  │   └── topic_inference.py
  └── generation/
      └── adaptive_generator.py
  ```

---

## Questions to Resolve

Before starting, clarify:
1. **Compute access**: Do we have GPU access? Which ones?
2. **Co-authors**: Who else is involved? Roles?
3. **Target venue**: ICSE vs ACL vs NeurIPS? (Different focuses)
4. **Evaluation priority**: Automatic metrics vs human evaluation?
5. **Code release timing**: With submission or after acceptance?

---

**Document Version**: 1.0
**Created**: December 2024
**Status**: Ready to start Phase 0
**Next Milestone**: POC validation (Week 1, Day 4-5)
