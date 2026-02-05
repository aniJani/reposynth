# Research Plan: CodeGnosis - Adapting Self-Awareness for Code Generation

## Executive Summary

Adapt the Gnosis self-awareness mechanism for **code generation tasks**, specifically for repository-aware code completion. Our key innovation: combine Gnosis-style hidden state analysis with **code-specific signals** and **retrieval augmentation**.

---

## What Gnosis Does (Baseline)

```
Input: LLM hidden states + attention maps during generation
Output: Scalar correctness probability (0-1)
Training: Supervised on (generation, is_correct) pairs
```

**Limitation**: Gnosis only PREDICTS errors, doesn't FIX them.

---

## Our Innovation: CodeGnosis

### Core Idea

```
CodeGnosis = Gnosis + Code Signals + Retrieval Intervention

When CodeGnosis predicts high error probability:
  1. Pause generation
  2. Retrieve relevant source code from repository
  3. Inject context and continue/regenerate
```

### Why This is Better for Code

| Aspect | Original Gnosis | CodeGnosis |
|--------|-----------------|------------|
| Domain | Math, QA, MMLU | Code generation |
| Error types | Binary (correct/wrong) | Multi-class (syntax, runtime, API, logic) |
| Verification | Answer matching | Execution + AST + API validation |
| Action on error | Just predict | Retrieve + fix |
| Context | None | Repository-aware retrieval |

---

## Architecture Design

### 1. Multi-Signal Feature Extraction

```
                    ┌─────────────────────────────────────┐
                    │         LLM Backbone (frozen)        │
                    └─────────────────────────────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
    ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
    │ Hidden States │        │ Attention Maps │        │ Output Logits │
    │   Encoder     │        │    Encoder     │        │   (CCE)       │
    └───────────────┘        └───────────────┘        └───────────────┘
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      ▼
                            ┌─────────────────┐
                            │  Feature Fusion │
                            │   (Gated MLP)   │
                            └─────────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
    ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
    │  Correctness  │        │  Error Type   │        │   Retrieval   │
    │    Score      │        │  Classifier   │        │   Trigger     │
    └───────────────┘        └───────────────┘        └───────────────┘
```

### 2. Code-Specific Signals (Beyond Hidden States)

| Signal | How to Compute | What it Indicates |
|--------|----------------|-------------------|
| Syntax Validity | Incremental AST parsing | Malformed code |
| API Token Attention | Attention to API-related tokens | Model focusing on API calls |
| Import Consistency | Check if used APIs are imported | Missing imports |
| Type Flow | Track type annotations | Type mismatches |
| Docstring Alignment | Similarity to function docstrings | Following spec |

### 3. Error Type Classification (Multi-Label)

Instead of binary correct/incorrect:

```python
error_types = {
    'syntax_error': bool,      # Code doesn't parse
    'import_error': bool,      # Missing/wrong imports
    'api_misuse': bool,        # Wrong method/params for library
    'type_error': bool,        # Type mismatches
    'runtime_error': bool,     # Crashes during execution
    'logic_error': bool,       # Runs but wrong output
    'hallucination': bool,     # Uses non-existent APIs
}
```

### 4. Retrieval-Augmented Correction

```python
def generate_with_codegnosis(prompt, model, retriever):
    while generating:
        # Generate next token
        hidden_states, attention, logits = model.forward(...)

        # CodeGnosis prediction
        error_prob, error_type, should_retrieve = codegnosis(
            hidden_states, attention, logits
        )

        if should_retrieve:
            # Get relevant source code
            context = retriever.search(
                query=partial_generation,
                error_type=error_type  # Targeted retrieval
            )

            # Inject context and continue
            prompt = inject_context(prompt, context)
            continue_generation()
```

---

## Training Pipeline

### Phase 1: Data Collection

Generate code completions and label them:

```python
training_data = []

for task in code_tasks:
    # Generate without retrieval
    generation, hidden_states, attention = model.generate_with_internals(task.prompt)

    # Multi-label verification
    labels = {
        'syntax_error': check_syntax(generation),
        'runtime_error': check_execution(generation),
        'api_misuse': check_api_usage(generation, repo_apis),
        'hallucination': check_api_exists(generation, repo_apis),
        'correct': run_tests(generation, task.tests),
    }

    training_data.append({
        'hidden_states': hidden_states,
        'attention': attention,
        'labels': labels,
    })
```

### Phase 2: Train CodeGnosis Head

```python
class CodeGnosisHead(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_error_types):
        # Hidden state encoder
        self.hs_encoder = nn.TransformerEncoder(...)

        # Attention pattern encoder
        self.attn_encoder = nn.Conv1d(...)

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Output heads
        self.correctness_head = nn.Linear(hidden_dim // 2, 1)
        self.error_type_head = nn.Linear(hidden_dim // 2, num_error_types)
        self.retrieval_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, hidden_states, attention_maps):
        hs_features = self.hs_encoder(hidden_states)
        attn_features = self.attn_encoder(attention_maps)

        fused = self.fusion(torch.cat([hs_features, attn_features], dim=-1))

        correctness = torch.sigmoid(self.correctness_head(fused))
        error_types = torch.sigmoid(self.error_type_head(fused))
        should_retrieve = torch.sigmoid(self.retrieval_head(fused))

        return correctness, error_types, should_retrieve
```

### Phase 3: Train Retrieval Trigger

Key insight: Train when retrieval HELPS, not just when errors occur.

```python
# For each training sample, also record:
# - Did retrieval improve the output?
# - Which error types does retrieval fix?

retrieval_helps_data = []

for sample in training_data:
    if sample['labels']['correct']:
        retrieval_helps_data.append((sample, retrieval_helped=False))
    else:
        # Re-generate with retrieval
        new_generation = generate_with_retrieval(sample['prompt'])
        retrieval_helped = is_correct(new_generation)
        retrieval_helps_data.append((sample, retrieval_helped))
```

---

## Improvements Over Original Gnosis

### 1. Code-Specific Error Taxonomy

Original Gnosis: Binary correct/incorrect
CodeGnosis: 7+ error types with different retrieval strategies

```python
retrieval_strategy = {
    'syntax_error': retrieve_similar_code_patterns,
    'api_misuse': retrieve_api_documentation,
    'hallucination': retrieve_actual_api_signatures,
    'type_error': retrieve_type_definitions,
    'logic_error': retrieve_test_cases_and_examples,
}
```

### 2. Partial Generation Detection

Gnosis waits for ~40% generation. We can detect earlier using code structure:

```python
def early_error_detection(partial_code):
    # Detect unclosed brackets - likely syntax error coming
    if count_open_brackets(partial_code) > 3:
        return high_error_probability

    # Detect unknown API call starting
    if starts_unknown_api_call(partial_code, known_apis):
        return high_hallucination_probability
```

### 3. Repository-Specific Fine-Tuning

Train CodeGnosis on specific repositories:

```python
# httpx-specific CodeGnosis
httpx_codegnosis = train_codegnosis(
    base_model=codellama,
    repo=httpx_source,
    api_signatures=extract_httpx_apis(),
    common_patterns=extract_httpx_patterns(),
)

# Can detect httpx-specific errors:
# - Using requests-style API with httpx
# - Forgetting async/await with AsyncClient
# - Wrong timeout parameter format
```

### 4. Confidence-Weighted Retrieval

Don't just retrieve on any error signal - weight by confidence:

```python
def smart_retrieval_decision(error_prob, error_types, generation_progress):
    # Early in generation: be more aggressive
    if generation_progress < 0.3:
        threshold = 0.3
    else:
        threshold = 0.6

    # Weight by error type severity
    severity_weights = {
        'hallucination': 1.0,  # Definitely retrieve
        'api_misuse': 0.8,
        'syntax_error': 0.5,   # Might self-correct
    }

    weighted_prob = sum(
        error_types[e] * severity_weights.get(e, 0.5)
        for e in error_types
    )

    return weighted_prob > threshold
```

---

## Evaluation Plan

### Metrics

1. **Error Prediction Accuracy**
   - Precision/Recall for each error type
   - AUROC for correctness prediction
   - Calibration (predicted probability vs actual)

2. **Retrieval Efficiency**
   - Retrievals triggered / total generations
   - Useful retrievals (led to correction) / total retrievals

3. **End-to-End Accuracy**
   - Code correctness with CodeGnosis vs without
   - Comparison to always-retrieve baseline

### Benchmarks

1. **httpx tasks** (our current focus)
   - API usage tasks
   - Error handling tasks
   - Async code tasks

2. **General code generation**
   - HumanEval
   - MBPP
   - CodeContests (harder)

3. **Repository-specific**
   - Tasks requiring specific library knowledge
   - Multi-file context tasks

---

## Implementation Phases

### Phase 1: Proof of Concept (Week 14)
- Extract hidden states from CodeLlama during generation
- Train simple MLP to predict syntax errors
- Compare to CCE baseline

### Phase 2: Full CodeGnosis (Week 15-16)
- Implement multi-signal architecture
- Train on multiple error types
- Add retrieval trigger

### Phase 3: Retrieval Integration (Week 17)
- Connect CodeGnosis to retrieval system
- End-to-end evaluation
- Compare to baselines

### Phase 4: Repository Fine-Tuning (Week 18)
- Fine-tune on httpx specifically
- Evaluate on httpx-specific tasks
- Publish results

---

## Why This Advances the Field

1. **First code-specific self-awareness mechanism**
   - Gnosis tested on math/QA, not code
   - Code has unique verifiability (execution)

2. **Actionable predictions**
   - Gnosis predicts, CodeGnosis predicts AND fixes
   - Retrieval integration is novel

3. **Multi-label error taxonomy**
   - More useful than binary correct/incorrect
   - Enables targeted interventions

4. **Repository-aware**
   - Adapts to specific codebases
   - Practical for real-world use

---

## References

- [Gnosis Paper](https://arxiv.org/abs/2512.20578) - Base self-awareness approach
- [Semantic Entropy (Nature 2024)](https://www.nature.com/articles/s41586-024-07421-0) - Alternative uncertainty estimation
- [SWE-bench](https://www.swebench.com/) - Repository-level code tasks
