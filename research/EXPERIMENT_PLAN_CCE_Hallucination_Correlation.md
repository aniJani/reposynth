# Experiment Plan: CCE Spikes Predict Hallucination

## Research Goal

**Prove that CCE (Code-Conditional Entropy) spikes during LLM generation predict hallucination risk, and that retrieval at those moments reduces errors.**

---

## Current Research Summary (Weeks 8-12)

### What Was Tested (Wrong Focus)

| Week | Focus | Result |
|------|-------|--------|
| Week 8 | Comprehensive evaluation framework | Baseline F1=0.62 |
| Week 9 | CCE ablation - does CCE help retrieval? | Inconclusive |
| Week 10 | Learned Query Pooler architecture | Cross-attention module |
| Week 11 | Learned vs Heuristic retrieval accuracy | Learned +64% on synthetic data |
| Week 12 | Per-repo fine-tuning | F1=0.56 (worse than baseline 0.62) |

**Problem:** All experiments tested "Can CCE spikes tell us WHICH files to retrieve?"
- Result: No - heuristic keyword matching works better

### What Should Be Tested (Your Actual Goal)

**Hypothesis:** CCE spikes predict WHEN the model will make errors (hallucinate)

This is fundamentally different:
- NOT about file selection
- ABOUT timing retrieval
- ABOUT error prediction

---

## The Right Experiment Design

### Phase 1: Collect CCE-Error Correlation Data

```
For each query in benchmark:
    1. Generate answer WITHOUT retrieval
    2. Record CCE at every token position
    3. Record the actual tokens generated
    4. Evaluate answer correctness (automated + human)
    5. Mark error regions in the generated text
```

**Output:** Dataset with (token_position, CCE_value, is_error) tuples

### Phase 2: Statistical Analysis

1. **Correlation Test**
   - Compute Pearson/Spearman correlation: CCE vs error probability
   - Hypothesis: High CCE → High error probability

2. **Threshold Analysis**
   - For CCE thresholds [2.0, 2.5, 3.0, 3.5, 4.0]:
     - Precision: P(error | CCE > threshold)
     - Recall: P(CCE > threshold | error)
   - Find optimal threshold

3. **Temporal Analysis**
   - Do errors occur AT the CCE spike or AFTER?
   - Window analysis: errors within N tokens of spike

### Phase 3: Retrieval Intervention Study

```
For each query:
    Condition A (Control): Generate without retrieval
    Condition B (CCE-triggered): Retrieve when CCE > threshold
    Condition C (Always retrieve): Retrieve before every response
    Condition D (Random): Retrieve at random intervals
```

**Key Metrics:**
- Answer correctness (0-1 scale)
- Hallucination rate (factual errors / total claims)
- Retrieval efficiency (correct answers / retrieval calls)

### Phase 4: Ablation Studies

1. **CCE Threshold Ablation**
   - Does threshold affect the correlation?

2. **Retrieval Quality Ablation**
   - Even with perfect retrieval (oracle), does CCE-triggered beat always-retrieve?

3. **Model Size Ablation**
   - Does correlation hold for different LLM sizes?

---

## Concrete Implementation

### Benchmark Requirements

Need queries where we can objectively evaluate correctness:

1. **Factual Code Questions** (objective ground truth)
   - "What is the default timeout in httpx?" → Check source code
   - "What exception does X raise?" → Verify in code
   - "What parameters does function Y accept?" → Check signature

2. **Code Generation** (executable verification)
   - "Write code to do X" → Run and test output
   - Errors = code doesn't work or produces wrong output

3. **Explanation Questions** (semantic similarity)
   - "Explain how X works" → Compare to reference explanation
   - Errors = contradicts source code or makes up features

### Error Detection Methods

1. **Factual Verification**
   - Extract claims from generated text
   - Verify each claim against source code
   - Error = claim contradicts source

2. **Semantic Drift**
   - Compute embedding similarity to relevant source files
   - Large drop = likely hallucination

3. **Contradiction Detection**
   - Check for internal contradictions in response
   - Check for contradictions with prompt/context

4. **Code Execution** (for code generation)
   - Run generated code
   - Compare output to expected

### CCE Computation

```python
def compute_cce_trace(model, tokenizer, prompt, max_tokens=200):
    """Generate text and record CCE at each position."""
    cce_trace = []
    tokens_generated = []

    input_ids = tokenizer.encode(prompt)

    for step in range(max_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]

            # Compute CCE (entropy of distribution)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            cce_trace.append(entropy.item())

            # Sample next token
            next_token = torch.argmax(logits)
            tokens_generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)])

            if next_token == tokenizer.eos_token_id:
                break

    return {
        'tokens': tokens_generated,
        'text': tokenizer.decode(tokens_generated),
        'cce_trace': cce_trace,
    }
```

### Error Annotation

```python
def annotate_errors(generated_text, ground_truth, source_files):
    """Mark which tokens/regions contain errors."""

    # 1. Extract claims from generated text
    claims = extract_claims(generated_text)

    # 2. Verify each claim
    error_spans = []
    for claim in claims:
        if not verify_claim(claim, source_files, ground_truth):
            error_spans.append(claim['span'])

    # 3. Map spans to token positions
    error_tokens = spans_to_tokens(error_spans, generated_text)

    return error_tokens  # List of token indices that are errors
```

---

## Expected Results

### If Hypothesis is Correct:

```
CCE Distribution:
    At error tokens:     mean=4.2, std=1.1
    At correct tokens:   mean=2.8, std=0.9

Correlation: r=0.67, p<0.001

Threshold Analysis (CCE > 3.0):
    Precision: 0.72 (72% of spikes precede errors)
    Recall:    0.58 (58% of errors have preceding spike)

Intervention Study:
    | Condition          | Correctness | Hallucination Rate |
    |--------------------|-------------|-------------------|
    | No retrieval       | 0.55        | 0.35              |
    | CCE-triggered      | 0.78        | 0.12              |
    | Always retrieve    | 0.72        | 0.18              |
    | Random retrieval   | 0.62        | 0.28              |
```

### This Would Prove:

1. **CCE spikes ARE predictive** - statistically significant correlation
2. **CCE-triggered retrieval works** - higher accuracy than no retrieval
3. **CCE is EFFICIENT** - better than always/random retrieval (fewer calls, same/better accuracy)

---

## Files to Create

1. **`Week13_CCE_Error_Correlation.ipynb`**
   - Phase 1 & 2: Data collection and statistical analysis
   - Focus: Prove the correlation exists

2. **`Week14_Retrieval_Intervention.ipynb`**
   - Phase 3: Compare retrieval strategies
   - Focus: Prove CCE-triggered retrieval helps

3. **`cce_correlation/`** module
   - `error_annotator.py` - Detect errors in generated text
   - `claim_extractor.py` - Extract verifiable claims
   - `correlation_analyzer.py` - Statistical analysis

---

## Key Differences from Previous Approach

| Aspect | Previous (Weeks 8-12) | New Approach |
|--------|----------------------|--------------|
| Goal | Which files to retrieve | When to retrieve |
| Metric | File retrieval F1 | Error rate reduction |
| Training | Learn file selector | No training needed |
| Validation | Match ground truth files | Reduce hallucinations |
| Complexity | Learned Query Pooler | Simple threshold |

---

## Why This Will Work

1. **Simpler hypothesis** - Just correlation, no complex learning
2. **Objective metrics** - Error rate is measurable
3. **Matches intuition** - High entropy = model uncertain = more likely wrong
4. **Useful result** - Even if correlation is moderate, it's still valuable

---

## Minimum Viable Experiment

If time-constrained, run this minimal version:

1. **20 factual questions** about httpx (objective answers)
2. **Generate without retrieval**, record CCE
3. **Manually annotate errors** in responses
4. **Compute correlation** between CCE and error positions
5. **Re-generate with CCE-triggered retrieval**
6. **Compare error rates**

This can be done in a single Colab session (~2-3 hours).
