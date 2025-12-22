# Research Questions and Hypotheses
## Contrastive Code Entropy (CCE) Research

**Document Version**: 1.0
**Created**: December 2024
**Status**: Pre-implementation formalization

---

## Research Objective

**Primary Goal**: Develop and evaluate Contrastive Code Entropy (CCE), a novel uncertainty metric that distinguishes between code knowledge uncertainty and language uncertainty to enable adaptive context retrieval during LLM code generation.

---

## Research Questions

### RQ1: Can entropy-based uncertainty detect missing code context?

**Question**: Can entropy measurements during LLM code generation reliably identify when the model lacks necessary code context (e.g., missing API documentation, unfamiliar libraries)?

**Motivation**:
- ARPO showed entropy spikes after tool use (external knowledge gaps)
- UnCert-CoT showed entropy varies with task difficulty
- **Unknown**: Does this apply to code generation during Q&A?

**Metrics**:
- **Spike detection precision**: % of detected spikes that actually indicate missing context
- **Spike detection recall**: % of actual missing context cases that trigger spikes
- **F1 score**: Harmonic mean of precision and recall

**Success Criteria**:
- Precision ≥ 0.70 (70% of spikes are true positives)
- Recall ≥ 0.60 (60% of missing context is detected)
- F1 ≥ 0.65

**Baseline**: Random spike detection (expected F1 ≈ 0.3-0.4)

---

### RQ2: Does Contrastive Code Entropy outperform raw entropy?

**Question**: Does separating entropy over code tokens vs language tokens improve detection of missing code context compared to standard Shannon entropy?

**Motivation**:
- Generic entropy conflates two cases:
  1. High code entropy (missing API knowledge) → **should retrieve**
  2. High language entropy (word choice) → **should NOT retrieve**
- CCE aims to separate these cases

**Comparison Methods**:
1. **Raw entropy**: H = -Σ p(x) log p(x)
2. **Normalized entropy**: H / log(V)
3. **Probability differential**: 1 - max(P)
4. **CCE (ours)**: H_code - H_language

**Metrics**:
- Spike detection F1 (primary)
- False positive rate (retrieving when not needed)
- False negative rate (missing needed context)

**Success Criteria**:
- CCE F1 > raw entropy F1 by ≥ 10 percentage points
- Statistical significance (p < 0.05, paired t-test)

**Hypothesis**: CCE achieves higher precision by filtering out language uncertainty.

---

### RQ3: Does adaptive retrieval improve quality and efficiency?

**Question**: Does uncertainty-triggered adaptive retrieval improve answer quality while reducing token usage compared to static pre-retrieval?

**Motivation**:
- ARPO achieved 50% fewer tool calls with adaptive approach
- Production systems (Cursor, Cody) do static pre-retrieval
- **Unknown**: Can we achieve efficiency gains in code Q&A?

**Comparison Baselines**:
1. **No context**: Answer without any retrieval
2. **Full context**: Include maximum allowed context (up to limit)
3. **Static retrieval**: RepoSynth base (pre-retrieval, no entropy)
4. **Random retrieval**: Retrieve random files when "uncertain"
5. **Adaptive CCE (ours)**: Retrieve when CCE > threshold

**Metrics**:
- **Answer correctness**: LLM-as-judge score [0-1]
- **Token efficiency**: (baseline_tokens - used_tokens) / baseline_tokens
- **Context precision**: Relevant files / Retrieved files
- **Context recall**: Retrieved files / Ground truth files

**Success Criteria**:
- Answer correctness ≥ 95% of full context baseline
- Token usage ≤ 70% of full context baseline
- **Efficiency gain**: Same quality with 30%+ fewer tokens

**Hypothesis**: Adaptive retrieval matches quality with fewer tokens by retrieving only when needed.

---

### RQ4: Where should entropy be measured in code generation?

**Question**: What is the optimal measurement strategy for detecting uncertainty during code generation?

**Motivation**:
- UnCert-CoT measures at line boundaries
- Measuring every token has high overhead
- Code has semantic structure (function calls, imports, etc.)

**Measurement Strategies**:
1. **Every token**: Measure at each generation step
2. **Every N tokens**: Measure every 10 tokens
3. **Line boundaries**: Measure at '\n' (UnCert-CoT style)
4. **Semantic boundaries (ours)**: Measure at:
   - Function calls: `foo(`
   - Imports: `import`, `from`
   - Assignments: `var = `
   - Method access: `object.`
   - Type annotations: `: Type`

**Metrics**:
- **Latency overhead**: % increase in generation time
- **Detection accuracy**: F1 for spike detection
- **Measurement frequency**: % of tokens measured

**Success Criteria**:
- Latency overhead < 20%
- Detection F1 ≥ 90% of "every token" baseline
- Measure < 30% of tokens

**Hypothesis**: Semantic boundaries achieve comparable detection with lower overhead.

---

## Hypotheses

### H1: Entropy spikes indicate missing code context

**Formal Hypothesis**:
> When an LLM generates code while lacking necessary context (APIs, libraries, examples), the entropy H(P) of the next-token distribution will exceed a threshold τ with precision P ≥ 0.7 and recall R ≥ 0.6.

**Operationalization**:
- **Context gap**: Ground truth files NOT in initial context
- **Entropy spike**: H(P) > τ (empirically determined)
- **Precision**: P(context_gap | spike) ≥ 0.7
- **Recall**: P(spike | context_gap) ≥ 0.6

**Testing Plan**:
- 100 code Q&A examples with known ground truth context
- Measure entropy at semantic boundaries
- Vary threshold τ ∈ {0.1, 0.2, 0.3, 0.4, 0.5}
- Plot precision-recall curve, find optimal operating point

**Null Hypothesis** (H1₀): Entropy spikes are not correlated with missing context (F1 ≤ 0.4, no better than random).

**Alternative Hypothesis** (H1₁): Entropy spikes correlate with missing context (F1 ≥ 0.65).

**Statistical Test**: Chi-squared test for independence (spike vs context gap).

---

### H2: CCE outperforms raw entropy by 10%+

**Formal Hypothesis**:
> Contrastive Code Entropy (CCE = H_code - H_language) achieves F1 score at least 10 percentage points higher than raw Shannon entropy for detecting missing code context.

**Operationalization**:
- **CCE**: Compute entropy over code tokens and language tokens separately, take difference
- **Raw entropy**: Standard H = -Σ p(x) log p(x) over all tokens
- **Improvement**: F1(CCE) - F1(raw) ≥ 0.10

**Testing Plan**:
- Same 100 examples as H1
- Compute both metrics at same measurement points
- Compare spike detection performance
- Paired t-test for significance (p < 0.05)

**Null Hypothesis** (H2₀): CCE performs no better than raw entropy (F1(CCE) ≤ F1(raw)).

**Alternative Hypothesis** (H2₁): CCE significantly outperforms raw entropy (F1(CCE) > F1(raw) + 0.10).

**Statistical Test**: Paired t-test on F1 scores across examples.

---

### H3: Adaptive retrieval achieves 30% token savings

**Formal Hypothesis**:
> Uncertainty-triggered adaptive retrieval achieves ≥95% of the answer quality of full-context retrieval while using ≤70% of the tokens.

**Operationalization**:
- **Answer quality**: LLM-as-judge correctness score [0-1]
- **Full context**: Include all ground truth files (baseline)
- **Adaptive CCE**: Retrieve only when CCE > threshold
- **Quality preservation**: score(adaptive) ≥ 0.95 × score(full)
- **Token savings**: tokens(adaptive) ≤ 0.70 × tokens(full)

**Testing Plan**:
- 100 code Q&A examples
- Measure answer quality (GPT-4 as judge)
- Measure tokens used
- Compare adaptive vs full context
- Statistical significance via paired t-test

**Null Hypothesis** (H3₀): Adaptive retrieval does not save tokens without quality loss (either quality < 95% or tokens > 70%).

**Alternative Hypothesis** (H3₁): Adaptive retrieval achieves both quality ≥ 95% and tokens ≤ 70%.

**Statistical Test**: Two-sided paired t-test for quality, one-sided for tokens.

---

### H4: Semantic boundaries are most efficient

**Formal Hypothesis**:
> Measuring entropy at semantic boundaries (function calls, imports, etc.) achieves ≥90% of the spike detection F1 of every-token measurement while measuring <30% of tokens.

**Operationalization**:
- **Every token baseline**: Measure entropy at each token
- **Semantic boundaries**: Measure only at code syntax points
- **F1 preservation**: F1(semantic) ≥ 0.90 × F1(every_token)
- **Efficiency**: measurements(semantic) < 0.30 × total_tokens

**Testing Plan**:
- 50 examples (subset for efficiency)
- Implement 4 measurement strategies
- Measure F1, latency, measurement frequency
- Analyze quality-efficiency tradeoff

**Null Hypothesis** (H4₀): Semantic boundaries do not maintain F1 while reducing measurements.

**Alternative Hypothesis** (H4₁): Semantic boundaries achieve F1 ≥ 0.90 × baseline with <30% measurements.

**Statistical Test**: Equivalence test (non-inferiority) for F1.

---

## Experimental Design

### Phase 1: Proof of Concept (Week 1)

**Objective**: Validate core assumption before full implementation.

**Experiment POC-1**: Manual entropy inspection
- **N**: 10 hand-crafted examples
  - 5 with missing code context (e.g., "How does the auth module work?" without auth files)
  - 5 with language choice only (e.g., "Explain what this function does" with full context)
- **Procedure**:
  1. Run CodeLlama-7B on each example
  2. Extract logits at decision points
  3. Compute raw entropy
  4. Manually inspect top-k predicted tokens
  5. Classify: Does high entropy correlate with missing context?
- **Success criterion**: Visual separation between two groups (missing context has higher entropy)

**Decision point**: If POC fails, reconsider entire approach. If POC succeeds, proceed to full implementation.

---

### Phase 2: Core Experiments (Weeks 10-11)

**Experiment 1**: CCE vs Raw Entropy (RQ1, RQ2)
- **Dataset**: 100 code Q&A examples
- **Models**: CodeLlama-7B, CodeLlama-13B
- **Methods**: Raw entropy, normalized entropy, prob diff, CCE
- **Analysis**: Precision-recall curves, F1 comparison, statistical tests

**Experiment 2**: Adaptive Retrieval (RQ3)
- **Dataset**: Same 100 examples
- **Baselines**: No context, full context, static retrieval, random retrieval
- **Method**: Adaptive CCE
- **Analysis**: Answer quality vs token usage tradeoff

**Experiment 3**: Measurement Strategies (RQ4)
- **Dataset**: 50 examples (subset)
- **Strategies**: Every token, every 10 tokens, line boundaries, semantic boundaries
- **Analysis**: F1 vs latency vs measurement frequency

**Experiment 4**: Threshold Sensitivity
- **Dataset**: 100 examples
- **Thresholds**: τ ∈ {0.1, 0.2, 0.3, 0.4, 0.5}
- **Analysis**: Precision-recall curves, optimal operating point

**Experiment 5**: Ablation Study
- **Components**: Code token filter, language token filter, normalization method
- **Analysis**: Performance degradation when removing each component

---

## Evaluation Metrics (Detailed)

### Uncertainty Detection Metrics

**Spike Detection Precision**:
```
precision = TP / (TP + FP)
where:
  TP = entropy spike AND missing context
  FP = entropy spike AND NOT missing context
```

**Spike Detection Recall**:
```
recall = TP / (TP + FN)
where:
  TP = entropy spike AND missing context
  FN = NO entropy spike AND missing context
```

**F1 Score**:
```
F1 = 2 × (precision × recall) / (precision + recall)
```

---

### Answer Quality Metrics

**Answer Correctness** (LLM-as-judge):
```
Prompt to GPT-4:
"Rate how well the predicted answer matches the ground truth answer.
Consider correctness, completeness, and accuracy.
Return a score from 0.0 to 1.0."

Input: predicted_answer, ground_truth_answer
Output: score ∈ [0, 1]
```

**Answer Completeness** (embedding similarity):
```
completeness = cosine_similarity(
  embedding(predicted_answer),
  embedding(ground_truth_answer)
)
```

**Hallucination Rate**:
```
hallucination_rate = unsupported_claims / total_claims
where:
  unsupported_claims = claims in answer NOT grounded in context
  (detected via NLI model or GPT-4)
```

---

### Efficiency Metrics

**Token Efficiency**:
```
efficiency = 1 - (tokens_used / baseline_tokens)
```

**Context Precision**:
```
precision = |retrieved_files ∩ ground_truth_files| / |retrieved_files|
```

**Context Recall**:
```
recall = |retrieved_files ∩ ground_truth_files| / |ground_truth_files|
```

**Latency Overhead**:
```
overhead = (time_with_entropy - time_baseline) / time_baseline
```

---

## Statistical Analysis Plan

### Significance Testing

**Paired t-test** (comparing methods on same examples):
- H0: mean(method_A) = mean(method_B)
- H1: mean(method_A) ≠ mean(method_B)
- Significance level: α = 0.05
- Test: Two-tailed paired t-test

**Effect Size** (Cohen's d):
```
d = (mean_A - mean_B) / pooled_std
Interpretation:
  d < 0.2: small effect
  0.2 ≤ d < 0.5: medium effect
  d ≥ 0.5: large effect
```

**Confidence Intervals** (95%):
```
CI = mean ± 1.96 × (std / sqrt(n))
```

---

## Threats to Validity

### Internal Validity

**Threat 1**: Token classification accuracy
- **Mitigation**: Test multiple classification methods, report sensitivity

**Threat 2**: Ground truth subjectivity (what context is "needed"?)
- **Mitigation**: Multiple annotators, inter-rater reliability (Kappa ≥ 0.7)

**Threat 3**: Threshold selection bias
- **Mitigation**: Cross-validation, separate train/test split for threshold tuning

### External Validity

**Threat 4**: Limited to TypeScript/Python
- **Mitigation**: Acknowledge limitation, discuss generalization in future work

**Threat 5**: Small model size (7B, 13B parameters)
- **Mitigation**: Test on multiple model sizes, acknowledge limitations of larger models

**Threat 6**: Benchmark may not represent real usage
- **Mitigation**: Curate diverse examples from real projects, report dataset statistics

### Construct Validity

**Threat 7**: LLM-as-judge subjectivity
- **Mitigation**: Human evaluation on subset (20 examples), compute correlation

**Threat 8**: Entropy may not measure "uncertainty"
- **Mitigation**: Qualitative analysis of high/low entropy cases, manual inspection

---

## Success Criteria (Summary)

### Minimum Publishable Results

For paper acceptance, we need **at least 2 of 4** hypotheses confirmed:

| Hypothesis | Minimum Threshold | Ideal Target |
|------------|------------------|--------------|
| H1: Entropy detects missing context | F1 ≥ 0.60 | F1 ≥ 0.70 |
| H2: CCE > raw entropy | +5 points | +10 points |
| H3: Token savings | 20% savings | 30% savings |
| H4: Semantic boundaries | F1 ≥ 0.85 × baseline | F1 ≥ 0.90 × baseline |

**Required for publication**:
- H1 must succeed (otherwise, entropy doesn't work at all)
- At least one of H2, H3, or H4 must succeed

**Ideal for strong paper**:
- All 4 hypotheses succeed
- Statistical significance (p < 0.05) for all comparisons
- Qualitative insights from failure cases

---

## Contingency Plans

### If H1 fails (entropy doesn't detect missing context)

**Plan A**: Pivot to analysis paper
- Title: "When Does Entropy Fail? A Study of Uncertainty Detection in Code Generation"
- Contribution: Diagnostic insights, failure mode analysis
- Venue: Workshop or short paper track

**Plan B**: Use attention weights instead of entropy
- Replace entropy with attention-based uncertainty
- Measure attention dispersion over context

---

### If H2 fails (CCE doesn't beat raw entropy)

**Plan A**: Report null result honestly
- Contribution: "We tested CCE, here's why it doesn't help"
- Still publishable if analysis is thorough

**Plan B**: Hybrid metric
- Combine CCE with other signals (attention, perplexity)
- Multi-factor uncertainty model

---

### If H3 fails (no efficiency gain)

**Plan A**: Focus on precision/recall (H2)
- Contribution: Better uncertainty detection, not efficiency
- Still valuable for adaptive systems

**Plan B**: Qualitative user study
- Test with real developers
- Measure perceived usefulness, not just tokens

---

### If H4 fails (semantic boundaries don't work)

**Plan A**: Use simpler strategy (every 10 tokens)
- Accept higher overhead as limitation
- Focus on effectiveness, not efficiency

**Plan B**: Learned measurement points
- Train classifier to predict when to measure
- Contribution: Adaptive measurement strategy

---

## Timeline Alignment

| Week | Research Question Focus |
|------|------------------------|
| 1 | POC validation (H1 preliminary) |
| 2-3 | Implementation (all metrics) |
| 4-5 | Monitor integration (RQ4) |
| 6-7 | Adaptive retrieval (RQ3) |
| 8-9 | Evaluation framework |
| 10 | RQ1, RQ2 (Exp 1, 2) |
| 11 | RQ3, RQ4 (Exp 3, 4, 5) |
| 12-13 | Analysis, paper writing |
| 14 | Submission |

---

## Expected Contributions (Paper Claims)

### Primary Contribution
**Contrastive Code Entropy (CCE)**: A novel uncertainty metric that distinguishes code knowledge gaps from language choice by computing entropy over code tokens vs language tokens separately.

### Secondary Contributions
1. **Code token taxonomy**: Classification of model vocabulary into code/language/other categories
2. **Semantic boundary measurement**: Code-aware strategy for when to measure entropy
3. **Adaptive retrieval system**: End-to-end implementation in RepoSynth
4. **Evaluation benchmark**: 100 code Q&A examples with ground truth context annotations

### Empirical Findings
- CCE improves spike detection F1 by X% over raw entropy (RQ2)
- Adaptive retrieval achieves Y% token savings with Z% quality preservation (RQ3)
- Semantic boundaries reduce measurements by W% with <T% F1 loss (RQ4)

*(Values X, Y, Z, W, T to be filled after experiments)*

---

## Open Research Questions (Future Work)

### Short-term (Within this project)
1. What is optimal threshold τ for different model sizes?
2. How does CCE performance vary across programming languages?
3. Can CCE be combined with attention-based uncertainty?

### Long-term (Future publications)
1. Can CCE be used for other tasks (code editing, debugging)?
2. Can measurement points be learned (RL for adaptive measurement)?
3. Does CCE work with larger models (70B+, GPT-4 scale)?
4. Can users calibrate CCE thresholds for their codebases?

---

**Document Status**: Complete
**Next Steps**:
1. Design POC experiment (10 examples)
2. Implement POC notebook for Google Colab
3. Run POC and validate H1 preliminary

**Last Updated**: December 2024
