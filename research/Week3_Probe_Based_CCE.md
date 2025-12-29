# Week 3: Probe-Based Code vs Language Uncertainty Classification

## The Improved Approach

### Original Plan Problem
- **Token Classification**: Use keyword lists to classify vocab as "code" or "language"
- **Coverage Issue**: Only ~50% of vocabulary covered (misses "pandas", "useState", "Firebase")
- **Brittleness**: Manual curation, hard to maintain

### New Solution: Hidden State Probes
Instead of keyword lists, use the LLM's own hidden states to classify tokens!

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  PROMPT: "How do I use pandas to read CSV?"                     │
│     ↓                                                            │
│  CodeLlama Forward Pass                                         │
│     ↓                                                            │
│  Hidden State at Last Token Position (4096 dims)                │
│     ↓                                                            │
│  [PROBE CLASSIFIER]                                             │
│     ↓                                                            │
│  P(will_generate_code) = 0.92  ← High! This is CODE mode        │
│     ↓                                                            │
│  Use this to classify next token probabilities                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Two-Stage Process

### Stage 1: Train Prompt-Level Probe
**Goal**: Classify entire prompts as "code-generating" vs "language-generating"

**Training Data** (20 examples):
- **Label 1 (Code)**: Prompts that will generate code
  - "Write a function to sort a list"
  - "How do I use pandas to read CSV?"
  - "Show me a React component with useState"
  - "Create a FastAPI endpoint"

- **Label 0 (Language)**: Prompts that will generate explanations
  - "Explain what recursion is"
  - "Describe how HTTP works"
  - "What are the benefits of TypeScript?"
  - "Why is async programming useful?"

**Method**: Logistic regression on hidden states (like the current notebook)

### Stage 2: Use Probe to Classify Vocabulary Tokens
**Key Insight**: We can simulate "what if the model predicts this token?"

For each token in vocabulary:
1. Create a synthetic prompt that would likely lead to that token
2. Extract hidden state
3. Run through probe → P(code_mode)
4. Token classification based on probe score

**OR (Simpler)**: Use the logits distribution directly!

When the probe says "P(code_mode) = 0.9":
- We expect high probability mass on CODE tokens
- We can partition the logits based on probe's confidence

## The Probe-Enhanced CCE Algorithm

```python
class ProbeBasedCCE:
    def __init__(self, probe_classifier):
        self.probe = probe_classifier  # Trained on prompts

    def compute_cce(self, prompt: str, model, tokenizer):
        # Get hidden state and logits
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)

        hidden_state = outputs.hidden_states[-1][:, -1, :]  # Last layer, last token
        logits = outputs.logits[:, -1, :].squeeze()

        # Probe predicts: will model generate code or language?
        probe_score = self.probe.predict_proba(hidden_state)[0, 1]  # P(code_mode)

        # Method 1: Weight tokens by probe score
        # - If probe_score = 0.9 → model is in "code mode"
        # - We expect code tokens to have high probability
        # - Compute entropy over probable code tokens vs probable language tokens

        probs = softmax(logits)

        # Get top-k tokens (where the probability mass is)
        top_k = 100
        top_indices = np.argsort(logits)[-top_k:]
        top_probs = probs[top_indices]

        # Classify these top tokens using a secondary method
        # (could use keyword list just for top-k, or embedding similarity)

        code_mask = self._classify_tokens_as_code(top_indices, tokenizer)
        lang_mask = ~code_mask

        # Compute entropy over code vs language partitions
        code_probs_norm = top_probs[code_mask] / top_probs[code_mask].sum()
        lang_probs_norm = top_probs[lang_mask] / top_probs[lang_mask].sum()

        H_code = shannon_entropy(code_probs_norm)
        H_lang = shannon_entropy(lang_probs_norm)

        CCE = H_code - H_lang

        return {
            'cce': CCE,
            'probe_score': probe_score,
            'h_code': H_code,
            'h_lang': H_lang,
            'uncertainty_type': 'code' if CCE > 0 else 'language'
        }

    def _classify_tokens_as_code(self, token_ids, tokenizer):
        """
        Classify tokens using lightweight method.

        Options:
        1. Keyword lookup (fast, works for top-k)
        2. Character patterns (contains _, camelCase, etc.)
        3. Pre-computed token embeddings
        """
        # For top-k tokens, even simple heuristics work well!
        code_mask = []
        for token_id in token_ids:
            token_str = tokenizer.decode([token_id])
            is_code = self._is_code_like(token_str)
            code_mask.append(is_code)
        return np.array(code_mask)

    def _is_code_like(self, token: str) -> bool:
        """Simple heuristics for code vs language."""
        # Keywords
        code_keywords = {'def', 'class', 'import', 'function', 'const', '(', '{', '='}
        if token.strip().lower() in code_keywords:
            return True

        # Patterns
        if '_' in token or token[0].isupper():  # snake_case or PascalCase
            return True
        if token.strip() in ['{', '}', '(', ')', '[', ']', ';', ':']:
            return True

        # Language words
        lang_words = {'the', 'is', 'this', 'that', 'explain', 'describes'}
        if token.strip().lower() in lang_words:
            return False

        return False  # Default to language (conservative)
```

## Why This is Better

### Advantages Over Keyword-Only
1. **Probe captures semantic intent**: "Will model generate code?" is learned, not hard-coded
2. **Better coverage**: Even if we classify top-k tokens simply, probe guides the overall uncertainty
3. **Adaptive**: Probe score tells us how confident to be about code/language split
4. **Principled**: Uses model's own representations

### Advantages Over Pure Hidden State Classification
1. **More interpretable**: We still get CCE metric (H_code - H_lang)
2. **Granular**: Can analyze entropy over specific token types
3. **Explainable**: Can inspect which tokens contribute to uncertainty

## Week 3 Experiment Design

### Test Examples (20 total)

**Code Uncertainty** (10 examples) - High H_code expected:
```python
[
    "import",                                    # Which module? pandas? numpy? requests?
    "from sklearn import",                       # Which sklearn module?
    "def process_data(df):\n    df.",          # Which pandas method?
    "const [state, setState] = use",            # useState? useEffect? useRef?
    "async function fetch_data() {\n    await" # Which async operation?
    "model = tf.keras.",                        # Which Keras class?
    "app = FastAPI()\n@app.",                   # Which decorator?
    "SELECT * FROM users WHERE",                # Which SQL condition?
    "git",                                       # Which git command?
    "docker run -",                             # Which docker flag?
]
```

**Language Uncertainty** (10 examples) - High H_lang expected:
```python
[
    "This function",                            # performs? calculates? returns?
    "The algorithm is",                         # efficient? recursive? complex?
    "Code quality can be",                      # improved? measured? maintained?
    "Explain what this code",                   # does? achieves? demonstrates?
    "The main advantage of async programming is", # performance? readability? scalability?
    "TypeScript provides better",               # type safety? tooling? developer experience?
    "Recursion is useful when",                 # Which use case?
    "REST APIs are designed to",                # Which principle?
    "The difference between let and const is",  # Which explanation?
    "Unit tests help",                          # Which benefit?
]
```

### Validation Criteria

**Success Metrics**:
1. **Probe Accuracy**: >80% on 20-example LOO CV
2. **CCE Separation**:
   - Mean CCE for "code uncertainty" > +0.5
   - Mean CCE for "language uncertainty" < -0.5
   - Statistical significance: p < 0.05
3. **Correlation**: Probe score should correlate with CCE

### Visualization

Plot for each example:
```
Example: "import"
├─ Probe Score: P(code_mode) = 0.95  ← High (model knows it will generate code)
├─ CCE: +1.2                          ← Positive (high code entropy)
├─ H_code: 3.8                        ← High (uncertain WHICH import)
├─ H_lang: 2.6                        ← Lower
└─ Interpretation: Model is in "code mode" but uncertain which module to import

Example: "This function"
├─ Probe Score: P(code_mode) = 0.12  ← Low (model knows it will explain)
├─ CCE: -0.9                          ← Negative (high language entropy)
├─ H_code: 2.1                        ← Lower
├─ H_lang: 3.0                        ← High (uncertain which explanation verb)
└─ Interpretation: Model is in "language mode" and uncertain which verb to use
```

## Implementation Plan

### Day 1: Train Prompt-Level Probe
- Create 20 prompt examples (10 code, 10 language)
- Extract hidden states
- Train logistic regression probe
- Validate with LOO CV

### Day 2: Integrate Probe with CCE
- Implement `ProbeBasedCCE` class
- Test on 20 uncertainty examples
- Compare with keyword-only CCE

### Day 3-4: Experiments and Visualization
- Run probe-based CCE on all examples
- Generate comparison plots
- Statistical analysis (t-tests, correlation)

### Day 5: Documentation and Comparison
- Document results
- Compare three approaches:
  1. Keyword-only CCE (Week 2)
  2. Probe-only classification (current notebook)
  3. Probe-enhanced CCE (new approach)

## Research Questions

**RQ1**: Can a prompt-level probe accurately predict whether the model will generate code or language?
- Hypothesis: >80% accuracy on held-out examples

**RQ2**: Does probe-enhanced CCE better distinguish code vs language uncertainty than keyword-only CCE?
- Hypothesis: Higher separation, better statistical significance

**RQ3**: Do probe scores correlate with CCE values?
- Hypothesis: High probe score → positive CCE (code uncertainty)
- Hypothesis: Low probe score → negative CCE (language uncertainty)

## Next Steps

After Week 3 validation:
- If probe-enhanced CCE works: Use it in Week 4-7 (monitoring system)
- If probe-only works better: Pivot to pure probe approach
- If keyword CCE is comparable: Stick with simpler method

The key insight: **We can use the model's own representations to guide our uncertainty analysis, rather than relying on brittle keyword lists!**
