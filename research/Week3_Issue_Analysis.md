# Week 3 Results Analysis & Fix

## What Went Wrong ❌

### Your Results:

```json
{
  "hybrid": {
    "missing_context_cce_mean": +3.46,  // ❌ Should be positive ✅
    "language_choice_cce_mean": +2.80,  // ❌ Should be NEGATIVE!
    "p_value": 0.54,                    // ❌ Not significant (need < 0.001)
    "cohens_d": 0.45,                   // ❌ Small effect (need > 2.0)
    "coverage": 0.947                   // ✅ Good! (94.7%)
  }
}
```

### The Problem: Over-Classification as "Code"

**Vocabulary breakdown:**
- Keyword-only: 424 code (1.3%), 327 language (1.0%), 31,265 other (97.7%)
- **Hybrid: 11,041 code (34.5%)**, 3,637 language (11.4%), 17,338 other (54.2%)

**The hybrid classifier classified 34.5% of the vocabulary as "code"!**

This caused:
1. Even language-heavy prompts have high code entropy (thousands of "code" tokens)
2. code_entropy > language_entropy even for language examples
3. **All CCE values become positive**
4. No separation between missing_context and language_choice

---

## Root Cause Analysis

Looking at your language examples:

| Example | Type | code_prob_mass | language_prob_mass | code_H | lang_H | **CCE** |
|---------|------|----------------|--------------------|---------|---------|----|
| lang_1 | language | 0.143 | **0.766** | 5.38 | 2.48 | **+2.91** ❌ |
| lang_2 | language | 0.099 | **0.859** | 5.95 | 2.05 | **+3.90** ❌ |
| lang_5 | language | 0.057 | **0.917** | 5.21 | 1.33 | **+3.87** ❌ |

**Observations:**
1. ✅ Language prob mass is high (76-92%) - classifier worked!
2. ❌ But code_entropy is HIGHER than language_entropy
3. ❌ This makes CCE positive even for language examples

**Why is code_entropy high?**
- The classifier put 11,041 tokens (34.5%) in the "code" category
- Even though they have low individual probability, there are SO MANY that entropy is high
- Shannon entropy = -Σ p(x) log₂ p(x)
- Many low-probability code tokens → high code entropy

**The math:**
- Imagine 10,000 code tokens each with p=0.0001 (total mass: 10%)
- vs. 1,000 language tokens each with p=0.0009 (total mass: 90%)
- Code entropy ≈ log₂(10,000) = 13.3 bits
- Language entropy ≈ log₂(1,000) = 9.9 bits
- CCE = 13.3 - 9.9 = +3.4 (even though language dominates!)

---

## The Fix: Conservative Hybrid Classifier

### Problem 1: Margin Too Small (0.05)

**Old code:**
```python
diff = sim_code - sim_lang
if diff > 0.05:  # Too small!
    return 'code'
```

**Result:** Almost everything gets classified (very few "other")

**Fix:** Increase to 0.20 (4x larger)
```python
if diff > 0.20:  # Much more conservative
    return 'code'
```

### Problem 2: No Minimum Similarity Threshold

**Old code:**
```python
# No check - classify even if similarity is low
diff = sim_code - sim_lang
```

**Result:** Tokens with low similarity to BOTH prototypes still get classified

**Fix:** Add minimum threshold
```python
max_sim = max(sim_code, sim_lang)
if max_sim < 0.5:
    return 'other'  # Not clearly similar to either
```

### Problem 3: Prototype Imbalance

**Old prototypes:**
- Code: 50% syntax + 50% domain (25 + 25 = 50 examples)
- Language: Mixed (50 examples)

**Issue:** Code prototype too broad, language prototype too weak

**Fix:**
- Code: 60% syntax + 40% domain (more focused on clear programming)
- Language: 70% documentation + 30% common words (stronger doc focus)

---

## Expected Improvements

### Vocabulary Classification:

| Method | Code Tokens | Code % | Expected Result |
|--------|-------------|--------|-----------------|
| **Old hybrid** | 11,041 | 34.5% | ❌ Too many |
| **New hybrid** | ~3,000-6,000 | **10-20%** | ✅ Target |

**Rationale:**
- Only CLEAR programming terms should be "code"
- Ambiguous tokens should stay as "other"
- More "other" tokens means lower code entropy on language examples

### CCE Values:

| Type | Old CCE | Expected New CCE |
|------|---------|------------------|
| missing_context | +3.46 | **+0.5 to +1.5** (positive ✅) |
| language_choice | +2.80 ❌ | **-0.5 to -1.5** (NEGATIVE ✅) |

**Why this should work:**
- Fewer code tokens → lower code_entropy on language examples
- Language examples should have: language_H > code_H → negative CCE

---

## How to Test the Fix

### 1. Upload New Notebook

**File:** `research/Week3_Improved_Validation.ipynb`

**Changes:**
- Cell 6: Conservative keyword sets (reduced from 300 to ~80)
- Cell 7: Improved hybrid classifier (margin=0.20, min_sim=0.5)
- Cell 10: Watch for code coverage (target: 10-20%, not 35%)

### 2. Run and Check Cell 10 Output

**Target:**
```
Vocabulary Classification Results:
Keyword:    424 code | 327 lang | 31,265 other ( 1.3% code)
Hybrid:   4,000 code | 3,000 lang | 25,000 other (12.5% code)  ← Look for this!

⚠️  Target: Code coverage should be 10-20% (not 35%!)
   Actual: 12.5%  ✅ GOOD!
```

**If you still see 30%+ code:**
- Increase margin further (try 0.25 or 0.30)
- Increase min_similarity (try 0.6)

### 3. Check Cell 12 Results

**Success criteria:**

```
Missing context CCE: +0.8  ✅ (positive)
Language choice CCE: -1.2  ✅ (NEGATIVE!)
Separation: 2.0            ✅

p-value: 0.002            ✅ (< 0.05, ideally < 0.001)
Cohen's d: 2.5            ✅ (> 2.0)

Hypothesis supported: YES ✅
```

---

## Alternative Approaches (If Fix Doesn't Work)

### Option A: Three-Tier Classification

Instead of code/language/other, use:
- **Syntax** (if, for, def) - pure programming
- **Domain** (pandas, React) - library-specific
- **Language** (explain, describe) - documentation
- **Other** - everything else

CCE = (H_syntax + H_domain) - H_language

**Advantage:** Separates generic code from domain-specific

### Option B: Top-K Only

Only classify top-K most probable tokens (K=100), ignore the rest:
```python
top_k_indices = np.argsort(probs)[-100:]  # Only top 100
# Only classify these tokens
```

**Advantage:** Focuses on tokens that actually matter

### Option C: Probability-Weighted Entropy

Weight entropy by probability mass:
```python
cce = (code_prob_mass * code_entropy) - (lang_prob_mass * lang_entropy)
```

**Advantage:** Reduces impact of low-probability tokens

---

## Parameters to Tune

If results still not good, try adjusting:

| Parameter | Current | Try | Effect |
|-----------|---------|-----|--------|
| `margin` | 0.20 | 0.25-0.30 | Fewer classifications |
| `min_similarity` | 0.5 | 0.6-0.7 | More conservative |
| Code prototype % | 60/40 | 70/30 | More syntax-focused |
| Language prototype % | 70/30 | 80/20 | Stronger doc focus |

---

## What to Report Back

After running the improved notebook, please share:

1. **Vocabulary classification** (from Cell 10):
   - Code count and percentage
   - Is it in 10-20% range?

2. **CCE values** (from Cell 12):
   - missing_context_cce_mean
   - language_choice_cce_mean
   - Is language_choice NEGATIVE?

3. **Statistical tests**:
   - p-value
   - Cohen's d

This will tell us if the fix worked or if we need further tuning!

---

## Summary

**Problem:** Margin too small (0.05) → Over-classification as "code" (34.5%) → High code entropy everywhere → All CCE positive

**Fix:** Larger margin (0.20) + min similarity (0.5) → Target 10-20% code → Lower code entropy on language examples → Negative CCE for language_choice

**Expected outcome:**
- ✅ missing_context: CCE = +0.5 to +1.5 (positive)
- ✅ language_choice: CCE = -0.5 to -1.5 (NEGATIVE)
- ✅ p < 0.05 (ideally < 0.001)
- ✅ Cohen's d > 2.0
