# Research Plan Update Summary

**Date**: December 22, 2024
**Status**: Week 1 POC Complete → Updated Plan for Week 2+

---

## What Changed

### Core Methodology: Now Using Hybrid Token Classification

**Before (Original Plan):**
- Keyword-only token classification
- Target: >70% vocabulary coverage
- Relies on predefined lists of programming keywords and common English words

**After (Updated Plan):**
- **Hybrid keyword + embedding classification**
- Target: >95% vocabulary coverage
- Two-stage approach:
  1. **Fast path**: Keyword lookup for common tokens (~50%)
  2. **Slow path**: Embedding similarity for domain-specific terms (~50%)

---

## Why This Change Was Necessary

### Week 1 POC Results Revealed Critical Limitation

**What Worked:**
- ✅ CCE successfully distinguished missing_context (CCE = +1.06) from language_choice (CCE = -1.01)
- ✅ Statistical significance achieved (p < 0.05, Cohen's d = 6.86)
- ✅ Core hypothesis validated

**What We Discovered:**
- ⚠️ Only ~50% of vocabulary was classified (code_prob_mass = 17%, language_prob_mass = 35%)
- ⚠️ **48% of tokens were classified as "other" and EXCLUDED from CCE**
- ⚠️ Critical terms missed: "requests", "pandas", "Firebase", "useState", "FastAPI"
- ⚠️ These are exactly the library/API names where uncertainty matters most!

**The Problem:**
```python
# What POC actually measured:
"How do I use requests to make a GET request?"
                ^^^^^^^^ → "other" (excluded from CCE!)

# CCE was measuring:
- Should I write `import` or `from`? (syntactic)
- NOT: Which method from requests library? (semantic)
```

---

## Key Updates to Research Plan

### 1. Week 2: Token Classification (Days 3-5)

**New Implementation:**

```python
class TokenClassifier:
    def __init__(self, tokenizer):
        # Fast path: Keywords
        self.code_keywords = {'def', 'class', 'import', ...}
        self.language_keywords = {'the', 'is', 'explain', ...}

        # Slow path: Embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.code_prototype = build_prototype([
            'def', 'class', 'requests', 'pandas', 'React',
            'useState', 'Firebase', 'FastAPI', ...
        ])
        self.language_prototype = build_prototype([
            'the', 'explain', 'describes', 'demonstrates', ...
        ])

    def classify(self, token_id):
        # Stage 1: Fast keyword lookup
        if token in keywords:
            return 'code' or 'language'

        # Stage 2: Embedding similarity
        sim_code = cosine_similarity(token_emb, code_prototype)
        sim_lang = cosine_similarity(token_emb, language_prototype)

        if sim_code > sim_lang + MARGIN:
            return 'code'  # ← Now captures "requests", "pandas", etc!
        elif sim_lang > sim_code + MARGIN:
            return 'language'
        else:
            return 'other'
```

**Day-by-Day Breakdown:**
- Day 3: Build keyword sets (fast path)
- Day 4: Create embedding prototypes (50 code examples + 50 language examples)
- Day 5: Implement hybrid classifier + validation tests

**Performance Targets:**
- Coverage: >95% (up from 70%)
- Overhead: <10% latency increase (vs keyword-only)
- Memory: ~50MB for embedding cache (acceptable)

### 2. Week 11: Expanded Ablation Study

**Added Experiments:**
1. Keyword-only CCE (70% coverage) vs Hybrid CCE (95% coverage)
2. Embedding-only CCE (no keywords)
3. Different similarity margins (0.05, 0.1, 0.2, 0.3)
4. **Attention entropy baseline** (to show CCE is better)

Expected result: Hybrid outperforms keyword-only by >15% F1

### 3. Updated Success Criteria

**New Technical Milestones:**
- ✅ Hybrid token classifier achieves >95% vocabulary coverage
- ✅ Hybrid approach outperforms keyword-only by >15% F1
- ✅ System overhead <10% (from embeddings)

### 4. Enhanced Research Contributions

**Original Contribution:**
> "We propose Contrastive Code Entropy (CCE) using keyword-based token classification"

**Updated Contribution (Stronger):**
> "We propose Contrastive Code Entropy (CCE) with hybrid semantic-syntactic token classification, achieving 95%+ vocabulary coverage while maintaining efficiency through two-stage classification"

This makes the research **more novel** and **more rigorous**.

---

## Implementation Timeline

### Week 1: ✅ COMPLETE
- POC validated hypothesis
- Identified coverage limitation
- Decided on hybrid approach

### Week 2: Token Classification (UPDATED)
- Day 1-2: Entropy calculator (unchanged)
- Day 3: Keyword sets + fast-path classification
- Day 4: Embedding prototypes + slow-path classification
- Day 5: Hybrid integration + validation

### Week 3-10: No Changes
- Rest of plan proceeds as originally designed

### Week 11: Ablation Study (ENHANCED)
- Added keyword vs hybrid comparison
- Added attention entropy baseline

---

## Why This Is Better

### Scientific Rigor
- ✅ Addresses real limitation discovered in POC
- ✅ 95% coverage vs 50% = actually measures what matters
- ✅ Captures uncertainty about specific libraries/frameworks

### Research Contribution
- ✅ More novel (hybrid classification is innovative)
- ✅ More complete (doesn't ignore half the vocabulary)
- ✅ More generalizable (works on real-world prompts)

### Practical Impact
- ✅ Detects missing API knowledge (not just syntactic choices)
- ✅ Works with modern frameworks (React, FastAPI, pandas, etc.)
- ✅ Still efficient (<10% overhead from embeddings)

### Publication Strength
- ✅ Stronger novelty claim
- ✅ Addresses obvious reviewer question: "Why not use embeddings?"
- ✅ Ablation study directly compares keyword vs hybrid
- ✅ Shows you're aware of attention-based alternatives

---

## What Stays The Same

1. **Core CCE Algorithm**: Still using contrastive entropy (H_code - H_language)
2. **Logit-based Uncertainty**: Still measuring output uncertainty (not attention)
3. **Adaptive Retrieval**: Still triggers retrieval when CCE > threshold
4. **Experimental Design**: Same baselines, metrics, dataset
5. **Timeline**: Still 15 weeks total
6. **Target Venues**: ICSE 2026, FSE 2026, ACL 2026

---

## Action Items (Next Steps)

### This Week (Week 2 Prep):
1. [ ] Install sentence-transformers: `pip install sentence-transformers`
2. [ ] Test model loading: Run test script in plan
3. [ ] Curate prototype examples:
   - Code: 50 examples (add library names to keywords)
   - Language: 50 examples (common + descriptive verbs)
4. [ ] Review hybrid classifier implementation (in updated plan)

### Week 2 (Starting Monday):
- Day 1-2: Entropy calculator implementation
- Day 3-5: Hybrid token classifier implementation
- End of week: Should have 95%+ coverage working!

---

## Questions Answered

### Q: Why not just use embeddings for everything?
**A:** Hybrid is best of both worlds:
- Keywords: Fast (O(1)), interpretable, precise
- Embeddings: Complete coverage, handles domain terms
- Combined: 95% coverage with minimal overhead

### Q: Why not use attention weights instead of logit entropy?
**A:**
- Attention measures input relevance (which tokens to attend to)
- Logit entropy measures output confidence (which token to generate)
- CCE needs output uncertainty to detect missing knowledge
- We'll compare against attention in ablation study (Week 11)

### Q: Does this complicate the implementation?
**A:** Minimal complexity increase:
- +100 lines of code for embedding classifier
- +80MB model download (one-time)
- +50MB memory at runtime
- +5-10% latency (cached lookups)
- **Huge benefit**: 95% coverage vs 50%

---

## Bottom Line

**Week 1 POC was successful** but revealed that keyword-only classification is insufficient for real-world use. The **hybrid approach** fixes this limitation while keeping the implementation practical and the research contribution strong.

**This change makes your research better, not more complicated.** It's exactly the kind of iteration you want to do EARLY (Week 1 decision point) rather than during paper reviews.

**Status**: Ready to proceed with confidence to Week 2 implementation! 🚀
