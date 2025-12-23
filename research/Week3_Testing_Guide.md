# Week 3 Testing Guide

## Objective

Validate that the Week 2 hybrid classifier implementation:
1. **Achieves >90% coverage** (vs 53% keyword-only from Week 1)
2. **Correctly classifies domain-specific terms** (pandas, Firebase, React, etc.)
3. **Maintains strong CCE separation** (p < 0.001, Cohen's d > 2.0)
4. **Runs with <10% overhead** vs keyword-only baseline

---

## What to Test

### 1. Upload to Google Colab

**File**: `research/Week3_Hybrid_Validation.ipynb`

**Steps:**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload Notebook**
3. Upload `Week3_Hybrid_Validation.ipynb`
4. Set runtime to **GPU** (Runtime → Change runtime type → GPU → T4)

### 2. Run All Cells

The notebook will:
- Install dependencies (transformers, sentence-transformers, scipy, etc.)
- Download CodeLlama-7B-Instruct (~13GB, first run only)
- Load embedding model (all-MiniLM-L6-v2)
- Run 10 test examples through **both** classifiers:
  - **Keyword-only** (Week 1 baseline)
  - **Hybrid** (Week 2 implementation)
- Compare coverage and CCE performance
- Generate visualizations and statistics

**Expected Runtime**: ~30 minutes (includes model download on first run)

---

## Expected Results

### Coverage Comparison

| Method | Expected Coverage | Week 1 Actual |
|--------|------------------|---------------|
| Keyword-only | ~53% | 53% ✅ |
| **Hybrid** | **>90%** | *To be validated* |

**What to check:**
- Hybrid "Other prob mass" should be <10%
- Should see significant improvement in coverage bar chart (Cell 13)

### Domain-Specific Terms

**Critical validation** - These should now be classified as 'code':

| Term | Week 1 (Keyword) | Week 2 (Hybrid) |
|------|------------------|-----------------|
| pandas | ❌ other | ✅ code |
| numpy | ❌ other | ✅ code |
| requests | ❌ other | ✅ code |
| Firebase | ❌ other | ✅ code |
| useState | ❌ other | ✅ code |
| FastAPI | ❌ other | ✅ code |

**Where to check:**
- Cell 11 output shows per-example coverage
- Look for "Code prob mass" increase in domain-specific examples (code_1-5)

### Statistical Validation

**Expected results (Hybrid method):**

| Metric | Target | Week 1 POC | Week 3 Expected |
|--------|--------|------------|-----------------|
| missing_context CCE | > 0 | +0.685 | Similar |
| language_choice CCE | < 0 | -1.345 | Similar |
| Separation | High | 2.030 | Maintained |
| p-value | < 0.001 | 0.0007 ✅ | < 0.001 |
| Cohen's d | > 2.0 | 3.369 ✅ | > 2.0 |

**What to check (Cell 12):**
- Hybrid method maintains strong separation
- p-value stays significant (p < 0.001)
- Cohen's d remains very large (|d| > 2.0)

### Visualizations

**Cell 13 produces 4 charts:**

1. **CCE by Type** (top-left)
   - Missing context: positive CCE (blue boxes above 0)
   - Language choice: negative CCE (orange boxes below 0)
   - Clear separation between groups

2. **Coverage Comparison** (top-right)
   - Hybrid should have much less gray (other) than Keyword
   - More blue (code) + orange (language) probability mass

3. **Per-Example CCE** (bottom-left)
   - All code_* examples: positive CCE
   - All lang_* examples: negative CCE
   - Hybrid should maintain/improve separation

4. **Coverage Improvement** (bottom-right)
   - All bars should be positive (hybrid improves coverage)
   - Tallest bars on code_* examples (domain-specific terms benefit most)

---

## Success Criteria

After running the notebook, check **Cell 15: Week 3 Validation Summary**

Should see:

```
✅ SUCCESS CRITERIA:

1. Coverage Improvement:
   Keyword-only: ~53%
   Hybrid:       >90%
   Gain:         +37+ percentage points
   Target: >90% coverage → PASS ✅

2. Statistical Significance:
   p-value: <0.001
   Target: p < 0.001 → PASS ✅

3. Effect Size:
   Cohen's d: >2.0
   Target: |d| > 2.0 (very large) → PASS ✅

4. CCE Direction:
   Missing context: >0
   Language choice: <0
   Target: missing > 0, language < 0 → PASS ✅

======================================================================
🎉 ALL WEEK 3 VALIDATION CRITERIA PASSED!

✅ Hybrid classifier ready for Week 4 integration
======================================================================
```

---

## Files Generated

After running, download these files from Colab:

1. **week3_validation_results.csv** - Detailed results for all 20 experiments (10 examples × 2 methods)
2. **week3_validation_summary.json** - Summary statistics (coverage, CCE means, statistical tests)
3. **week3_validation_results.png** - 4-panel visualization

These can be used for documentation and presentations.

---

## Troubleshooting

### Out of Memory Error

**Issue**: `CUDA out of memory`

**Solution**:
- Use T4 GPU (not CPU) in Colab runtime settings
- Restart runtime and run again
- If persists, use smaller model: Change `MODEL_NAME` in Cell 3 to `"codellama/CodeLlama-7b-hf"` (non-instruct version uses less memory)

### Model Download Slow

**Issue**: Taking too long to download CodeLlama

**Solution**:
- First run will download ~13GB
- Subsequent runs use cached model (much faster)
- If timeout occurs, just restart and run again (will resume download)

### Embedding Model Error

**Issue**: `sentence_transformers` import fails

**Solution**:
- Make sure Cell 1 (installation) completed successfully
- Restart runtime and run cells in order
- Check that `!pip install sentence-transformers` succeeded

### Low Coverage Result

**Issue**: Hybrid coverage still <90%

**Possible causes**:
- Embedding model didn't load correctly (check Cell 4)
- Prototypes not built (check Cell 8 output)
- Embedding margin too high (default 0.05 is good, but can try 0.03)

**Debug**:
- Check Cell 11 output - should see "Hybrid" using embeddings
- Look at `embedding_cache` size after Cell 11 - should be >100 entries

---

## Next Steps After Validation

### If All Tests Pass ✅

1. **Document results** in `research/Week3_Results_Summary.md`
2. **Compare to Week 1 POC**:
   - Coverage improvement
   - Domain-specific term classification
   - CCE separation maintained
3. **Proceed to Week 4**: Uncertainty monitoring implementation

### If Some Tests Fail ❌

1. **Review failure cases**:
   - Which examples had low coverage?
   - Which domain terms still misclassified?
   - Is CCE separation maintained?

2. **Potential fixes**:
   - **Low coverage**: Reduce embedding margin (try 0.03 instead of 0.05)
   - **Domain terms misclassified**: Add more domain examples to prototypes
   - **Weak separation**: Check if coverage improvement affected entropy calculation

3. **Re-run tests** after adjustments

---

## Questions to Answer

After completing the validation, you should be able to answer:

1. ✅ **Did hybrid classifier achieve >90% coverage?**
   - Keyword-only: ~53%
   - Hybrid: ____%
   - Improvement: +___ percentage points

2. ✅ **Are domain-specific terms now classified correctly?**
   - pandas: ☐ code ☐ language ☐ other
   - Firebase: ☐ code ☐ language ☐ other
   - useState: ☐ code ☐ language ☐ other

3. ✅ **Is CCE separation maintained?**
   - missing_context CCE: ____
   - language_choice CCE: ____
   - p-value: ____
   - Cohen's d: ____

4. ✅ **How much did coverage improve per example?**
   - code_1 (requests): ____% → ____%
   - code_2 (useState): ____% → ____%
   - code_3 (Firebase): ____% → ____%
   - code_4 (pandas): ____% → ____%

---

## Contact

If you encounter issues not covered in this guide, please:
- Check `research/Week2_Complete_Summary.md` for implementation details
- Review `packages/python-orchestrator/orchestrator/entropy/token_classifier.py` for classifier code
- Open an issue with:
  - Error message
  - Cell number where error occurred
  - Colab runtime settings (GPU/CPU)

---

**Ready to validate!** 🚀

Upload `Week3_Hybrid_Validation.ipynb` to Google Colab and run all cells to validate the Week 2 implementation.
