# Week 7 Integration Tests

Complete test suite for validating adaptive generation pipeline with uncertainty-triggered context retrieval.

## Overview

This test suite validates that all Week 7 components work together correctly:

1. **AdaptiveGenerator** - Main generation loop
2. **UncertaintyMonitor** - Entropy/CCE measurement
3. **TopicInferrer** - Query construction from uncertainty
4. **AdaptiveRetriever** - Context fetching
5. **ContextManager** - Token budget management

## Test Categories

### Missing Context (5 tests)
Tests that system detects missing API/library knowledge and retrieves documentation:
- TC001: Pandas API without context
- TC002: React state management
- TC003: Firebase authentication
- TC004: FastAPI route dependencies
- TC005: NumPy array operations

### Language Choice Only (3 tests)
Tests that system DOES NOT retrieve for pure language questions:
- TC006: Synonym definition question
- TC007: How-to phrasing question
- TC008: Architecture overview question

### Mixed Uncertainty (2 tests)
Tests scenarios with both confident and uncertain phases:
- TC009: Code question with partial context
- TC010: Complex multi-part question

## Running Tests

### Run All Tests
```bash
cd research/integration_tests
python test_runner.py
```

### Run Single Test
```bash
python test_runner.py
```

## Test Output

### Console Output
```
======================================================================
Running Test: TC001 - Pandas API Without Context
======================================================================
Category: missing_context
Query: How do I use pandas.read_csv()...

Generating...
✓ Generation complete
  Predicted token: token42
  Time: 0.12s

✅ Test TC001 PASSED
```

### Results JSON
```json
{
  "total_tests": 10,
  "passed_tests": 9,
  "failed_tests": 1,
  "success_rate": 0.9,
  "test_results": [...]
}
```

## Mock Components

The test suite uses full-simulation mocks to avoid requiring actual LLM:

### MockCodeLlama
- Simulates realistic uncertainty patterns
- Generates high/low entropy logits on demand
- Reproducible via seed

### MockRetriever
- Simulates semantic search with relevance scoring
- Built-in documentation for common libraries (pandas, React, Firebase, etc.)
- Realistic latency simulation
- Customizable mock data

### MockTokenizer
- Simulates HuggingFace tokenizer
- Handles special tokens
- Approximate subword tokenization

## Dependencies

```bash
pip install numpy
```

Mock components use only standard library + numpy (already required).

## Troubleshooting

### Test Fails with "Generation failed"
- Check mock component initialization
- Verify test case configuration
- Check for exception trace

### High Variability in Results
- Verify mock seed is consistent
- Check for randomness in test implementation

## Next Steps

After integration tests pass:
1. Review failed tests and fix issues
2. Run performance validation: `python performance_validation.py`
3. Proceed to Week 8 (Evaluation Framework)
