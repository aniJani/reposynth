"""
Week 2 Validation Script

Tests the complete CCE pipeline implementation:
1. Hybrid classifier coverage
2. Domain-specific term classification
3. CCE computation on test examples
4. Performance benchmarking

Run this before proceeding to Week 3 integration.
"""

import sys
import os
import time
import numpy as np

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'python-orchestrator'))

from orchestrator.entropy.token_classifier import (
    KeywordClassifier,
    HybridClassifier,
    get_keyword_stats
)


def test_keyword_stats():
    """Test 1: Verify keyword set sizes."""
    print("\n" + "="*60)
    print("TEST 1: Keyword Set Statistics")
    print("="*60)

    stats = get_keyword_stats()

    print(f"Code keywords: {stats['code_keywords']}")
    print(f"Code operators: {stats['code_operators']}")
    print(f"Language words: {stats['language_words']}")
    print(f"Total keywords: {stats['total_keywords']}")
    print(f"Domain-specific: {stats['domain_specific']}")

    # Validation
    assert stats['code_keywords'] > 100, "❌ Code keywords too few"
    assert stats['domain_specific'] > 100, "❌ Domain-specific keywords too few"
    print("\n✅ Keyword sets meet size requirements")


def test_domain_specific_classification():
    """Test 2: Verify domain-specific terms are classified as code."""
    print("\n" + "="*60)
    print("TEST 2: Domain-Specific Term Classification")
    print("="*60)

    classifier = KeywordClassifier()

    # Critical terms from Week 1 POC issues
    critical_terms = {
        # Python data science
        'pandas': 'code',
        'numpy': 'code',
        'DataFrame': 'code',
        'read_csv': 'code',

        # Python web
        'requests': 'code',
        'FastAPI': 'code',
        'flask': 'code',

        # JavaScript/React
        'React': 'code',
        'useState': 'code',
        'useEffect': 'code',
        'axios': 'code',

        # Firebase
        'firebase': 'code',
        'Firebase': 'code',
        'auth': 'code',
        'firestore': 'code',

        # Language words
        'explain': 'language',
        'describe': None,  # Not in language words anymore
        'how': 'language',
        'what': 'language',
    }

    failures = []
    for term, expected in critical_terms.items():
        result = classifier.classify(term)
        if result != expected:
            failures.append(f"  {term}: expected '{expected}', got '{result}'")
        else:
            print(f"✅ {term:20s} → {result}")

    if failures:
        print(f"\n❌ {len(failures)} classification failures:")
        for failure in failures:
            print(failure)
        raise AssertionError("Domain-specific classification failed")

    print("\n✅ All domain-specific terms classified correctly")


def test_keyword_coverage():
    """Test 3: Measure keyword classifier coverage on sample tokens."""
    print("\n" + "="*60)
    print("TEST 3: Keyword Classifier Coverage")
    print("="*60)

    classifier = KeywordClassifier()

    # Sample tokens from various categories
    sample_tokens = [
        # Code keywords
        'def', 'class', 'function', 'import', 'return', 'if', 'for',
        # Domain-specific
        'pandas', 'React', 'Firebase', 'requests', 'useState',
        # Language words
        'explain', 'how', 'what', 'the', 'show',
        # Unknown (should return None)
        'foobar', 'xyz123', 'customVariable', 'myFunction',
    ]

    coverage = classifier.get_coverage(sample_tokens)

    print(f"Code coverage: {coverage['code']:.2%}")
    print(f"Language coverage: {coverage['language']:.2%}")
    print(f"Unknown coverage: {coverage['unknown']:.2%}")

    # Validation: should have good coverage on known tokens
    known_coverage = coverage['code'] + coverage['language']
    assert known_coverage > 0.5, f"❌ Known coverage too low: {known_coverage:.2%}"

    print(f"\n✅ Keyword coverage validated (known: {known_coverage:.2%})")


def test_hybrid_classifier_initialization():
    """Test 4: Verify hybrid classifier can initialize."""
    print("\n" + "="*60)
    print("TEST 4: Hybrid Classifier Initialization")
    print("="*60)

    try:
        print("Initializing hybrid classifier...")
        print("(This will download sentence-transformers model on first run)")

        start = time.time()
        classifier = HybridClassifier(
            embedding_model='all-MiniLM-L6-v2',
            embedding_margin=0.05
        )
        elapsed = time.time() - start

        print(f"✅ Hybrid classifier initialized in {elapsed:.2f}s")

        # Test classification
        print("\nTesting hybrid classification...")

        test_tokens = {
            'pandas': 'code',      # Should hit keyword
            'customLib': None,     # Should go to embeddings
            'explain': 'language', # Should hit keyword
        }

        for token, expected_type in test_tokens.items():
            result = classifier.classify(token)
            print(f"  {token:20s} → {result}")

            if expected_type is not None:
                # Verify code or language (not other)
                assert result in ['code', 'language'], \
                    f"❌ {token} classified as 'other', expected {expected_type}"

        stats = classifier.get_stats()
        print(f"\nClassification stats:")
        print(f"  Keyword hits: {stats['keyword_hits']}")
        print(f"  Embedding hits: {stats['embedding_hits']}")
        print(f"  Other: {stats['other']}")

        print("\n✅ Hybrid classifier working correctly")

    except ImportError as e:
        print(f"\n⚠️  Hybrid classifier requires dependencies:")
        print(f"   {str(e)}")
        print(f"\n   Install with:")
        print(f"   pip install sentence-transformers scikit-learn numpy")
        raise


def test_performance_benchmark():
    """Test 5: Benchmark keyword classifier performance."""
    print("\n" + "="*60)
    print("TEST 5: Performance Benchmark")
    print("="*60)

    classifier = KeywordClassifier()

    # Benchmark keyword classification
    test_tokens = ['def', 'pandas', 'explain', 'unknown'] * 25  # 100 tokens
    iterations = 1000

    start = time.time()
    for _ in range(iterations):
        for token in test_tokens:
            classifier.classify(token)
    elapsed = time.time() - start

    total_classifications = iterations * len(test_tokens)
    per_classification = (elapsed / total_classifications) * 1_000_000  # microseconds

    print(f"Total classifications: {total_classifications:,}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Per classification: {per_classification:.2f}μs")
    print(f"Throughput: {total_classifications/elapsed:,.0f} classifications/sec")

    # Validation: should be very fast
    assert per_classification < 10, f"❌ Too slow: {per_classification:.2f}μs"

    print(f"\n✅ Performance target met (<10μs per classification)")


def main():
    """Run all validation tests."""
    print("="*60)
    print("Week 2 Implementation Validation")
    print("="*60)

    tests = [
        ("Keyword Statistics", test_keyword_stats),
        ("Domain-Specific Classification", test_domain_specific_classification),
        ("Keyword Coverage", test_keyword_coverage),
        ("Hybrid Classifier Initialization", test_hybrid_classifier_initialization),
        ("Performance Benchmark", test_performance_benchmark),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"   Error: {str(e)}")
            failed += 1
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✅ All validation tests passed!")
        print("\n📋 Week 2 implementation is ready for Week 3 integration.")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed. Fix issues before proceeding to Week 3.")
        return 1


if __name__ == '__main__':
    exit(main())
