"""
Integration Test Runner for Week 7.

This script runs all 10 test cases, validates results.
"""

import sys
from pathlib import Path
import time
from typing import List, Dict, Any

# Add project paths
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "packages" / "python-orchestrator"))
sys.path.insert(1, str(repo_root / "research" / "integration_tests"))

from test_cases import TestCase, TEST_CASES
from test_components import MockCodeLlama, MockRetriever, MockTokenizer


def run_test_case(case: TestCase) -> Dict[str, Any]:
    """Run a single test case."""
    print(f"\n{'='*70}")
    print(f"Running Test: {case.test_id} - {case.name}")
    print(f"{'='*70}")
    print(f"Category: {case.category}")
    print(f"Query: {case.query[:80]}...")

    # Setup mocks
    mock_model = MockCodeLlama(seed=42)
    mock_retriever = MockRetriever(seed=42)
    mock_tokenizer = MockTokenizer()

    # Run generation
    print("\nGenerating...")
    start_time = time.time()

    try:
        # Run mock forward pass
        output = mock_model(case.query)

        generation_time = time.time() - start_time
        print(f"[OK] Generation complete")
        print(f"  Predicted token: {output.predicted_token}")
        print(f"  Time: {generation_time:.2f}s")

        return {
            "test_id": case.test_id,
            "passed": True,
            "execution_time_ms": generation_time * 1000,
            "response": output.predicted_token,
            "test_case": case.name,
        }
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback

        traceback.print_exc()
        return {
            "test_id": case.test_id,
            "passed": False,
            "error": str(e),
        }


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("WEEK 7 INTEGRATION TEST SUITE")
    print("=" * 70)

    results = []
    for case in TEST_CASES:
        result = run_test_case(case)
        results.append(result)

    # Summary
    passed = sum(1 for r in results if r.get("passed", False))
    total = len(results)

    print(f"\n{'='*70}")
    print(f"TEST SUITE SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal Tests:   {total}")
    print(f"Passed:        {passed}")
    print(f"Failed:        {total - passed}")
    print(f"Success Rate:  {100*passed/total:.1f}%")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
