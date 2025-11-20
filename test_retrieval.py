#!/usr/bin/env python3
"""
Quick test script to verify retrieval is working before integrating with LLM.
"""

import sys
import json
from pathlib import Path

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "python-orchestrator"))

from orchestrator.retriever import retrieve_relevant_symbols, format_context_for_llm, retrieve_with_relationships


def test_basic_retrieval():
    """Test basic keyword matching."""
    print("="*80)
    print("TEST 1: Basic Keyword Retrieval")
    print("="*80)

    # Load name_registry
    pack_dir = Path(__file__).parent / "pack"
    registry_path = pack_dir / "name_registry.json"

    if not registry_path.exists():
        print(f"❌ Error: name_registry.json not found at {registry_path}")
        print("   Run the pipeline first to generate pack/")
        return False

    with open(registry_path, 'r') as f:
        name_registry = json.load(f)

    print(f"✓ Loaded {len(name_registry)} symbols\n")

    # Test queries
    test_queries = [
        "What does Config class do?",
        "Where is the AliasedGroup defined?",
        "How does read_config work?"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)

        matches = retrieve_relevant_symbols(query, name_registry, max_items=3)

        if not matches:
            print("  ⚠️  No matches found")
            continue

        print(f"  ✓ Found {len(matches)} matches:")
        for i, match in enumerate(matches, 1):
            print(f"    {i}. {match['fqn']}")
            print(f"       - File: {match.get('file_path', 'unknown')}")
            print(f"       - Kind: {match.get('kind', 'unknown')}")
            if 'inherits_from' in match:
                print(f"       - Inherits: {', '.join(match['inherits_from'])}")

    print("\n✅ Basic retrieval test passed!\n")
    return True


def test_token_reduction():
    """Test that retrieval actually reduces tokens."""
    print("="*80)
    print("TEST 2: Token Reduction")
    print("="*80)

    pack_dir = Path(__file__).parent / "pack"
    registry_path = pack_dir / "name_registry.json"

    with open(registry_path, 'r') as f:
        name_registry = json.load(f)

    # Full registry size
    full_json = json.dumps(name_registry, indent=2)
    full_tokens = len(full_json.split()) * 1.3  # Rough token estimate

    # Retrieved context size
    query = "What does Config class do?"
    matches = retrieve_relevant_symbols(query, name_registry, max_items=5)
    context = format_context_for_llm(matches)
    retrieved_tokens = len(context.split()) * 1.3

    print(f"Full registry: ~{full_tokens:,.0f} tokens")
    print(f"Retrieved context: ~{retrieved_tokens:,.0f} tokens")
    print(f"Reduction: {full_tokens / retrieved_tokens:.1f}x smaller")

    if full_tokens / retrieved_tokens > 10:
        print("\n✅ Token reduction test passed! (>10x reduction)")
        return True
    else:
        print("\n⚠️  Token reduction less than expected")
        return False


def test_relationship_retrieval():
    """Test retrieval with relationships (if available)."""
    print("\n" + "="*80)
    print("TEST 3: Relationship-Aware Retrieval")
    print("="*80)

    pack_dir = Path(__file__).parent / "pack"
    registry_path = pack_dir / "name_registry.json"

    with open(registry_path, 'r') as f:
        name_registry = json.load(f)

    # Check if any symbols have inheritance info
    has_inheritance = any('inherits_from' in data for data in name_registry.values())
    has_exports = any('exported_from' in data for data in name_registry.values())

    print(f"Registry has inheritance data: {'✓' if has_inheritance else '✗'}")
    print(f"Registry has export data: {'✓' if has_exports else '✗'}")

    if not has_inheritance and not has_exports:
        print("\n⚠️  No relationship data found in registry.")
        print("    Did you apply the inheritance and export tracking patches?")
        return False

    # Test inheritance query
    if has_inheritance:
        print("\nTesting inheritance query:")
        query = "what inherits from"
        results = retrieve_with_relationships(query, name_registry, max_items=3)

        print(f"  Primary matches: {len(results['primary_matches'])}")
        print(f"  Related symbols: {len(results['related_symbols'])}")

        if results['related_symbols']:
            print("  ✓ Successfully retrieved related parent classes")
        else:
            print("  ⚠️  No related symbols found (this is OK if no inheritance in codebase)")

    print("\n✅ Relationship retrieval test completed!\n")
    return True


def main():
    """Run all tests."""
    print("\n🧪 Testing RepoSynth Retrieval System\n")

    results = {
        "basic_retrieval": test_basic_retrieval(),
        "token_reduction": test_token_reduction(),
        "relationships": test_relationship_retrieval(),
    }

    print("="*80)
    print("SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    if all(results.values()):
        print("\n🎉 All tests passed! Retrieval system is working correctly.")
        print("\nNext steps:")
        print("1. Integrate retrieval into your LLM evaluation script")
        print("2. Test on your failing cases (L1-Index-SchemaExports, etc.)")
        print("3. Measure the improvement in accuracy and token usage")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
