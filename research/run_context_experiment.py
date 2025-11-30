#!/usr/bin/env python3
"""
Context Optimizer Benchmark Script

Evaluates the Graph-Knapsack context optimization algorithm across different
token budgets and measures the trade-off between context size and coverage.

Usage:
    python run_context_experiment.py <pack_path> [--entry-point <file>]
    
Example:
    python run_context_experiment.py ../pack --entry-point src/index.ts
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "python-orchestrator"))

from orchestrator.context_optimizer import (
    optimize_context,
    calculate_file_scores,
    get_context_window_presets,
    PruningReport
)
from orchestrator.token_utils import TokenCounter


@dataclass
class ExperimentResult:
    """Result from a single optimization experiment."""
    token_budget: int
    model_name: str
    files_included: int
    files_pruned: int
    tokens_used: int
    tokens_pruned: int
    budget_utilization: float  # percentage
    coverage: float  # percentage of total files included
    processing_time_ms: float


@dataclass
class ExperimentSummary:
    """Summary of all experiments."""
    pack_path: str
    entry_point: Optional[str]
    total_files: int
    total_tokens: int
    results: List[ExperimentResult]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pack_path": self.pack_path,
            "entry_point": self.entry_point,
            "total_files": self.total_files,
            "total_tokens": self.total_tokens,
            "results": [asdict(r) for r in self.results]
        }


def load_pack_data(pack_path: Path) -> Dict[str, Any]:
    """Load necessary data from a pack directory."""
    data = {}
    
    # Load import graph
    import_graph_path = pack_path / "import_graph.json"
    if import_graph_path.exists():
        with open(import_graph_path, 'r', encoding='utf-8') as f:
            data["import_graph"] = json.load(f)
    else:
        data["import_graph"] = {}
    
    # Load token map
    token_map_path = pack_path / "token_map.json"
    if token_map_path.exists():
        with open(token_map_path, 'r', encoding='utf-8') as f:
            data["token_map"] = json.load(f)
    else:
        # Generate token map from source spans if not available
        print("Token map not found, generating from source spans...")
        data["token_map"] = generate_token_map_from_spans(pack_path)
    
    # Load name registry for entry point suggestions
    name_registry_path = pack_path / "name_registry.json"
    if name_registry_path.exists():
        with open(name_registry_path, 'r', encoding='utf-8') as f:
            data["name_registry"] = json.load(f)
    else:
        data["name_registry"] = {}
    
    return data


def generate_token_map_from_spans(pack_path: Path) -> Dict[str, int]:
    """Generate token map from source spans if token_map.json doesn't exist."""
    source_spans_path = pack_path / "source_spans.json"
    token_map = {}
    
    if source_spans_path.exists():
        counter = TokenCounter()
        with open(source_spans_path, 'r', encoding='utf-8') as f:
            spans = json.load(f)
            for file_path, span_data in spans.items():
                if isinstance(span_data, dict) and 'full_source' in span_data:
                    token_map[file_path] = counter.count(span_data['full_source'])
                elif isinstance(span_data, str):
                    token_map[file_path] = counter.count(span_data)
    
    return token_map


def suggest_entry_points(import_graph: Dict[str, List[str]], name_registry: Dict[str, Any]) -> List[str]:
    """Suggest entry points based on import graph analysis."""
    # Build reverse graph
    reverse_graph: Dict[str, List[str]] = {}
    for file_path, imports in import_graph.items():
        for imported in imports:
            if imported not in reverse_graph:
                reverse_graph[imported] = []
            reverse_graph[imported].append(file_path)
    
    # Find files with high in-degree (many dependents)
    in_degrees = {f: len(reverse_graph.get(f, [])) for f in import_graph.keys()}
    sorted_files = sorted(in_degrees.items(), key=lambda x: -x[1])
    
    # Return top 5 entry points
    return [f for f, _ in sorted_files[:5]]


def run_experiment(
    import_graph: Dict[str, List[str]],
    token_map: Dict[str, int],
    candidate_files: List[str],
    seed_files: List[str],
    token_budget: int,
    model_name: str
) -> ExperimentResult:
    """Run a single optimization experiment."""
    start_time = time.perf_counter()
    
    total_files = len(candidate_files)
    total_tokens = sum(token_map.get(f, 0) for f in candidate_files)
    
    # Run optimization
    included_files, report = optimize_context(
        candidate_files=candidate_files,
        import_graph=import_graph,
        token_map=token_map,
        max_tokens=token_budget,
        seed_files=seed_files
    )
    
    end_time = time.perf_counter()
    processing_time_ms = (end_time - start_time) * 1000
    
    # Calculate metrics
    budget_utilization = (report.total_tokens_included / token_budget * 100) if token_budget > 0 else 0
    coverage = (len(included_files) / total_files * 100) if total_files > 0 else 0
    
    return ExperimentResult(
        token_budget=token_budget,
        model_name=model_name,
        files_included=len(report.included_files),
        files_pruned=len(report.pruned_files),
        tokens_used=report.total_tokens_included,
        tokens_pruned=report.total_tokens_pruned,
        budget_utilization=round(budget_utilization, 1),
        coverage=round(coverage, 1),
        processing_time_ms=round(processing_time_ms, 2)
    )


def run_all_experiments(
    pack_path: Path,
    entry_point: Optional[str] = None
) -> ExperimentSummary:
    """Run experiments across all context window presets."""
    print(f"\n🔬 Context Optimizer Benchmark")
    print(f"{'='*60}")
    
    # Load pack data
    print(f"📦 Loading pack from: {pack_path}")
    data = load_pack_data(pack_path)
    import_graph = data["import_graph"]
    token_map = data["token_map"]
    name_registry = data["name_registry"]
    
    # Get all candidate files
    all_files = list(set(import_graph.keys()) | set(token_map.keys()))
    total_files = len(all_files)
    total_tokens = sum(token_map.get(f, 0) for f in all_files)
    
    print(f"📊 Total files: {total_files}")
    print(f"📊 Total tokens: {total_tokens:,}")
    
    # Determine entry point
    if entry_point:
        seed_files = [entry_point]
        print(f"📍 Using entry point: {entry_point}")
    else:
        suggested = suggest_entry_points(import_graph, name_registry)
        if suggested:
            seed_files = [suggested[0]]
            print(f"📍 Auto-selected entry point: {suggested[0]}")
            print(f"   (Other candidates: {suggested[1:4]})")
        else:
            seed_files = []
            print("⚠️  No entry point found, using heuristic selection")
    
    # Get presets and run experiments
    presets = get_context_window_presets()
    results: List[ExperimentResult] = []
    
    print(f"\n{'Model':<20} {'Budget':>12} {'Files':>8} {'Tokens':>12} {'Usage':>8} {'Coverage':>10}")
    print("-" * 72)
    
    for model_name, token_budget in presets.items():
        if model_name == "unlimited":
            # Skip unlimited for meaningful comparison
            continue
        
        result = run_experiment(
            import_graph=import_graph,
            token_map=token_map,
            candidate_files=all_files,
            seed_files=seed_files,
            token_budget=token_budget,
            model_name=model_name
        )
        results.append(result)
        
        print(f"{model_name:<20} {token_budget:>12,} {result.files_included:>8} "
              f"{result.tokens_used:>12,} {result.budget_utilization:>7.1f}% "
              f"{result.coverage:>9.1f}%")
    
    # Add unlimited comparison
    unlimited_result = ExperimentResult(
        token_budget=total_tokens,
        model_name="unlimited",
        files_included=total_files,
        files_pruned=0,
        tokens_used=total_tokens,
        tokens_pruned=0,
        budget_utilization=100.0,
        coverage=100.0,
        processing_time_ms=0
    )
    results.append(unlimited_result)
    
    print("-" * 72)
    print(f"{'unlimited':<20} {total_tokens:>12,} {total_files:>8} "
          f"{total_tokens:>12,} {'100.0':>7}% {'100.0':>9}%")
    
    return ExperimentSummary(
        pack_path=str(pack_path),
        entry_point=entry_point,
        total_files=total_files,
        total_tokens=total_tokens,
        results=results
    )


def print_analysis(summary: ExperimentSummary):
    """Print analysis insights from the experiment results."""
    print(f"\n📈 Analysis Insights")
    print(f"{'='*60}")
    
    # Find the smallest model that covers >80% of files
    for result in summary.results:
        if result.coverage >= 80 and result.model_name != "unlimited":
            print(f"💡 Recommended: {result.model_name} ({result.token_budget:,} tokens)")
            print(f"   - Covers {result.coverage}% of files")
            print(f"   - Uses only {result.budget_utilization}% of budget")
            break
    
    # Calculate efficiency gains
    if len(summary.results) >= 2:
        smallest = summary.results[0]
        largest = summary.results[-2]  # Second to last (before unlimited)
        
        print(f"\n📊 Efficiency comparison:")
        print(f"   - {smallest.model_name}: {smallest.files_included} files in {smallest.tokens_used:,} tokens")
        print(f"   - {largest.model_name}: {largest.files_included} files in {largest.tokens_used:,} tokens")
        
        if largest.tokens_used > smallest.tokens_used and largest.files_included > smallest.files_included:
            additional_files = largest.files_included - smallest.files_included
            additional_tokens = largest.tokens_used - smallest.tokens_used
            tokens_per_file = additional_tokens / additional_files if additional_files > 0 else 0
            print(f"   - Each additional file costs ~{tokens_per_file:,.0f} tokens on average")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the Graph-Knapsack context optimization algorithm"
    )
    parser.add_argument(
        "pack_path",
        type=str,
        help="Path to the pack directory"
    )
    parser.add_argument(
        "--entry-point",
        type=str,
        default=None,
        help="Entry point file path (optional, will auto-detect if not provided)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (optional)"
    )
    
    args = parser.parse_args()
    pack_path = Path(args.pack_path)
    
    if not pack_path.exists():
        print(f"❌ Pack path not found: {pack_path}")
        sys.exit(1)
    
    # Run experiments
    summary = run_all_experiments(pack_path, args.entry_point)
    
    # Print analysis
    print_analysis(summary)
    
    # Save results if output path provided
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
    
    print(f"\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
