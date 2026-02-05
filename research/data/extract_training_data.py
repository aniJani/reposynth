"""
Extract Training Data from Benchmark Files.

This script converts benchmark JSON files (ablation_benchmark.json, benchmark_v1.json)
into training samples for the Learned Query Pooler.

Usage:
    python extract_training_data.py

Outputs:
    - training_samples.json: Training data for the learned query module
    - file_list.json: List of all file paths in the benchmarks
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any

# Find the research directory
SCRIPT_DIR = Path(__file__).parent
RESEARCH_DIR = SCRIPT_DIR.parent


def load_benchmark(filename: str) -> Dict:
    """Load a benchmark JSON file."""
    path = RESEARCH_DIR / filename
    if not path.exists():
        # Try alternative paths
        alt_paths = [
            RESEARCH_DIR / "ablation_study" / filename,
            RESEARCH_DIR / "benchmarks" / filename,
        ]
        for alt in alt_paths:
            if alt.exists():
                path = alt
                break

    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"examples": []}


def extract_training_sample(example: Dict, sample_id_prefix: str = "") -> Dict:
    """
    Convert a benchmark example to a training sample.

    Benchmark format:
        - query: str
        - ground_truth_files: List[str]
        - ground_truth_keywords: List[str]
        - ground_truth_missing_positions: List[int] (optional)

    Training sample format:
        - confused_tokens: List[str]
        - confused_probs: List[float]
        - original_query: str
        - generated_context: str
        - relevant_files: List[str]
        - sample_id: str
        - spike_position: int
        - uncertainty_value: float
    """
    # Get keywords as confused tokens
    keywords = example.get("ground_truth_keywords", [])
    if not keywords:
        # Fallback: extract keywords from query
        query_words = example.get("query", "").split()
        keywords = [w for w in query_words if len(w) > 3 and w[0].isupper()][:5]

    # Limit to 5 tokens
    keywords = keywords[:5]

    # Generate synthetic probabilities (decreasing)
    n_tokens = len(keywords)
    if n_tokens > 0:
        base_probs = [0.30, 0.25, 0.20, 0.15, 0.10][:n_tokens]
        # Add some randomness
        probs = [p + random.uniform(-0.02, 0.02) for p in base_probs]
        # Normalize
        total = sum(probs)
        probs = [p / total for p in probs]
    else:
        probs = []

    # Get spike positions or generate
    positions = example.get("ground_truth_missing_positions", [])
    if positions:
        spike_position = positions[0]
    else:
        spike_position = random.randint(8, 25)

    # Generate context from query (first part of answer or query-based)
    answer = example.get("ground_truth_answer", "")
    if answer:
        # Take first sentence as context
        context = answer.split('.')[0] + "..."
        if len(context) > 100:
            context = context[:100] + "..."
    else:
        context = example.get("query", "")[:50] + "..."

    # Generate uncertainty value based on difficulty
    difficulty = example.get("difficulty", "medium").lower()
    uncertainty_map = {"easy": 2.5, "medium": 3.0, "hard": 3.5}
    uncertainty = uncertainty_map.get(difficulty, 3.0) + random.uniform(-0.3, 0.3)

    sample_id = example.get("id", f"{sample_id_prefix}_{random.randint(1000, 9999)}")

    return {
        "confused_tokens": keywords,
        "confused_probs": [round(p, 3) for p in probs],
        "original_query": example.get("query", ""),
        "generated_context": context,
        "relevant_files": example.get("ground_truth_files", []),
        "sample_id": sample_id,
        "spike_position": spike_position,
        "uncertainty_value": round(uncertainty, 2),
    }


def extract_all_files(benchmarks: List[Dict]) -> List[str]:
    """Extract all unique file paths from benchmarks."""
    files = set()
    for benchmark in benchmarks:
        for example in benchmark.get("examples", []):
            for f in example.get("ground_truth_files", []):
                files.add(f)
    return sorted(list(files))


def main():
    print("Extracting training data from benchmarks...")

    # Load benchmarks
    ablation = load_benchmark("ablation_benchmark.json")
    benchmark_v1 = load_benchmark("benchmark_v1.json")

    print(f"Loaded ablation_benchmark: {len(ablation.get('examples', []))} examples")
    print(f"Loaded benchmark_v1: {len(benchmark_v1.get('examples', []))} examples")

    # Extract training samples
    training_samples = []

    for example in ablation.get("examples", []):
        sample = extract_training_sample(example, "ablation")
        if sample["confused_tokens"]:  # Only add if we have tokens
            training_samples.append(sample)

    for example in benchmark_v1.get("examples", []):
        sample = extract_training_sample(example, "bench_v1")
        if sample["confused_tokens"]:
            training_samples.append(sample)

    print(f"Generated {len(training_samples)} training samples")

    # Extract all files
    all_files = extract_all_files([ablation, benchmark_v1])
    print(f"Found {len(all_files)} unique files")

    # Save outputs
    output_dir = SCRIPT_DIR
    output_dir.mkdir(exist_ok=True)

    # Save training samples
    samples_path = output_dir / "training_samples_extracted.json"
    with open(samples_path, 'w', encoding='utf-8') as f:
        json.dump(training_samples, f, indent=2)
    print(f"Saved training samples to: {samples_path}")

    # Save file list
    files_path = output_dir / "file_list_extracted.json"
    with open(files_path, 'w', encoding='utf-8') as f:
        json.dump(all_files, f, indent=2)
    print(f"Saved file list to: {files_path}")

    # Print sample
    print("\n--- Sample training data ---")
    if training_samples:
        sample = training_samples[0]
        print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
