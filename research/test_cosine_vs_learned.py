"""
Quick test: Compare learned bilinear scorer vs simple cosine similarity
on the httpx benchmark.

Hypothesis: Cosine similarity might work better because it doesn't have
learned weights that are domain-specific.
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from typing import List, Dict

# Load results to get the benchmark
with open('week9_v3_fixed_results.json', 'r') as f:
    results = json.load(f)

# Recreate the benchmark and file structure
BENCHMARK = [
    {"id": "httpx_001", "query": "How do I make a simple GET request with httpx?",
     "ground_truth": ["httpx/_api.py", "httpx/_client.py"],
     "keywords": ["get", "request", "client"]},
    {"id": "httpx_002", "query": "How do I send POST data with httpx?",
     "ground_truth": ["httpx/_api.py", "httpx/_client.py"],
     "keywords": ["post", "data", "json"]},
    {"id": "httpx_003", "query": "How do I use httpx with async/await?",
     "ground_truth": ["httpx/_client.py"],
     "keywords": ["async", "await", "AsyncClient"]},
    {"id": "httpx_004", "query": "How do I set custom headers in httpx?",
     "ground_truth": ["httpx/_client.py", "httpx/_models.py"],
     "keywords": ["headers", "Headers"]},
    {"id": "httpx_005", "query": "How do I handle timeouts in httpx?",
     "ground_truth": ["httpx/_config.py", "httpx/_client.py"],
     "keywords": ["timeout", "Timeout"]},
    {"id": "httpx_006", "query": "How does httpx handle cookies?",
     "ground_truth": ["httpx/_client.py", "httpx/_models.py"],
     "keywords": ["cookies", "Cookies"]},
    {"id": "httpx_007", "query": "How do I handle authentication in httpx?",
     "ground_truth": ["httpx/_auth.py", "httpx/_client.py"],
     "keywords": ["auth", "authentication", "BasicAuth"]},
    {"id": "httpx_008", "query": "How does httpx handle redirects?",
     "ground_truth": ["httpx/_client.py"],
     "keywords": ["redirect", "follow_redirects"]},
    {"id": "httpx_009", "query": "How do I upload files with httpx?",
     "ground_truth": ["httpx/_content.py", "httpx/_client.py"],
     "keywords": ["files", "upload", "multipart"]},
    {"id": "httpx_010", "query": "How does httpx handle HTTP/2?",
     "ground_truth": ["httpx/_transports/default.py", "httpx/_client.py"],
     "keywords": ["http2", "HTTP/2"]},
]

# Simulated file list (same as httpx)
FILE_LIST = [
    "httpx/_api.py",
    "httpx/_auth.py",
    "httpx/_client.py",
    "httpx/_config.py",
    "httpx/_content.py",
    "httpx/_decoders.py",
    "httpx/_models.py",
    "httpx/_transports/base.py",
    "httpx/_transports/default.py",
    "httpx/_transports/mock.py",
    "httpx/_urlparse.py",
    "httpx/_urls.py",
    "httpx/_utils.py",
]

# Simulated file content summaries (what ContentAwareFileCache would extract)
FILE_SUMMARIES = {
    "httpx/_api.py": "httpx api module provides get post put delete patch head options request functions for making HTTP requests synchronous API",
    "httpx/_auth.py": "httpx auth authentication module BasicAuth DigestAuth NetRCAuth provides authentication handlers for HTTP requests",
    "httpx/_client.py": "httpx client module Client AsyncClient class for making HTTP requests session cookies headers timeout redirects async await",
    "httpx/_config.py": "httpx config configuration Timeout Limits Proxy settings connection pool timeout configuration",
    "httpx/_content.py": "httpx content module encode_content multipart file upload form data content encoding",
    "httpx/_decoders.py": "httpx decoders module content decoding gzip deflate brotli response decompression",
    "httpx/_models.py": "httpx models Request Response Headers Cookies URL data models for HTTP request response",
    "httpx/_transports/base.py": "httpx transports base BaseTransport AsyncBaseTransport abstract transport interface",
    "httpx/_transports/default.py": "httpx transports default HTTPTransport AsyncHTTPTransport HTTP/1.1 HTTP/2 connection transport",
    "httpx/_transports/mock.py": "httpx transports mock MockTransport for testing mock HTTP responses",
    "httpx/_urlparse.py": "httpx urlparse URL parsing utilities parse URL components",
    "httpx/_urls.py": "httpx urls URL class URL manipulation query parameters",
    "httpx/_utils.py": "httpx utils utility functions helpers",
}

def compute_f1(predicted: List[str], ground_truth: List[str]) -> float:
    """Compute F1 score."""
    if not predicted or not ground_truth:
        return 0.0
    pred_set = set(predicted)
    gt_set = set(ground_truth)
    hits = len(pred_set & gt_set)
    precision = hits / len(pred_set)
    recall = hits / len(gt_set)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def test_cosine_similarity():
    """Test pure cosine similarity matching."""
    print("=" * 70)
    print("TEST: Pure Cosine Similarity (No Learned Weights)")
    print("=" * 70)

    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed files
    file_texts = [FILE_SUMMARIES.get(f, f) for f in FILE_LIST]
    file_embs = encoder.encode(file_texts, convert_to_tensor=True)
    file_embs = F.normalize(file_embs, p=2, dim=-1)

    results = []

    for ex in BENCHMARK:
        # Create query from question + keywords
        query_text = f"{ex['query']} {' '.join(ex['keywords'])}"
        query_emb = encoder.encode([query_text], convert_to_tensor=True)
        query_emb = F.normalize(query_emb, p=2, dim=-1)

        # Cosine similarity
        similarities = torch.mm(query_emb, file_embs.T).squeeze(0)

        # Get top 3
        top_indices = torch.topk(similarities, k=3).indices.tolist()
        predicted = [FILE_LIST[i] for i in top_indices]

        f1 = compute_f1(predicted, ex['ground_truth'])
        results.append({
            'id': ex['id'],
            'predicted': predicted,
            'ground_truth': ex['ground_truth'],
            'f1': f1,
            'top_scores': [float(similarities[i]) for i in top_indices],
        })

        print(f"\n[{ex['id']}] {ex['query'][:40]}...")
        print(f"  Ground truth: {ex['ground_truth']}")
        print(f"  Predicted:    {predicted}")
        print(f"  F1: {f1:.2f}")

    avg_f1 = np.mean([r['f1'] for r in results])
    print(f"\n{'=' * 70}")
    print(f"AVERAGE F1: {avg_f1:.3f}")
    print(f"{'=' * 70}")

    return results, avg_f1


def test_keyword_heuristic():
    """Test keyword matching (like the heuristic baseline)."""
    print("\n" + "=" * 70)
    print("TEST: Keyword Heuristic (Baseline)")
    print("=" * 70)

    # Build file keywords
    file_keywords = {}
    for f in FILE_LIST:
        parts = re.split(r'[/\\._]', f.lower())
        keywords = set(p for p in parts if len(p) >= 2)
        # Add content keywords
        summary = FILE_SUMMARIES.get(f, '')
        keywords.update(w.lower() for w in summary.split() if len(w) >= 3)
        file_keywords[f] = keywords

    results = []

    for ex in BENCHMARK:
        query_keywords = set(w.lower() for w in ex['query'].split() if len(w) >= 2)
        query_keywords.update(k.lower() for k in ex['keywords'])

        scores = {}
        for f, kw in file_keywords.items():
            scores[f] = len(query_keywords & kw)

        sorted_files = sorted(scores.items(), key=lambda x: -x[1])
        predicted = [f for f, _ in sorted_files[:3]]

        f1 = compute_f1(predicted, ex['ground_truth'])
        results.append({
            'id': ex['id'],
            'predicted': predicted,
            'ground_truth': ex['ground_truth'],
            'f1': f1,
        })

        print(f"\n[{ex['id']}] {ex['query'][:40]}...")
        print(f"  Ground truth: {ex['ground_truth']}")
        print(f"  Predicted:    {predicted}")
        print(f"  F1: {f1:.2f}")

    avg_f1 = np.mean([r['f1'] for r in results])
    print(f"\n{'=' * 70}")
    print(f"AVERAGE F1: {avg_f1:.3f}")
    print(f"{'=' * 70}")

    return results, avg_f1


if __name__ == "__main__":
    print("\nComparing scoring methods on httpx benchmark\n")

    # Test both methods
    cosine_results, cosine_f1 = test_cosine_similarity()
    heuristic_results, heuristic_f1 = test_keyword_heuristic()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"Learned Scorer (from results):  F1 = 0.170")
    print(f"Cosine Similarity:              F1 = {cosine_f1:.3f}")
    print(f"Keyword Heuristic:              F1 = {heuristic_f1:.3f}")
    print(f"Heuristic (end-to-end):         F1 = 0.720")

    if cosine_f1 > 0.170:
        print(f"\n>>> Cosine similarity is {(cosine_f1 - 0.170) / 0.170 * 100:.1f}% better than learned scorer!")

    # Per-example comparison
    print("\n" + "-" * 70)
    print("Per-Example Comparison:")
    print("-" * 70)
    print(f"{'Example':<12} | {'Learned':<8} | {'Cosine':<8} | {'Heuristic':<8}")
    print("-" * 50)

    learned_f1s = [0.0, 0.4, 0.0, 0.5, 0.0, 0.4, 0.0, 0.0, 0.0, 0.4]  # From results
    for i, (cos, heur) in enumerate(zip(cosine_results, heuristic_results)):
        print(f"{cos['id']:<12} | {learned_f1s[i]:<8.2f} | {cos['f1']:<8.2f} | {heur['f1']:<8.2f}")
