"""
Test the updated HybridFileScorer on httpx benchmark.
"""

import sys
sys.path.insert(0, r'E:\SummerProjects\reposynth\packages\python-orchestrator')

import torch
import numpy as np
from typing import List, Dict

# Import the updated module
from orchestrator.retrieval.learned_query import (
    create_learned_query_module,
    HybridFileScorer,
)

# Benchmark
BENCHMARK = [
    {"id": "httpx_001", "query": "How do I make a simple GET request with httpx?",
     "ground_truth": ["httpx/_api.py", "httpx/_client.py"],
     "tokens": ["get", "request", "client"]},
    {"id": "httpx_002", "query": "How do I send POST data with httpx?",
     "ground_truth": ["httpx/_api.py", "httpx/_client.py"],
     "tokens": ["post", "data", "json"]},
    {"id": "httpx_003", "query": "How do I use httpx with async/await?",
     "ground_truth": ["httpx/_client.py"],
     "tokens": ["async", "await", "AsyncClient"]},
    {"id": "httpx_004", "query": "How do I set custom headers in httpx?",
     "ground_truth": ["httpx/_client.py", "httpx/_models.py"],
     "tokens": ["headers", "Headers"]},
    {"id": "httpx_005", "query": "How do I handle timeouts in httpx?",
     "ground_truth": ["httpx/_config.py", "httpx/_client.py"],
     "tokens": ["timeout", "Timeout"]},
    {"id": "httpx_006", "query": "How does httpx handle cookies?",
     "ground_truth": ["httpx/_client.py", "httpx/_models.py"],
     "tokens": ["cookies", "Cookies"]},
    {"id": "httpx_007", "query": "How do I handle authentication in httpx?",
     "ground_truth": ["httpx/_auth.py", "httpx/_client.py"],
     "tokens": ["auth", "authentication", "BasicAuth"]},
    {"id": "httpx_008", "query": "How does httpx handle redirects?",
     "ground_truth": ["httpx/_client.py"],
     "tokens": ["redirect", "follow_redirects"]},
    {"id": "httpx_009", "query": "How do I upload files with httpx?",
     "ground_truth": ["httpx/_content.py", "httpx/_client.py"],
     "tokens": ["files", "upload", "multipart"]},
    {"id": "httpx_010", "query": "How does httpx handle HTTP/2?",
     "ground_truth": ["httpx/_transports/default.py", "httpx/_client.py"],
     "tokens": ["http2", "HTTP/2"]},
]

# Simulated file list and contents
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

FILE_CONTENTS = {
    "httpx/_api.py": """
\"\"\"Top-level API for making HTTP requests.\"\"\"

def get(url, **kwargs):
    \"\"\"Send a GET request.\"\"\"
    return request("GET", url, **kwargs)

def post(url, data=None, json=None, **kwargs):
    \"\"\"Send a POST request.\"\"\"
    return request("POST", url, data=data, json=json, **kwargs)

def put(url, data=None, **kwargs):
    return request("PUT", url, data=data, **kwargs)

def delete(url, **kwargs):
    return request("DELETE", url, **kwargs)

def request(method, url, **kwargs):
    \"\"\"Construct and send a request.\"\"\"
    with Client() as client:
        return client.request(method, url, **kwargs)
""",
    "httpx/_auth.py": """
\"\"\"Authentication handlers for HTTP requests.\"\"\"

class Auth:
    \"\"\"Base class for authentication.\"\"\"
    pass

class BasicAuth(Auth):
    \"\"\"HTTP Basic authentication.\"\"\"
    def __init__(self, username, password):
        self.username = username
        self.password = password

class DigestAuth(Auth):
    \"\"\"HTTP Digest authentication.\"\"\"
    pass

class NetRCAuth(Auth):
    \"\"\"Use credentials from .netrc file.\"\"\"
    pass
""",
    "httpx/_client.py": """
\"\"\"HTTP client for making requests with session management.\"\"\"

class Client:
    \"\"\"Synchronous HTTP client with connection pooling.\"\"\"

    def __init__(self, auth=None, cookies=None, headers=None,
                 timeout=None, follow_redirects=True):
        self.auth = auth
        self.cookies = cookies or Cookies()
        self.headers = headers or Headers()
        self.timeout = timeout
        self.follow_redirects = follow_redirects

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def request(self, method, url, **kwargs):
        \"\"\"Send a request.\"\"\"
        pass

class AsyncClient:
    \"\"\"Asynchronous HTTP client for async/await usage.\"\"\"

    async def __aenter__(self):
        return self

    async def get(self, url, **kwargs):
        return await self.request("GET", url, **kwargs)

    async def request(self, method, url, **kwargs):
        \"\"\"Send an async request.\"\"\"
        pass
""",
    "httpx/_config.py": """
\"\"\"Configuration classes for httpx.\"\"\"

class Timeout:
    \"\"\"Timeout configuration for requests.\"\"\"

    def __init__(self, timeout=5.0, connect=None, read=None, write=None):
        self.timeout = timeout
        self.connect = connect or timeout
        self.read = read or timeout
        self.write = write or timeout

class Limits:
    \"\"\"Connection pool limits.\"\"\"

    def __init__(self, max_connections=100, max_keepalive=20):
        self.max_connections = max_connections
        self.max_keepalive = max_keepalive

class Proxy:
    \"\"\"Proxy configuration.\"\"\"
    pass
""",
    "httpx/_content.py": """
\"\"\"Request content encoding and file uploads.\"\"\"

def encode_content(data=None, files=None, json=None):
    \"\"\"Encode request content.\"\"\"
    pass

class MultipartEncoder:
    \"\"\"Encode multipart form data with file uploads.\"\"\"

    def __init__(self, data=None, files=None):
        self.data = data
        self.files = files

    def encode(self):
        \"\"\"Encode the multipart data.\"\"\"
        pass
""",
    "httpx/_decoders.py": """
\"\"\"Response content decoders.\"\"\"

class Decoder:
    pass

class GZipDecoder(Decoder):
    \"\"\"Decode gzip-compressed responses.\"\"\"
    pass

class DeflateDecoder(Decoder):
    \"\"\"Decode deflate-compressed responses.\"\"\"
    pass

class BrotliDecoder(Decoder):
    \"\"\"Decode brotli-compressed responses.\"\"\"
    pass
""",
    "httpx/_models.py": """
\"\"\"HTTP data models for requests and responses.\"\"\"

class Request:
    \"\"\"An HTTP request.\"\"\"

    def __init__(self, method, url, headers=None, content=None):
        self.method = method
        self.url = url
        self.headers = headers or Headers()
        self.content = content

class Response:
    \"\"\"An HTTP response.\"\"\"

    def __init__(self, status_code, headers=None, content=None):
        self.status_code = status_code
        self.headers = headers or Headers()
        self.content = content

class Headers:
    \"\"\"HTTP headers container.\"\"\"
    pass

class Cookies:
    \"\"\"HTTP cookies container.\"\"\"
    pass

class URL:
    \"\"\"URL representation.\"\"\"
    pass
""",
    "httpx/_transports/base.py": """
\"\"\"Base transport interface.\"\"\"

class BaseTransport:
    \"\"\"Base class for HTTP transports.\"\"\"

    def handle_request(self, request):
        raise NotImplementedError

class AsyncBaseTransport:
    \"\"\"Base class for async HTTP transports.\"\"\"

    async def handle_async_request(self, request):
        raise NotImplementedError
""",
    "httpx/_transports/default.py": """
\"\"\"Default HTTP transport with HTTP/1.1 and HTTP/2 support.\"\"\"

class HTTPTransport(BaseTransport):
    \"\"\"Synchronous HTTP transport.\"\"\"

    def __init__(self, http2=False):
        self.http2 = http2

    def handle_request(self, request):
        \"\"\"Handle HTTP/1.1 or HTTP/2 request.\"\"\"
        pass

class AsyncHTTPTransport(AsyncBaseTransport):
    \"\"\"Asynchronous HTTP transport with HTTP/2 support.\"\"\"

    def __init__(self, http2=False):
        self.http2 = http2
""",
    "httpx/_transports/mock.py": """
\"\"\"Mock transport for testing.\"\"\"

class MockTransport(BaseTransport):
    \"\"\"Transport that returns mock responses for testing.\"\"\"

    def __init__(self, handler):
        self.handler = handler

    def handle_request(self, request):
        return self.handler(request)
""",
    "httpx/_urlparse.py": """
\"\"\"URL parsing utilities.\"\"\"

def parse_url(url):
    \"\"\"Parse a URL string into components.\"\"\"
    pass

def unquote(string):
    pass

def quote(string):
    pass
""",
    "httpx/_urls.py": """
\"\"\"URL class and utilities.\"\"\"

class URL:
    \"\"\"URL with query parameter support.\"\"\"

    def __init__(self, url, params=None):
        self.url = url
        self.params = params

    def copy_with(self, **kwargs):
        pass
""",
    "httpx/_utils.py": """
\"\"\"Utility functions.\"\"\"

def normalize_header_key(key):
    pass

def normalize_header_value(value):
    pass
""",
}


def compute_f1(predicted: List[str], ground_truth: List[str]) -> float:
    if not predicted or not ground_truth:
        return 0.0
    pred_set = set(predicted)
    gt_set = set(ground_truth)
    hits = len(pred_set & gt_set)
    precision = hits / len(pred_set)
    recall = hits / len(gt_set)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def test_hybrid_module():
    """Test the full LearnedQueryModule with hybrid scoring."""
    print("=" * 70)
    print("TEST: LearnedQueryModule with HybridFileScorer")
    print("=" * 70)

    # Create module with hybrid scoring (default)
    module = create_learned_query_module()
    print(f"Scoring method: {module.scoring_method}")

    # Set files with content
    module.set_available_files(FILE_LIST, FILE_CONTENTS)
    print(f"Files loaded: {len(FILE_LIST)}")

    results = []

    for ex in BENCHMARK:
        # Call module
        result = module(
            confused_tokens=ex['tokens'],
            confused_probs=[0.3, 0.25, 0.2][:len(ex['tokens'])],
            original_query=ex['query'],
            generated_context="",
        )

        f1 = compute_f1(result.matched_files, ex['ground_truth'])
        results.append({
            'id': ex['id'],
            'predicted': result.matched_files,
            'ground_truth': ex['ground_truth'],
            'f1': f1,
            'confidence': result.confidence,
        })

        print(f"\n[{ex['id']}] {ex['query'][:45]}...")
        print(f"  Ground truth: {ex['ground_truth']}")
        print(f"  Predicted:    {result.matched_files}")
        print(f"  F1: {f1:.2f}, Confidence: {result.confidence:.2f}")

    avg_f1 = np.mean([r['f1'] for r in results])
    print(f"\n{'=' * 70}")
    print(f"AVERAGE F1: {avg_f1:.3f}")
    print(f"{'=' * 70}")

    return results, avg_f1


def test_standalone_hybrid_scorer():
    """Test the HybridFileScorer directly."""
    print("\n" + "=" * 70)
    print("TEST: Standalone HybridFileScorer")
    print("=" * 70)

    scorer = HybridFileScorer(
        semantic_weight=0.5,
        keyword_weight=0.4,
        exact_match_boost=0.1,
    )

    scorer.set_files(FILE_LIST, FILE_CONTENTS)
    print(f"Files loaded: {len(FILE_LIST)}")

    results = []

    for ex in BENCHMARK:
        # Create a simple query embedding (just for testing)
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        query_text = f"{ex['query']} {' '.join(ex['tokens'])}"
        query_emb = torch.tensor(encoder.encode([query_text])[0])

        matched, scores = scorer.score_files(
            query_emb=query_emb,
            query_tokens=ex['tokens'],
            query_text=ex['query'],
            top_k=3,
        )

        f1 = compute_f1(matched, ex['ground_truth'])
        results.append({
            'id': ex['id'],
            'predicted': matched,
            'ground_truth': ex['ground_truth'],
            'f1': f1,
        })

        print(f"\n[{ex['id']}] {ex['query'][:45]}...")
        print(f"  Ground truth: {ex['ground_truth']}")
        print(f"  Predicted:    {matched}")
        print(f"  F1: {f1:.2f}")

    avg_f1 = np.mean([r['f1'] for r in results])
    print(f"\n{'=' * 70}")
    print(f"AVERAGE F1: {avg_f1:.3f}")
    print(f"{'=' * 70}")

    return results, avg_f1


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING UPDATED HYBRID SCORING")
    print("=" * 70)

    # Test standalone scorer
    standalone_results, standalone_f1 = test_standalone_hybrid_scorer()

    # Test full module
    module_results, module_f1 = test_hybrid_module()

    # Summary comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"Previous learned scorer (bilinear):  F1 = 0.170")
    print(f"Standalone HybridFileScorer:         F1 = {standalone_f1:.3f}")
    print(f"LearnedQueryModule (hybrid):         F1 = {module_f1:.3f}")
    print(f"Heuristic baseline:                  F1 = 0.620")
    print(f"Heuristic (end-to-end):              F1 = 0.720")

    improvement = (module_f1 - 0.170) / 0.170 * 100
    print(f"\n>>> Hybrid is {improvement:.1f}% better than learned bilinear!")

    gap_to_heuristic = 0.620 - module_f1
    print(f">>> Gap to heuristic: {gap_to_heuristic:.3f}")
