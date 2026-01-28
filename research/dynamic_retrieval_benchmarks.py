"""
Dynamic Retrieval Benchmark Examples for CCE Query Evolution

These benchmarks test MULTI-HOP retrieval where:
1. Initial query provides partial information
2. Generation discovers new identifiers requiring additional retrieval
3. Correct answer requires files retrieved at DIFFERENT hops

Key differences from single-hop Q&A benchmarks:
- ground_truth_files_ordered: Files in ORDER they should be discovered
- hop_triggers: What identifiers/concepts trigger each hop
- intermediate_discoveries: What the model learns at each hop
- requires_all_hops: Boolean indicating answer needs ALL files
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class DynamicBenchmarkExample:
    """Benchmark example designed for multi-hop dynamic retrieval."""

    id: str
    category: str  # implementation, debugging, extension, integration
    difficulty: str  # medium, hard, expert

    # The task (not a question, but a generation task)
    task: str

    # Files in ORDER they should be retrieved
    ground_truth_files_ordered: List[str]

    # What triggers each hop (identifiers discovered during generation)
    hop_triggers: List[Dict[str, any]]

    # What information is gained at each hop
    intermediate_discoveries: List[str]

    # Final answer that requires information from ALL hops
    ground_truth_answer: str

    # Keywords that might appear in confused tokens
    confused_token_hints: List[str]

    # Does the answer require all files? (for scoring)
    requires_all_hops: bool = True

    # Minimum hops needed for correct answer
    min_hops_required: int = 2


# =============================================================================
# HTTPX DYNAMIC BENCHMARKS
# =============================================================================

HTTPX_DYNAMIC_BENCHMARKS = [

    # -------------------------------------------------------------------------
    # CATEGORY: Implementation Tasks (Code Generation)
    # -------------------------------------------------------------------------

    DynamicBenchmarkExample(
        id='dyn_impl_001',
        category='implementation',
        difficulty='hard',
        task='Implement a retry mechanism for failed httpx requests with exponential backoff',
        ground_truth_files_ordered=[
            'httpx/_client.py',      # Hop 1: Base request handling
            'httpx/_exceptions.py',   # Hop 2: Exception types for retry conditions
            'httpx/_config.py',       # Hop 3: Configuration patterns
        ],
        hop_triggers=[
            {'hop': 1, 'trigger': 'initial_task', 'discovers': ['Client', 'request', 'send']},
            {'hop': 2, 'trigger': 'exception_handling', 'discovers': ['TimeoutException', 'ConnectError', 'HTTPStatusError']},
            {'hop': 3, 'trigger': 'configuration', 'discovers': ['Timeout', 'Limits', 'DEFAULT_TIMEOUT_CONFIG']},
        ],
        intermediate_discoveries=[
            'Hop 1: Client.send() is the core request method',
            'Hop 2: TimeoutException and ConnectError are retryable, HTTPStatusError for 5xx',
            'Hop 3: Configuration follows Timeout/Limits pattern',
        ],
        ground_truth_answer='''Wrap Client.send() with retry logic:
1. Catch TimeoutException, ConnectError, and HTTPStatusError (5xx only)
2. Use exponential backoff: delay = base_delay * (2 ** attempt)
3. Follow httpx config pattern with Retry(max_attempts=3, backoff_factor=0.5)
4. Integrate via transport wrapper or client subclass''',
        confused_token_hints=['retry', 'backoff', 'timeout', 'exception', 'attempts'],
        requires_all_hops=True,
        min_hops_required=3,
    ),

    DynamicBenchmarkExample(
        id='dyn_impl_002',
        category='implementation',
        difficulty='hard',
        task='Add request/response logging middleware to httpx client',
        ground_truth_files_ordered=[
            'httpx/_client.py',           # Hop 1: Client request flow
            'httpx/_transports/base.py',  # Hop 2: Transport interface
            'httpx/_models.py',           # Hop 3: Request/Response models
        ],
        hop_triggers=[
            {'hop': 1, 'trigger': 'initial_task', 'discovers': ['Client', '_transport', 'send']},
            {'hop': 2, 'trigger': 'transport_discovery', 'discovers': ['BaseTransport', 'handle_request']},
            {'hop': 3, 'trigger': 'model_details', 'discovers': ['Request', 'Response', 'headers', 'content']},
        ],
        intermediate_discoveries=[
            'Hop 1: Client uses _transport.handle_request() for actual requests',
            'Hop 2: BaseTransport has handle_request(request) -> Response interface',
            'Hop 3: Request has url, method, headers, content; Response has status_code, headers, stream',
        ],
        ground_truth_answer='''Create a LoggingTransport wrapper:
1. Subclass BaseTransport
2. Wrap handle_request to log Request.method, Request.url, Request.headers
3. Log Response.status_code, Response.headers, timing
4. Pass to Client(transport=LoggingTransport(HTTPTransport()))''',
        confused_token_hints=['transport', 'middleware', 'logging', 'request', 'response'],
        requires_all_hops=True,
        min_hops_required=3,
    ),

    DynamicBenchmarkExample(
        id='dyn_impl_003',
        category='implementation',
        difficulty='expert',
        task='Implement connection pooling with per-host limits for httpx',
        ground_truth_files_ordered=[
            'httpx/_client.py',               # Hop 1: Client connection management
            'httpx/_config.py',               # Hop 2: Limits configuration
            'httpx/_transports/default.py',   # Hop 3: Transport pool implementation
        ],
        hop_triggers=[
            {'hop': 1, 'trigger': 'initial_task', 'discovers': ['Client', 'limits', '_transport']},
            {'hop': 2, 'trigger': 'limits_config', 'discovers': ['Limits', 'max_connections', 'max_keepalive_connections']},
            {'hop': 3, 'trigger': 'transport_pool', 'discovers': ['HTTPTransport', 'httpcore', '_pool']},
        ],
        intermediate_discoveries=[
            'Hop 1: Client accepts limits parameter, passes to transport',
            'Hop 2: Limits(max_connections=100, max_keepalive_connections=20) controls pool',
            'Hop 3: HTTPTransport wraps httpcore which manages actual connection pool',
        ],
        ground_truth_answer='''Extend Limits for per-host control:
1. Add per_host_max_connections to Limits dataclass
2. Modify HTTPTransport to create per-host pools
3. Track connections per host in dict
4. Implement host-based pool selection in handle_request''',
        confused_token_hints=['pool', 'connections', 'limits', 'host', 'transport'],
        requires_all_hops=True,
        min_hops_required=3,
    ),

    # -------------------------------------------------------------------------
    # CATEGORY: Debugging Tasks
    # -------------------------------------------------------------------------

    DynamicBenchmarkExample(
        id='dyn_debug_001',
        category='debugging',
        difficulty='hard',
        task='Debug why requests timeout when using a proxy with httpx',
        ground_truth_files_ordered=[
            'httpx/_exceptions.py',          # Hop 1: Understand timeout exception
            'httpx/_config.py',              # Hop 2: Timeout configuration
            'httpx/_transports/default.py',  # Hop 3: Proxy transport handling
        ],
        hop_triggers=[
            {'hop': 1, 'trigger': 'timeout_error', 'discovers': ['TimeoutException', 'ConnectTimeout', 'ReadTimeout']},
            {'hop': 2, 'trigger': 'timeout_config', 'discovers': ['Timeout', 'connect', 'read', 'pool']},
            {'hop': 3, 'trigger': 'proxy_transport', 'discovers': ['HTTPTransport', 'proxy', '_proxy_url']},
        ],
        intermediate_discoveries=[
            'Hop 1: ConnectTimeout vs ReadTimeout - proxy uses connect phase',
            'Hop 2: Timeout has separate connect/read/pool values',
            'Hop 3: Proxy connections go through extra connect phase',
        ],
        ground_truth_answer='''Proxy timeout issue:
1. Proxy requires TWO connect phases (to proxy, then to target)
2. Default connect timeout may be too short for proxy chain
3. Fix: Timeout(connect=30.0) or separate proxy_connect timeout
4. Check HTTPTransport proxy configuration for tunnel setup''',
        confused_token_hints=['timeout', 'proxy', 'connect', 'tunnel', 'transport'],
        requires_all_hops=True,
        min_hops_required=3,
    ),

    DynamicBenchmarkExample(
        id='dyn_debug_002',
        category='debugging',
        difficulty='hard',
        task='Debug why httpx async requests are slower than expected',
        ground_truth_files_ordered=[
            'httpx/_client.py',              # Hop 1: AsyncClient implementation
            'httpx/_transports/default.py',  # Hop 2: Async transport
            'httpx/_config.py',              # Hop 3: Connection limits
        ],
        hop_triggers=[
            {'hop': 1, 'trigger': 'async_client', 'discovers': ['AsyncClient', 'aclose', '_transport']},
            {'hop': 2, 'trigger': 'async_transport', 'discovers': ['AsyncHTTPTransport', 'httpcore', 'async']},
            {'hop': 3, 'trigger': 'pool_limits', 'discovers': ['Limits', 'max_connections', 'max_keepalive']},
        ],
        intermediate_discoveries=[
            'Hop 1: AsyncClient reuses connections if not closed properly',
            'Hop 2: AsyncHTTPTransport manages connection pool',
            'Hop 3: Default Limits may bottleneck concurrent requests',
        ],
        ground_truth_answer='''Async performance issues:
1. Use context manager: async with httpx.AsyncClient() as client
2. Increase Limits(max_connections=100) for concurrency
3. Enable HTTP/2: AsyncClient(http2=True) for multiplexing
4. Ensure proper await on all requests for parallelism''',
        confused_token_hints=['async', 'await', 'pool', 'connections', 'concurrent'],
        requires_all_hops=True,
        min_hops_required=3,
    ),

    # -------------------------------------------------------------------------
    # CATEGORY: Feature Extension
    # -------------------------------------------------------------------------

    DynamicBenchmarkExample(
        id='dyn_ext_001',
        category='extension',
        difficulty='hard',
        task='Add automatic content decompression for custom encodings in httpx',
        ground_truth_files_ordered=[
            'httpx/_models.py',     # Hop 1: Response content handling
            'httpx/_content.py',    # Hop 2: Content encoding
            'httpx/_decoders.py',   # Hop 3: Decoder implementations
        ],
        hop_triggers=[
            {'hop': 1, 'trigger': 'response_content', 'discovers': ['Response', 'content', 'stream']},
            {'hop': 2, 'trigger': 'content_encoding', 'discovers': ['encode_content', 'ByteStream']},
            {'hop': 3, 'trigger': 'decoders', 'discovers': ['ContentDecoder', 'GZipDecoder', 'DeflateDecoder']},
        ],
        intermediate_discoveries=[
            'Hop 1: Response.content uses stream with decoder chain',
            'Hop 2: Content encoding/decoding flows through ByteStream',
            'Hop 3: Decoders implement decode(data) -> bytes interface',
        ],
        ground_truth_answer='''Add custom decompression:
1. Create CustomDecoder(ContentDecoder) with decode() method
2. Register in SUPPORTED_DECODERS dict
3. Response auto-selects based on Content-Encoding header
4. Chain decoders for multi-encoding support''',
        confused_token_hints=['decoder', 'content', 'encoding', 'gzip', 'stream'],
        requires_all_hops=True,
        min_hops_required=3,
    ),

    DynamicBenchmarkExample(
        id='dyn_ext_002',
        category='extension',
        difficulty='expert',
        task='Implement request signing (HMAC) as httpx authentication',
        ground_truth_files_ordered=[
            'httpx/_auth.py',       # Hop 1: Auth interface
            'httpx/_models.py',     # Hop 2: Request model for signing
            'httpx/_client.py',     # Hop 3: Where auth is applied
        ],
        hop_triggers=[
            {'hop': 1, 'trigger': 'auth_interface', 'discovers': ['Auth', 'auth_flow', 'require_request_body']},
            {'hop': 2, 'trigger': 'request_signing', 'discovers': ['Request', 'url', 'method', 'headers', 'content']},
            {'hop': 3, 'trigger': 'auth_flow', 'discovers': ['Client', '_build_auth', 'request']},
        ],
        intermediate_discoveries=[
            'Hop 1: Auth uses generator-based auth_flow for request/response cycle',
            'Hop 2: Request has all components needed for HMAC: url, method, headers, content',
            'Hop 3: Client calls auth before sending, can modify request',
        ],
        ground_truth_answer='''Implement HMAC auth:
1. Subclass Auth, set require_request_body = True
2. Implement auth_flow generator that yields signed request
3. Compute HMAC-SHA256(secret, method + url + body)
4. Add Authorization header with signature
5. Use: Client(auth=HMACAuth(key, secret))''',
        confused_token_hints=['auth', 'hmac', 'signature', 'header', 'request'],
        requires_all_hops=True,
        min_hops_required=3,
    ),

    # -------------------------------------------------------------------------
    # CATEGORY: Integration Tasks
    # -------------------------------------------------------------------------

    DynamicBenchmarkExample(
        id='dyn_int_001',
        category='integration',
        difficulty='hard',
        task='Integrate httpx with OpenTelemetry for distributed tracing',
        ground_truth_files_ordered=[
            'httpx/_client.py',              # Hop 1: Client request lifecycle
            'httpx/_transports/base.py',     # Hop 2: Transport interface
            'httpx/_models.py',              # Hop 3: Inject trace headers
        ],
        hop_triggers=[
            {'hop': 1, 'trigger': 'client_lifecycle', 'discovers': ['Client', 'send', '_transport']},
            {'hop': 2, 'trigger': 'transport_hooks', 'discovers': ['BaseTransport', 'handle_request']},
            {'hop': 3, 'trigger': 'header_injection', 'discovers': ['Request', 'headers', 'Headers']},
        ],
        intermediate_discoveries=[
            'Hop 1: Client.send() is interception point for tracing',
            'Hop 2: Transport wrapper can add span around handle_request',
            'Hop 3: Request.headers is mutable for trace context injection',
        ],
        ground_truth_answer='''OpenTelemetry integration:
1. Create TracingTransport(BaseTransport) wrapper
2. In handle_request: start span, inject traceparent header
3. Add span attributes: http.method, http.url, http.status_code
4. Use propagator to inject W3C trace context into Request.headers
5. Client(transport=TracingTransport(HTTPTransport()))''',
        confused_token_hints=['tracing', 'span', 'headers', 'transport', 'context'],
        requires_all_hops=True,
        min_hops_required=3,
    ),

    DynamicBenchmarkExample(
        id='dyn_int_002',
        category='integration',
        difficulty='expert',
        task='Implement circuit breaker pattern for httpx requests',
        ground_truth_files_ordered=[
            'httpx/_client.py',       # Hop 1: Client request handling
            'httpx/_exceptions.py',   # Hop 2: Exception types for failure detection
            'httpx/_config.py',       # Hop 3: Configuration patterns
            'httpx/_transports/base.py',  # Hop 4: Transport for clean integration
        ],
        hop_triggers=[
            {'hop': 1, 'trigger': 'initial_task', 'discovers': ['Client', 'send', 'request']},
            {'hop': 2, 'trigger': 'failure_types', 'discovers': ['ConnectError', 'TimeoutException', 'HTTPStatusError']},
            {'hop': 3, 'trigger': 'config_pattern', 'discovers': ['Timeout', 'Limits', 'dataclass']},
            {'hop': 4, 'trigger': 'clean_wrapper', 'discovers': ['BaseTransport', 'handle_request']},
        ],
        intermediate_discoveries=[
            'Hop 1: Client.send() calls transport, good interception point',
            'Hop 2: ConnectError, TimeoutException count as failures; HTTPStatusError 5xx too',
            'Hop 3: Use dataclass pattern for CircuitBreakerConfig',
            'Hop 4: Transport wrapper provides cleanest integration',
        ],
        ground_truth_answer='''Circuit breaker implementation:
1. Create CircuitBreakerConfig(failure_threshold=5, reset_timeout=30)
2. States: CLOSED (normal), OPEN (failing), HALF_OPEN (testing)
3. Create CircuitBreakerTransport(BaseTransport) wrapper
4. Track failures per host, open circuit on threshold
5. Raise CircuitOpenError when open, test periodically in half-open''',
        confused_token_hints=['circuit', 'breaker', 'failure', 'threshold', 'transport'],
        requires_all_hops=True,
        min_hops_required=4,
    ),

    # -------------------------------------------------------------------------
    # CATEGORY: Complex Multi-File Tasks
    # -------------------------------------------------------------------------

    DynamicBenchmarkExample(
        id='dyn_complex_001',
        category='complex',
        difficulty='expert',
        task='Implement request caching with ETags and conditional requests for httpx',
        ground_truth_files_ordered=[
            'httpx/_client.py',       # Hop 1: Request/Response flow
            'httpx/_models.py',       # Hop 2: Response headers, ETag handling
            'httpx/_transports/base.py',  # Hop 3: Transport for cache layer
            'httpx/_content.py',      # Hop 4: Content storage
        ],
        hop_triggers=[
            {'hop': 1, 'trigger': 'initial_task', 'discovers': ['Client', 'send', 'Response']},
            {'hop': 2, 'trigger': 'etag_headers', 'discovers': ['Response', 'headers', 'ETag', 'Last-Modified']},
            {'hop': 3, 'trigger': 'cache_layer', 'discovers': ['BaseTransport', 'handle_request']},
            {'hop': 4, 'trigger': 'content_storage', 'discovers': ['ByteStream', 'content', 'stream']},
        ],
        intermediate_discoveries=[
            'Hop 1: Client request flow, Response returned from transport',
            'Hop 2: Response.headers contains ETag, Last-Modified for caching',
            'Hop 3: Transport wrapper is ideal cache interception point',
            'Hop 4: Cache needs to store response content as bytes',
        ],
        ground_truth_answer='''ETag caching implementation:
1. CachingTransport wraps BaseTransport with dict cache
2. On request: check cache for URL, add If-None-Match: <etag> header
3. On 304 response: return cached response
4. On 200: store response with ETag, return response
5. Cache key: (method, url), value: (etag, response_bytes)''',
        confused_token_hints=['cache', 'etag', 'conditional', 'transport', 'headers'],
        requires_all_hops=True,
        min_hops_required=4,
    ),
]


# =============================================================================
# CERBERUS DYNAMIC BENCHMARKS
# =============================================================================

CERBERUS_DYNAMIC_BENCHMARKS = [
    DynamicBenchmarkExample(
        id='dyn_cerb_001',
        category='implementation',
        difficulty='hard',
        task='Implement a custom async validation rule for Cerberus that validates against external API',
        ground_truth_files_ordered=[
            'cerberus/validator.py',  # Hop 1: Base validator and _validate pattern
            'cerberus/schema.py',     # Hop 2: Schema rule registration
            'cerberus/errors.py',     # Hop 3: Error handling for async failures
        ],
        hop_triggers=[
            {'hop': 1, 'trigger': 'validation_rule', 'discovers': ['Validator', '_validate_', 'constraint']},
            {'hop': 2, 'trigger': 'rule_registry', 'discovers': ['SchemaRegistry', 'rules', 'types']},
            {'hop': 3, 'trigger': 'error_handling', 'discovers': ['ValidationError', 'ErrorHandler', '_error']},
        ],
        intermediate_discoveries=[
            'Hop 1: Rules are _validate_<name>(self, constraint, field, value)',
            'Hop 2: Schema rules registered in SchemaRegistry, discoverable',
            'Hop 3: Errors added via _error(field, message) method',
        ],
        ground_truth_answer='''Async validation with external API:
1. Create _validate_async_api(self, constraint, field, value)
2. Store async results in validator state for deferred check
3. Add _error() call for API failures
4. Consider custom Validator subclass with async validate()''',
        confused_token_hints=['validate', 'async', 'rule', 'constraint', 'error'],
        requires_all_hops=True,
        min_hops_required=3,
    ),

    DynamicBenchmarkExample(
        id='dyn_cerb_002',
        category='debugging',
        difficulty='hard',
        task='Debug why nested schema validation fails silently in Cerberus',
        ground_truth_files_ordered=[
            'cerberus/validator.py',  # Hop 1: Nested validation flow
            'cerberus/schema.py',     # Hop 2: Schema compilation
            'cerberus/errors.py',     # Hop 3: Error collection for nested
        ],
        hop_triggers=[
            {'hop': 1, 'trigger': 'nested_validation', 'discovers': ['_validate_schema', 'child_validator']},
            {'hop': 2, 'trigger': 'schema_compile', 'discovers': ['Schema', 'expand', 'validate_schema']},
            {'hop': 3, 'trigger': 'nested_errors', 'discovers': ['ErrorTree', '_errors', 'child_errors']},
        ],
        intermediate_discoveries=[
            'Hop 1: Nested schemas use child_validator with separate error collection',
            'Hop 2: Schema must be expanded before nested validation',
            'Hop 3: Nested errors stored in ErrorTree, may not surface without error_handler',
        ],
        ground_truth_answer='''Nested validation debug:
1. Child validators have separate _errors dict
2. Errors merged post-validation via error handler
3. Check allow_unknown=True not swallowing errors
4. Use verbose error handler to see nested error tree''',
        confused_token_hints=['nested', 'schema', 'child', 'errors', 'validate'],
        requires_all_hops=True,
        min_hops_required=3,
    ),
]


# =============================================================================
# TYPER DYNAMIC BENCHMARKS
# =============================================================================

TYPER_DYNAMIC_BENCHMARKS = [
    DynamicBenchmarkExample(
        id='dyn_typer_001',
        category='implementation',
        difficulty='hard',
        task='Implement custom parameter type with validation for Typer CLI',
        ground_truth_files_ordered=[
            'typer/main.py',     # Hop 1: Typer parameter handling
            'typer/params.py',   # Hop 2: Parameter definitions
            'typer/core.py',     # Hop 3: Click integration
        ],
        hop_triggers=[
            {'hop': 1, 'trigger': 'parameter_type', 'discovers': ['Typer', 'command', 'Argument']},
            {'hop': 2, 'trigger': 'param_definition', 'discovers': ['Argument', 'Option', 'ParamMeta']},
            {'hop': 3, 'trigger': 'click_type', 'discovers': ['click', 'ParamType', 'convert']},
        ],
        intermediate_discoveries=[
            'Hop 1: Typer wraps Click, parameters via function annotations',
            'Hop 2: Argument/Option pass click_type to Click',
            'Hop 3: Custom types need click.ParamType with convert() method',
        ],
        ground_truth_answer='''Custom Typer parameter type:
1. Subclass click.ParamType with convert(self, value, param, ctx)
2. Implement validation in convert(), raise BadParameter on failure
3. Use in Typer: def cmd(val: Annotated[MyType, typer.Argument(click_type=MyParamType())])
4. Type hint with custom class for IDE support''',
        confused_token_hints=['type', 'param', 'convert', 'click', 'validate'],
        requires_all_hops=True,
        min_hops_required=3,
    ),
]


# =============================================================================
# COMBINED BENCHMARK SELECTION
# =============================================================================

def get_dynamic_benchmarks(codebase: str = 'httpx') -> List[DynamicBenchmarkExample]:
    """Get dynamic benchmarks for specified codebase."""
    benchmarks = {
        'httpx': HTTPX_DYNAMIC_BENCHMARKS,
        'cerberus': CERBERUS_DYNAMIC_BENCHMARKS,
        'typer': TYPER_DYNAMIC_BENCHMARKS,
    }
    return benchmarks.get(codebase.lower(), HTTPX_DYNAMIC_BENCHMARKS)


def score_multi_hop_retrieval(
    retrieved_files: List[str],
    example: DynamicBenchmarkExample
) -> Dict[str, float]:
    """
    Score retrieval against multi-hop ground truth.

    Returns:
        dict with:
        - recall: What % of required files were retrieved
        - precision: What % of retrieved files were relevant
        - order_score: How well retrieval order matches expected
        - hop_coverage: What % of required hops were completed
    """
    ground_truth = set(example.ground_truth_files_ordered)
    retrieved = set(retrieved_files)

    # Basic metrics
    true_positives = ground_truth & retrieved
    recall = len(true_positives) / len(ground_truth) if ground_truth else 0
    precision = len(true_positives) / len(retrieved) if retrieved else 0

    # Order score (how well retrieval order matches expected)
    order_score = 0.0
    if retrieved_files:
        gt_order = {f: i for i, f in enumerate(example.ground_truth_files_ordered)}
        matched_order = []
        for f in retrieved_files:
            if f in gt_order:
                matched_order.append(gt_order[f])

        if len(matched_order) >= 2:
            # Check if in ascending order
            in_order = sum(1 for i in range(len(matched_order)-1)
                          if matched_order[i] < matched_order[i+1])
            order_score = in_order / (len(matched_order) - 1)
        elif len(matched_order) == 1:
            order_score = 1.0 if matched_order[0] == 0 else 0.5

    # Hop coverage
    hop_coverage = len(true_positives) / example.min_hops_required

    return {
        'recall': recall,
        'precision': precision,
        'f1': 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0,
        'order_score': order_score,
        'hop_coverage': min(1.0, hop_coverage),
        'complete': recall == 1.0,  # All required files retrieved
    }


# =============================================================================
# NOTEBOOK INTEGRATION
# =============================================================================

def create_notebook_benchmark_cell(codebase_option: int = 2) -> str:
    """Generate notebook cell code for dynamic benchmarks."""

    return '''# Cell B: Dynamic Retrieval Benchmark Examples
#
# These benchmarks test MULTI-HOP retrieval where:
# 1. Initial query provides partial information
# 2. Generation discovers new identifiers requiring additional retrieval
# 3. Correct answer requires files retrieved at DIFFERENT hops

from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class DynamicBenchmarkExample:
    """Benchmark example designed for multi-hop dynamic retrieval."""
    id: str
    category: str
    difficulty: str
    task: str
    ground_truth_files_ordered: List[str]  # Files in ORDER they should be retrieved
    hop_triggers: List[Dict]               # What triggers each hop
    intermediate_discoveries: List[str]    # What's learned at each hop
    ground_truth_answer: str
    confused_token_hints: List[str]
    requires_all_hops: bool = True
    min_hops_required: int = 2


# Import dynamic benchmarks
import sys
sys.path.insert(0, '.')
from dynamic_retrieval_benchmarks import (
    get_dynamic_benchmarks,
    score_multi_hop_retrieval,
    HTTPX_DYNAMIC_BENCHMARKS,
    CERBERUS_DYNAMIC_BENCHMARKS,
    TYPER_DYNAMIC_BENCHMARKS
)

# Select benchmarks based on codebase
if CODEBASE_OPTION == 1:
    dynamic_benchmarks = CERBERUS_DYNAMIC_BENCHMARKS
elif CODEBASE_OPTION == 2:
    dynamic_benchmarks = HTTPX_DYNAMIC_BENCHMARKS
elif CODEBASE_OPTION == 3:
    dynamic_benchmarks = TYPER_DYNAMIC_BENCHMARKS

print("=" * 70)
print("DYNAMIC RETRIEVAL BENCHMARKS")
print("=" * 70)
print(f"Total examples: {len(dynamic_benchmarks)}")
print(f"\\nBy Category:")
from collections import Counter
cat_counts = Counter(ex.category for ex in dynamic_benchmarks)
for cat, count in sorted(cat_counts.items()):
    print(f"  {cat}: {count}")
print(f"\\nBy Difficulty:")
diff_counts = Counter(ex.difficulty for ex in dynamic_benchmarks)
for diff, count in sorted(diff_counts.items()):
    print(f"  {diff}: {count}")
print(f"\\nBy Min Hops Required:")
hop_counts = Counter(ex.min_hops_required for ex in dynamic_benchmarks)
for hops, count in sorted(hop_counts.items()):
    print(f"  {hops} hops: {count}")
'''


if __name__ == '__main__':
    # Print summary
    print("=" * 70)
    print("DYNAMIC RETRIEVAL BENCHMARKS SUMMARY")
    print("=" * 70)

    for name, benchmarks in [
        ('httpx', HTTPX_DYNAMIC_BENCHMARKS),
        ('cerberus', CERBERUS_DYNAMIC_BENCHMARKS),
        ('typer', TYPER_DYNAMIC_BENCHMARKS),
    ]:
        print(f"\n{name.upper()}: {len(benchmarks)} examples")
        for ex in benchmarks:
            print(f"  [{ex.id}] {ex.category}/{ex.difficulty}: {ex.task[:50]}...")
            print(f"      Files: {' -> '.join(f.split('/')[-1] for f in ex.ground_truth_files_ordered)}")
            print(f"      Min hops: {ex.min_hops_required}")
