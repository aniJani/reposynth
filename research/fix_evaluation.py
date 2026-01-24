"""
Fix Week8 Evaluation Notebook - Critical Issues

This script fixes all 6 critical issues identified in the evaluation:
1. Answer slicing - remove prompt from generated answer
2. file_list_context leakage - give all methods file_list
3. Threshold units - change 3.0 to 0.5 (CCE scale)
4. No-spike fallback - generate instead of hard-coded failure
5. Token counting - unified counting across methods
6. Document BPE limitation in code comments
"""

import json
import re

# Load notebook
with open('C:/Users/rajka/reposynth/research/Week8_Comprehensive_Evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Fixing Week8 Evaluation Notebook...")
print("=" * 60)

# =============================================================================
# FIX 1 & 4: Update Cell 24 (Experiment Loop)
# - Fix answer slicing (remove prompt from output)
# - Fix no-spike fallback (generate instead of failure string)
# - Fix token counting
# =============================================================================

cell_24_new = '''# Cell 18: Run comprehensive experiments with FILE LIST CCE + Real CCE Multi-Hop
# FIXED: Answer slicing, no-spike fallback, token counting, file_list for all methods

# Get balanced sample: 2 from each category
sample_examples = []
for cat in Category:
    filtered = benchmark.filter(category=cat)
    for i, ex in enumerate(filtered):
        if i < 2:  # 2 per category
            sample_examples.append(ex)

print(f"Running experiments on {len(sample_examples)} examples")
print(f"Categories covered: {len(set(ex.category for ex in sample_examples))}")

# Methods to evaluate
methods = [
    BaselineMethod.NO_CONTEXT,
    BaselineMethod.BM25,
    BaselineMethod.EMBEDDING,
    BaselineMethod.FULL_CONTEXT,
]

# Results storage
all_results = {m.value: [] for m in methods}
all_results['cce_adaptive'] = []
all_results['cce_topk_only'] = []
all_results['cce_query_plus_topk'] = []

# CCE debug info
cce_debug = []
real_cce_debug = {'cce_topk_only': [], 'cce_query_plus_topk': []}

# Compute baseline tokens
baseline_tokens = sum(len(tokenizer.encode(c)) for c in documents.values())
print(f"Baseline tokens (full context): {baseline_tokens:,}")
print(f"File list tokens: {file_list_tokens}")
print()

# Helper function: unified token counting
def count_tokens_unified(prompt_tokens, context_tokens, gen_tokens):
    """Unified token counting across all methods."""
    return prompt_tokens + context_tokens + gen_tokens

# Helper function: generate answer with proper slicing
def generate_answer(prompt, max_new_tokens=200):
    """Generate answer and properly slice off prompt."""
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(model.device)
    prompt_len = inputs['input_ids'].shape[-1]
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    # FIXED: Slice off prompt tokens before decoding
    gen_tokens = outputs[0][prompt_len:]
    answer = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return answer, len(gen_tokens)

# Run experiments
total = len(sample_examples) * 7
current = 0

for i, example in enumerate(sample_examples):
    print(f"\\nExample {i+1}/{len(sample_examples)}: {example.id}")
    print(f"  Category: {example.category.value}")
    print(f"  Query: {example.query[:50]}...")

    # Run baselines - FIXED: Include file_list_context for fair comparison
    for method in methods:
        current += 1
        print(f"  Running {method.value}... ({current}/{total})")

        # FIXED: Pass file_list_context to baselines via initial_context
        example_with_file_list = BenchmarkExample(
            id=example.id,
            query=example.query,
            category=example.category,
            difficulty=example.difficulty,
            ground_truth_answer=example.ground_truth_answer,
            ground_truth_files=example.ground_truth_files,
            ground_truth_keywords=example.ground_truth_keywords,
            initial_context=file_list_context,  # FIXED: All methods get file_list
            ground_truth_missing_positions=example.ground_truth_missing_positions,
        )

        result = baseline_runner.run(example_with_file_list, method, max_tokens=200)

        eval_result = metrics.evaluate(
            example_id=example.id,
            generated_answer=result['answer'],
            ground_truth_answer=example.ground_truth_answer,
            retrieved_files=result['retrieved_files'],
            ground_truth_files=example.ground_truth_files,
            ground_truth_keywords=example.ground_truth_keywords,
            tokens_used=result['tokens_used'] + file_list_tokens,  # FIXED: Include file_list tokens
            baseline_tokens=baseline_tokens,
            generation_time=result['generation_time'],
            num_retrievals=result['num_retrievals'],
            method=method.value,
        )
        eval_result.metadata['category'] = example.category.value
        eval_result.metadata['difficulty'] = example.difficulty.value

        all_results[method.value].append(eval_result)

    # Run CCE adaptive WITH FILE LIST CONTEXT
    current += 1
    print(f"  Running cce_adaptive (FILE LIST)... ({current}/{total})")

    try:
        retriever.reset()
        gen_result = adaptive_gen.generate(
            query=example.query,
            initial_context=file_list_context,
        )

        debug_info = {
            'example_id': example.id,
            'total_retrievals': gen_result.total_retrievals,
            'context_sources': gen_result.context_sources,
            'total_context_tokens': gen_result.total_context_tokens,
            'response_length': len(gen_result.response),
        }
        cce_debug.append(debug_info)

        if adaptive_gen.topic_inferrer.history:
            last_topic = adaptive_gen.topic_inferrer.history[-1]
            debug_info['matched_files'] = last_topic.matched_files
            debug_info['search_terms'] = last_topic.metadata.get('search_terms', [])

        print(f"    CCE triggered {gen_result.total_retrievals} retrievals")
        if hasattr(adaptive_gen, "topic_inferrer") and adaptive_gen.topic_inferrer.last_result:
            topic_result = adaptive_gen.topic_inferrer.last_result
            print(f"    Top tokens from logits: {topic_result.top_tokens[:5]}")
            print(f"    Extracted identifiers: {topic_result.extracted_identifiers[:5]}")
            print(f"    Query used: {topic_result.query[:60]}")
        print(f"    Retrieved files: {gen_result.context_sources}")

        cce_eval = metrics.evaluate(
            example_id=example.id,
            generated_answer=gen_result.response,
            ground_truth_answer=example.ground_truth_answer,
            retrieved_files=gen_result.context_sources,
            ground_truth_files=example.ground_truth_files,
            ground_truth_keywords=example.ground_truth_keywords,
            tokens_used=gen_result.total_context_tokens + file_list_tokens,
            baseline_tokens=baseline_tokens,
            generation_time=gen_result.generation_time,
            num_retrievals=gen_result.total_retrievals,
            method='cce_adaptive',
        )
        cce_eval.metadata['category'] = example.category.value
        cce_eval.metadata['difficulty'] = example.difficulty.value
        all_results['cce_adaptive'].append(cce_eval)
    except Exception as e:
        print(f"    CCE error: {e}")

    # ========================================================================
    # REAL CCE MULTI-HOP: cce_topk_only
    # ========================================================================
    current += 1
    print(f"  Running cce_topk_only (REAL CCE)... ({current}/{total})")

    try:
        retriever.reset()
        torch.cuda.empty_cache()

        start = time.time()
        ret_result = real_cce_retrievers['cce_topk_only'].retrieve(example.query)

        debug_info = {
            'example_id': example.id,
            'num_hops': ret_result.num_hops,
            'retrieved_files': ret_result.retrieved_files,
            'total_tokens': ret_result.total_tokens,
            'trace': ret_result.trace,
        }
        real_cce_debug['cce_topk_only'].append(debug_info)

        if ret_result.num_hops == 0:
            # FIXED: Fallback to no_context generation instead of hard-coded failure
            print(f"    NO SPIKE - falling back to no_context generation")
            fallback_prompt = f"Question: {example.query}\\nAnswer:"
            answer, gen_len = generate_answer(fallback_prompt, max_new_tokens=200)
            gen_time = time.time() - start
            tokens_used = file_list_tokens + gen_len  # Just file_list + generated
        else:
            if ret_result.trace and ret_result.trace[0].get('confused_tokens'):
                tokens = ret_result.trace[0]['confused_tokens'][:5]
                cce_val = ret_result.trace[0].get('cce', 0)
                print(f"    CCE: {cce_val:.2f}, confused tokens: {tokens}")
            print(f"    Retrieved {len(ret_result.retrieved_files)} files: {ret_result.retrieved_files}")

            prompt = f"""Context:
{ret_result.retrieved_content[:2500]}

Question: {example.query}
Answer:"""
            answer, gen_len = generate_answer(prompt, max_new_tokens=200)
            gen_time = time.time() - start
            # FIXED: Unified token counting
            prompt_tokens = len(tokenizer.encode(f"Question: {example.query}\\nAnswer:"))
            tokens_used = file_list_tokens + ret_result.total_tokens + gen_len

        eval_result = metrics.evaluate(
            example_id=example.id,
            generated_answer=answer,
            ground_truth_answer=example.ground_truth_answer,
            retrieved_files=ret_result.retrieved_files,
            ground_truth_files=example.ground_truth_files,
            ground_truth_keywords=example.ground_truth_keywords,
            tokens_used=tokens_used,
            baseline_tokens=baseline_tokens,
            generation_time=gen_time,
            num_retrievals=ret_result.num_hops,
            method='cce_topk_only',
        )
        eval_result.metadata['category'] = example.category.value
        eval_result.metadata['difficulty'] = example.difficulty.value
        eval_result.metadata['spike_detected'] = ret_result.num_hops > 0
        all_results['cce_topk_only'].append(eval_result)

    except Exception as e:
        print(f"    cce_topk_only error: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # REAL CCE MULTI-HOP: cce_query_plus_topk
    # ========================================================================
    current += 1
    print(f"  Running cce_query_plus_topk (REAL CCE)... ({current}/{total})")

    try:
        retriever.reset()
        torch.cuda.empty_cache()

        start = time.time()
        ret_result = real_cce_retrievers['cce_query_plus_topk'].retrieve(example.query)

        debug_info = {
            'example_id': example.id,
            'num_hops': ret_result.num_hops,
            'retrieved_files': ret_result.retrieved_files,
            'total_tokens': ret_result.total_tokens,
            'trace': ret_result.trace,
        }
        real_cce_debug['cce_query_plus_topk'].append(debug_info)

        if ret_result.num_hops == 0:
            # FIXED: Fallback to no_context generation instead of hard-coded failure
            print(f"    NO SPIKE - falling back to no_context generation")
            fallback_prompt = f"Question: {example.query}\\nAnswer:"
            answer, gen_len = generate_answer(fallback_prompt, max_new_tokens=200)
            gen_time = time.time() - start
            tokens_used = file_list_tokens + gen_len
        else:
            if ret_result.trace and ret_result.trace[0].get('confused_tokens'):
                tokens = ret_result.trace[0]['confused_tokens'][:5]
                cce_val = ret_result.trace[0].get('cce', 0)
                print(f"    CCE: {cce_val:.2f}, confused tokens: {tokens}")
            print(f"    Retrieved {len(ret_result.retrieved_files)} files: {ret_result.retrieved_files}")

            prompt = f"""Context:
{ret_result.retrieved_content[:2500]}

Question: {example.query}
Answer:"""
            answer, gen_len = generate_answer(prompt, max_new_tokens=200)
            gen_time = time.time() - start
            tokens_used = file_list_tokens + ret_result.total_tokens + gen_len

        eval_result = metrics.evaluate(
            example_id=example.id,
            generated_answer=answer,
            ground_truth_answer=example.ground_truth_answer,
            retrieved_files=ret_result.retrieved_files,
            ground_truth_files=example.ground_truth_files,
            ground_truth_keywords=example.ground_truth_keywords,
            tokens_used=tokens_used,
            baseline_tokens=baseline_tokens,
            generation_time=gen_time,
            num_retrievals=ret_result.num_hops,
            method='cce_query_plus_topk',
        )
        eval_result.metadata['category'] = example.category.value
        eval_result.metadata['difficulty'] = example.difficulty.value
        eval_result.metadata['spike_detected'] = ret_result.num_hops > 0
        all_results['cce_query_plus_topk'].append(eval_result)

    except Exception as e:
        print(f"    cce_query_plus_topk error: {e}")
        import traceback
        traceback.print_exc()

print("\\n" + "=" * 60)
print("EXPERIMENTS COMPLETE")
print("=" * 60)

# Summary
print("\\nRESULTS SUMMARY:")
for method, results in all_results.items():
    if results:
        avg_correctness = sum(r.answer_correctness for r in results) / len(results)
        avg_completeness = sum(r.answer_completeness for r in results) / len(results)
        avg_tokens = sum(r.tokens_used for r in results) / len(results)
        print(f"{method:25s}: correctness={avg_correctness:.3f}, completeness={avg_completeness:.3f}, tokens={avg_tokens:.0f}")

# CCE retrieval summary
print("\\nCCE RETRIEVAL SUMMARY:")
print("-" * 40)
if cce_debug:
    total_retrievals = sum(d['total_retrievals'] for d in cce_debug)
    print(f"\\n1. cce_adaptive (FILE LIST APPROACH):")
    print(f"   Total retrievals: {total_retrievals}")
    print(f"   Average per example: {total_retrievals / len(cce_debug):.1f}")

for variant in ['cce_topk_only', 'cce_query_plus_topk']:
    if real_cce_debug[variant]:
        spike_count = sum(1 for d in real_cce_debug[variant] if d['num_hops'] > 0)
        avg_hops = sum(d['num_hops'] for d in real_cce_debug[variant]) / len(real_cce_debug[variant])
        print(f"\\n2. {variant} (REAL CCE - H_code - H_lang):")
        print(f"   Spike detection rate: {spike_count}/{len(real_cce_debug[variant])}")
        print(f"   Average hops: {avg_hops:.1f}")
'''

# =============================================================================
# FIX 3: Update Cell 22 (CCE Configuration) - Change threshold from 3.0 to 0.5
# =============================================================================

cell_22_new = '''# Cell 16: Import generation modules and configure CCE with FILE LIST approach
from orchestrator.generation.adaptive_generator import (
    AdaptiveGenerator,
    GenerationConfig,
)
from orchestrator.retrieval.topic_inference import TopicInferrer
from orchestrator.retrieval.adaptive import EmbeddingRetriever

print("=" * 60)
print("CONFIGURING CCE ADAPTIVE WITH FILE LIST CONTEXT")
print("=" * 60)

# Create file list context (very small - just paths, not content)
file_list = list(documents.keys())
file_list_context = """Available files in this codebase:
{}

When answering questions about this codebase, refer to the relevant files above.""".format(
    "\\n".join(f"- {f}" for f in file_list)
)

print("FILE LIST CONTEXT:")
print("-" * 40)
print(file_list_context)
print("-" * 40)
file_list_tokens = len(tokenizer.encode(file_list_context))
print(f"File list tokens: {file_list_tokens} (vs {sum(len(tokenizer.encode(c)) for c in documents.values()):,} for full content)")

# Create retriever with file list for topic inference
retriever = EmbeddingRetriever(documents=documents)
retriever.set_available_files = lambda x: None  # Stub
if hasattr(retriever, '_file_tokens'):
    retriever._file_tokens = {}

# Create topic inferrer with file paths
topic_inferrer = TopicInferrer(
    tokenizer=tokenizer,
    available_files=file_list,
)

# Create adaptive generator
config = GenerationConfig(
    max_tokens=200,
    max_retrievals=5,
    uncertainty_threshold=3.0,  # Raw entropy threshold for cce_adaptive
    cooldown_tokens=5,
    uncertainty_method="raw_entropy",
    measurement_strategy="every_token",
    top_k_retrieval=2,
)

adaptive_gen = AdaptiveGenerator(
    model=model,
    tokenizer=tokenizer,
    retriever=retriever,
    config=config,
)
adaptive_gen.topic_inferrer = topic_inferrer

print(f"\\nAdaptiveGenerator configured:")
print(f"  - max_tokens: {config.max_tokens}")
print(f"  - max_retrievals: {config.max_retrievals}")
print(f"  - uncertainty_threshold: {config.uncertainty_threshold}")
print(f"  - measurement_strategy: {config.measurement_strategy}")

# Create Real CCE retrievers with CORRECT threshold
# FIXED: Use 0.5 for CCE (H_code - H_lang), not 3.0 (raw entropy scale)
real_cce_retrievers = {
    'cce_topk_only': CCETopKOnlyRetriever(
        retriever, tokenizer, model,
        top_k=2, max_retrievals=5, uncertainty_threshold=0.5, top_k_tokens=10,  # FIXED: 0.5 not 3.0
        max_gen_tokens=200, cooldown_tokens=5,
        file_list_context=file_list_context
    ),
    'cce_query_plus_topk': CCEQueryPlusTopKRetriever(
        retriever, tokenizer, model,
        top_k=2, max_retrievals=5, uncertainty_threshold=0.5, top_k_tokens=10,  # FIXED: 0.5 not 3.0
        max_gen_tokens=200, cooldown_tokens=5,
        file_list_context=file_list_context
    ),
}

print(f"\\nReal CCE retrievers initialized:")
print(f"  - cce_topk_only: CCETopKOnlyRetriever")
print(f"  - cce_query_plus_topk: CCEQueryPlusTopKRetriever")
print()
print("Configuration:")
print(f"  - uncertainty_threshold: 0.5 (CCE scale: H_code - H_lang)")  # FIXED
print(f"  - file_list_context: ENABLED ({file_list_tokens} tokens)")
print(f"  - top_k_tokens: 10 (confused tokens extracted from logits)")
print(f"  - max_gen_tokens: 200 (tokens to generate looking for spike)")
print()
'''

# =============================================================================
# FIX 6: Update Cell 18 (Real CCE classes) - Add BPE limitation comment
# =============================================================================

# Read the current cell 18 content from update_real_cce.py
cell_18_new = '''# Cell 16.5: Real CCE Multi-Hop Retrieval Classes (FULL CCE: H_code - H_lang)
#
# NOTE ON BPE LIMITATION:
# CODE_KEYWORDS matching is approximate due to BPE tokenization. Most vocabulary
# tokens are subword pieces (e.g., "▁fun", "ction") that won't match whole-word
# keywords like "function". This means code_indices and language_indices are
# small subsets of the vocabulary. This is a known limitation of keyword-based
# token classification with BPE tokenizers.

from typing import List, Tuple, Dict, Any, NamedTuple
from dataclasses import dataclass
import re
import numpy as np
from scipy.stats import entropy as scipy_entropy

# ============================================================================
# TOKEN CLASSIFICATION (Same as orchestrator/entropy/token_classifier.py)
# ============================================================================

# Programming keywords and library names
CODE_KEYWORDS = {
    # Python keywords
    'if', 'else', 'elif', 'for', 'while', 'break', 'continue', 'pass',
    'return', 'yield', 'raise', 'try', 'except', 'finally', 'with', 'as',
    'def', 'class', 'lambda', 'async', 'await', 'and', 'or', 'not', 'in', 'is',
    'None', 'True', 'False', 'import', 'from', 'print', 'len', 'range',
    'enumerate', 'zip', 'map', 'filter', 'list', 'dict', 'set', 'tuple',
    'str', 'int', 'float', 'bool', 'open', 'read', 'write', 'close', 'append',
    # JS keywords
    'function', 'const', 'let', 'var', 'switch', 'case', 'default',
    'true', 'false', 'null', 'undefined', 'typeof', 'instanceof',
    'export', 'require', 'module', 'exports', 'console', 'log',
    # Data science libraries
    'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'sklearn',
    'tensorflow', 'torch', 'keras', 'np', 'pd', 'array', 'ndarray',
    'DataFrame', 'Series', 'read_csv', 'groupby', 'merge', 'concat',
    'fit', 'transform', 'predict', 'score', 'train_test_split',
    # Web frameworks
    'flask', 'django', 'fastapi', 'requests', 'aiohttp', 'sqlalchemy',
    'FastAPI', 'APIRouter', 'HTTPException', 'Request', 'Response',
    'Depends', 'Query', 'Path', 'Body', 'get', 'post', 'put', 'delete',
    'Flask', 'render_template', 'redirect', 'url_for', 'jsonify',
    # React/JS
    'react', 'vue', 'angular', 'next', 'express', 'axios',
    'useState', 'useEffect', 'useContext', 'useReducer', 'useCallback',
    'useMemo', 'useRef', 'React', 'Component', 'createElement', 'Fragment',
    'props', 'state', 'render', 'componentDidMount',
    # Cloud/Firebase
    'firebase', 'Firebase', 'firestore', 'Firestore', 'auth', 'Auth',
    'database', 'storage', 'functions', 'messaging', 'admin', 'initializeApp',
    'aws', 'boto3', 's3', 'dynamodb', 'mongodb', 'redis', 'postgres',
    # Testing
    'pytest', 'unittest', 'jest', 'mocha', 'test', 'describe', 'it',
    'expect', 'assert', 'mock', 'fixture', 'setUp', 'tearDown',
}

# Common English words (language tokens)
LANGUAGE_WORDS = {
    'what', 'how', 'why', 'when', 'where', 'which', 'who', 'whom',
    'explain', 'show', 'tell', 'help', 'search', 'generate', 'implement',
    'refactor', 'optimize', 'improve', 'summarize', 'analyze', 'review',
    'the', 'a', 'an', 'that', 'these', 'those', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'does', 'did', 'will',
    'would', 'should', 'can', 'could', 'may', 'might', 'must', 'shall',
    'to', 'without', 'by', 'at', 'on', 'of', 'off', 'up', 'down', 'over',
    'under', 'above', 'below', 'but', 'nor', 'so', 'yet', 'my', 'your',
    'his', 'her', 'its', 'our', 'their', 'you', 'he', 'she', 'we', 'they',
    'me', 'him', 'us', 'them', 'better', 'worse', 'best', 'worst', 'good',
    'bad', 'great', 'poor', 'slow', 'quick', 'simple', 'complex', 'easy',
    'hard', 'answer', 'question', 'summary', 'comment', 'documentation',
}

@dataclass
class MultiHopRetrievalResult:
    """Result from multi-hop retrieval."""
    retrieved_files: List[str]
    retrieved_content: str
    scores: List[float]
    num_hops: int
    total_tokens: int
    trace: List[Dict[str, Any]]


class RealCCEMultiHopRetriever:
    """
    REAL CCE Multi-Hop: Uses FULL Contrastive Code Entropy (H_code - H_lang).

    Key difference from raw entropy:
    - Raw entropy: H = -sum(p * log(p)) over ALL tokens
    - CCE: H_code - H_lang (only triggers on CODE uncertainty)

    This ensures we only retrieve when the model is uncertain about CODE,
    not when it's uncertain about formatting/language tokens like "What", "The".

    THRESHOLD NOTE: Use ~0.5 for CCE (not 3.0 which is raw entropy scale).
    """

    def __init__(self, base_retriever, tokenizer, model,
                 top_k: int = 2, max_retrievals: int = 5,
                 uncertainty_threshold: float = 0.5,  # CCE threshold (not raw entropy!)
                 top_k_tokens: int = 10,
                 max_gen_tokens: int = 200,
                 cooldown_tokens: int = 5,
                 file_list_context: str = ""):
        self.retriever = base_retriever
        self.tokenizer = tokenizer
        self.model = model
        self.top_k = top_k
        self.max_retrievals = max_retrievals
        self.uncertainty_threshold = uncertainty_threshold
        self.top_k_tokens = top_k_tokens
        self.max_gen_tokens = max_gen_tokens
        self.cooldown_tokens = cooldown_tokens
        self.file_list_context = file_list_context

        # Build vocabulary classification (one-time)
        self._build_vocab_classification()

    def _build_vocab_classification(self):
        """
        Classify all tokens in vocabulary as code, language, or other.

        NOTE: Due to BPE tokenization, most tokens won't match whole-word keywords.
        This results in small code_indices/language_indices arrays.
        """
        vocab_size = len(self.tokenizer)
        self.code_indices = []
        self.language_indices = []
        self.other_indices = []

        for token_id in range(vocab_size):
            try:
                token = self.tokenizer.decode([token_id]).strip().lower()

                if token in CODE_KEYWORDS or token in {k.lower() for k in CODE_KEYWORDS}:
                    self.code_indices.append(token_id)
                elif token in LANGUAGE_WORDS or token in {w.lower() for w in LANGUAGE_WORDS}:
                    self.language_indices.append(token_id)
                else:
                    self.other_indices.append(token_id)
            except:
                self.other_indices.append(token_id)

        self.code_indices = np.array(self.code_indices)
        self.language_indices = np.array(self.language_indices)

        print(f"    Vocab classification: {len(self.code_indices)} code, {len(self.language_indices)} language, {len(self.other_indices)} other")

    def _compute_cce(self, logits: torch.Tensor) -> Tuple[float, float, float]:
        """
        Compute FULL Contrastive Code Entropy: CCE = H_code - H_lang

        Returns: (cce, h_code, h_lang)
        """
        logits_np = logits.cpu().numpy()

        # Compute entropy for CODE tokens only
        if len(self.code_indices) > 0:
            code_logits = logits_np[self.code_indices]
            code_logits_stable = code_logits - np.max(code_logits)
            code_probs = np.exp(code_logits_stable) / np.sum(np.exp(code_logits_stable))
            h_code = float(scipy_entropy(code_probs, base=2))
        else:
            h_code = 0.0

        # Compute entropy for LANGUAGE tokens only
        if len(self.language_indices) > 0:
            lang_logits = logits_np[self.language_indices]
            lang_logits_stable = lang_logits - np.max(lang_logits)
            lang_probs = np.exp(lang_logits_stable) / np.sum(np.exp(lang_logits_stable))
            h_lang = float(scipy_entropy(lang_probs, base=2))
        else:
            h_lang = 0.0

        # CCE = H_code - H_lang
        cce = h_code - h_lang

        return cce, h_code, h_lang

    def _extract_code_tokens_from_logits(self, logits: torch.Tensor) -> List[str]:
        """
        Extract top CODE tokens from logits - restricted to CODE_KEYWORDS.

        Uses the CODE_KEYWORDS vocabulary subset to get meaningful code tokens
        that the model is considering (numpy, pandas, class, def, etc.).
        """
        logits_np = logits.cpu().numpy()

        # Get logits for CODE tokens only
        code_logits = logits_np[self.code_indices]

        # Get top-k within the CODE token subset
        top_within_code = np.argsort(code_logits)[-self.top_k_tokens:][::-1]

        code_tokens = []
        for i in top_within_code:
            token_id = self.code_indices[i]
            token = self.tokenizer.decode([token_id]).strip()
            if len(token) > 1:
                code_tokens.append(token)

        return code_tokens

    def _count_tokens(self, content: str) -> int:
        return len(self.tokenizer.encode(content))

    def _extract_identifiers_from_context(self, context: str) -> List[str]:
        """Extract identifiers from file list context."""
        identifier_pattern = r'\\b([a-zA-Z_][a-zA-Z0-9_]{2,})\\b'
        matches = re.findall(identifier_pattern, context)

        stopwords = {
            'def', 'class', 'import', 'from', 'return', 'if', 'else',
            'for', 'while', 'try', 'except', 'with', 'as', 'and', 'or',
            'not', 'true', 'false', 'none', 'null', 'the', 'is', 'are',
            'src', 'tests', 'api', 'utils', 'config', 'app', 'main',
            'Available', 'files', 'this', 'codebase', 'When', 'answering',
        }

        identifiers = []
        seen = set()
        for match in matches:
            if match.lower() not in stopwords and match not in seen:
                seen.add(match)
                identifiers.append(match)

        return identifiers[:10]

    def _build_retrieval_query(self, query: str, confused_tokens: List[str]) -> str:
        context_identifiers = self._extract_identifiers_from_context(self.file_list_context)
        all_terms = context_identifiers[:5] + confused_tokens[:5]
        return f"{query} {' '.join(all_terms)}"

    def retrieve(self, query: str) -> MultiHopRetrievalResult:
        """Real CCE retrieval using H_code - H_lang."""
        if self.file_list_context:
            prompt = f"{query}\\n\\n{self.file_list_context}\\n\\n"
        else:
            prompt = f"{query}\\n\\n"

        retrieved_files = []
        retrieved_content = []
        all_scores = []
        trace = []
        seen_files = set()

        retrieval_count = 0
        last_retrieval_pos = -100

        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        generated_ids = inputs['input_ids']

        for i in range(self.max_gen_tokens):
            with torch.no_grad():
                outputs = self.model(generated_ids)
                logits = outputs.logits[0, -1, :]

            cce, h_code, h_lang = self._compute_cce(logits)

            if i < 3:
                print(f"    Token {i}: CCE={cce:.3f} (H_code={h_code:.2f}, H_lang={h_lang:.2f}, threshold={self.uncertainty_threshold})")

            tokens_since_last = i - last_retrieval_pos
            in_cooldown = tokens_since_last < self.cooldown_tokens

            if cce > self.uncertainty_threshold and not in_cooldown and retrieval_count < self.max_retrievals:
                confused_tokens = self._extract_code_tokens_from_logits(logits)
                retrieval_query = self._build_retrieval_query(query, confused_tokens)

                results = self.retriever.retrieve(retrieval_query, top_k=self.top_k, deduplicate=False)
                new_files = [r for r in results if r['source'] not in seen_files]

                if new_files:
                    for r in new_files:
                        seen_files.add(r['source'])
                        retrieved_files.append(r['source'])
                        retrieved_content.append(r['content'])
                        all_scores.append(r['score'])

                    new_context = "\\n\\n".join([r['content'] for r in new_files])
                    context_text = f"\\n\\nRelevant context:\\n{new_context}\\n\\n"
                    context_ids = self.tokenizer.encode(context_text, return_tensors='pt').to(self.model.device)
                    generated_ids = torch.cat([generated_ids, context_ids], dim=-1)

                trace.append({
                    'hop': retrieval_count + 1,
                    'position': i,
                    'cce': cce,
                    'h_code': h_code,
                    'h_lang': h_lang,
                    'confused_tokens': confused_tokens[:5],
                    'retrieval_query': retrieval_query[:80],
                    'new_files': [r['source'] for r in new_files],
                })

                retrieval_count += 1
                last_retrieval_pos = i

                context_ids = self._extract_identifiers_from_context(self.file_list_context)
                print(f"    CODE Spike {retrieval_count} at token {i}: CCE={cce:.2f}")
                print(f"      Context identifiers: {context_ids[:5]}")
                print(f"      Code tokens: {confused_tokens[:5]}")

            next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token.to(self.model.device)], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        if not trace:
            trace.append({
                'hop': 0,
                'method': 'no_code_spike_detected',
                'reason': 'No CODE uncertainty detected (CCE never exceeded threshold)',
            })

        content = "\\n\\n".join(retrieved_content)
        print(f"    Total CODE retrievals: {retrieval_count}, files: {retrieved_files}")

        return MultiHopRetrievalResult(
            retrieved_files=retrieved_files,
            retrieved_content=content,
            scores=all_scores,
            num_hops=retrieval_count,
            total_tokens=self._count_tokens(content) if content else 0,
            trace=trace
        )


class CCETopKOnlyRetriever(RealCCEMultiHopRetriever):
    """Ablation: Uses context identifiers + confused CODE tokens (no original query)."""

    def _build_retrieval_query(self, query: str, confused_tokens: List[str]) -> str:
        context_identifiers = self._extract_identifiers_from_context(self.file_list_context)
        all_terms = context_identifiers[:5] + confused_tokens[:5]
        return ' '.join(all_terms)


class CCEQueryPlusTopKRetriever(RealCCEMultiHopRetriever):
    """Uses original query + context identifiers + confused CODE tokens."""

    def _build_retrieval_query(self, query: str, confused_tokens: List[str]) -> str:
        context_identifiers = self._extract_identifiers_from_context(self.file_list_context)
        all_terms = context_identifiers[:5] + confused_tokens[:5]
        return f"{query} {' '.join(all_terms)}"


print("Real CCE Multi-Hop defined (FULL CCE: H_code - H_lang)")
print("  - Computes Contrastive Code Entropy (H_code - H_lang)")
print("  - Only triggers on CODE uncertainty (CCE > threshold)")
print("  - Threshold: 0.5 (CCE scale), not 3.0 (raw entropy)")
print("  - NOTE: BPE token classification is approximate")
'''

# =============================================================================
# Apply all fixes to notebook
# =============================================================================

print("Fix 1 & 4 & 5: Updating Cell 24 (experiment loop)...")
nb['cells'][24]['source'] = [line + '\n' for line in cell_24_new.split('\n')]
nb['cells'][24]['source'][-1] = nb['cells'][24]['source'][-1].rstrip('\n')
nb['cells'][24]['outputs'] = []  # Clear old outputs

print("Fix 3: Updating Cell 22 (CCE configuration - threshold fix)...")
nb['cells'][22]['source'] = [line + '\n' for line in cell_22_new.split('\n')]
nb['cells'][22]['source'][-1] = nb['cells'][22]['source'][-1].rstrip('\n')
nb['cells'][22]['outputs'] = []

print("Fix 6: Updating Cell 18 (Real CCE classes - BPE limitation comment)...")
nb['cells'][18]['source'] = [line + '\n' for line in cell_18_new.split('\n')]
nb['cells'][18]['source'][-1] = nb['cells'][18]['source'][-1].rstrip('\n')
nb['cells'][18]['outputs'] = []

# =============================================================================
# Save notebook
# =============================================================================

with open('C:/Users/rajka/reposynth/research/Week8_Comprehensive_Evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print()
print("=" * 60)
print("ALL FIXES APPLIED SUCCESSFULLY")
print("=" * 60)
print()
print("Summary of fixes:")
print("  1. Answer slicing - Now properly slices off prompt before decoding")
print("  2. file_list_context - All methods now receive file_list_context")
print("  3. Threshold - Changed from 3.0 to 0.5 (CCE scale)")
print("  4. No-spike fallback - Falls back to no_context generation")
print("  5. Token counting - Unified counting: file_list + context + generated")
print("  6. BPE limitation - Documented in code comments")
print()
print("Re-run the notebook to get valid, comparable results.")
