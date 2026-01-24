"""
Fix Real CCE to extract identifiers from GENERATED text (dynamic),
not just from static file_list_context.

This matches cce_adaptive's TopicInferrer behavior.
"""

import json

with open('C:/Users/rajka/reposynth/research/Week8_Comprehensive_Evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Fixing Real CCE to use dynamic identifier extraction...")
print("=" * 60)

cell_18_new = '''# Cell 16.5: Real CCE Multi-Hop Retrieval Classes (FULL CCE: H_code - H_lang)
#
# ENHANCED with cce_adaptive features:
# - Top-k token extraction from ALL vocab
# - File path matching
# - DYNAMIC identifier extraction from GENERATED TEXT (not just file list)

from typing import List, Tuple, Dict, Any, Set
from dataclasses import dataclass
import re
import numpy as np
from scipy.stats import entropy as scipy_entropy

# ============================================================================
# TOKEN CLASSIFICATION & STOPWORDS
# ============================================================================

CODE_KEYWORDS = {
    'if', 'else', 'elif', 'for', 'while', 'break', 'continue', 'pass',
    'return', 'yield', 'raise', 'try', 'except', 'finally', 'with', 'as',
    'def', 'class', 'lambda', 'async', 'await', 'and', 'or', 'not', 'in', 'is',
    'None', 'True', 'False', 'import', 'from', 'print', 'len', 'range',
    'enumerate', 'zip', 'map', 'filter', 'list', 'dict', 'set', 'tuple',
    'str', 'int', 'float', 'bool', 'open', 'read', 'write', 'close', 'append',
    'function', 'const', 'let', 'var', 'switch', 'case', 'default',
    'true', 'false', 'null', 'undefined', 'typeof', 'instanceof',
    'export', 'require', 'module', 'exports', 'console', 'log',
    'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'sklearn',
    'tensorflow', 'torch', 'keras', 'np', 'pd', 'array', 'ndarray',
    'DataFrame', 'Series', 'read_csv', 'groupby', 'merge', 'concat',
    'fit', 'transform', 'predict', 'score', 'train_test_split',
    'flask', 'django', 'fastapi', 'requests', 'aiohttp', 'sqlalchemy',
    'FastAPI', 'APIRouter', 'HTTPException', 'Request', 'Response',
    'Depends', 'Query', 'Path', 'Body', 'get', 'post', 'put', 'delete',
    'Flask', 'render_template', 'redirect', 'url_for', 'jsonify',
    'react', 'vue', 'angular', 'next', 'express', 'axios',
    'useState', 'useEffect', 'useContext', 'useReducer', 'useCallback',
    'useMemo', 'useRef', 'React', 'Component', 'createElement', 'Fragment',
    'props', 'state', 'render', 'componentDidMount',
    'firebase', 'Firebase', 'firestore', 'Firestore', 'auth', 'Auth',
    'database', 'storage', 'functions', 'messaging', 'admin', 'initializeApp',
    'aws', 'boto3', 's3', 'dynamodb', 'mongodb', 'redis', 'postgres',
    'pytest', 'unittest', 'jest', 'mocha', 'test', 'describe', 'it',
    'expect', 'assert', 'mock', 'fixture', 'setUp', 'tearDown',
}

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

# Stopwords for filtering (comprehensive list from cce_adaptive)
STOPWORDS = {
    'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall',
    'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else',
    'this', 'that', 'these', 'those', 'it', 'its',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'under', 'again', 'further',
    'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'just',
    'What', 'How', 'Why', 'When', 'Where', 'Which',
    # Common keywords to filter
    'def', 'class', 'import', 'from', 'return', 'if', 'else',
    'for', 'while', 'try', 'except', 'with', 'as', 'and', 'or',
    'not', 'true', 'false', 'none', 'null', 'undefined',
    'function', 'const', 'let', 'var', 'async', 'await',
    # File list boilerplate
    'Available', 'files', 'codebase', 'answering', 'questions',
    'about', 'refer', 'relevant', 'above', 'Context', 'Question', 'Answer',
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
    REAL CCE Multi-Hop with cce_adaptive features:
    - Top-k token extraction from ALL vocab
    - DYNAMIC identifier extraction from GENERATED TEXT
    - File path matching
    - Uses FULL CCE: H_code - H_lang
    """

    def __init__(self, base_retriever, tokenizer, model,
                 top_k: int = 2, max_retrievals: int = 5,
                 uncertainty_threshold: float = 0.5,
                 top_k_tokens: int = 20,
                 max_gen_tokens: int = 200,
                 cooldown_tokens: int = 5,
                 file_list_context: str = "",
                 context_window: int = 300,  # Characters of generated text to analyze
                 available_files: List[str] = None):
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
        self.context_window = context_window  # NEW: How much generated text to analyze

        # File path matching
        self.available_files: List[str] = available_files or []
        self._file_tokens: Dict[str, Set[str]] = {}
        if available_files:
            self._preprocess_file_paths(available_files)
        elif file_list_context:
            self._extract_files_from_context(file_list_context)

        self._build_vocab_classification()

    def _extract_files_from_context(self, context: str):
        """Extract file paths from file list context."""
        pattern = r'[-\\s]*([\\w/\\\\_]+\\.py)'
        matches = re.findall(pattern, context)
        self.available_files = matches
        self._preprocess_file_paths(matches)

    def _preprocess_file_paths(self, file_paths: List[str]):
        """Extract searchable tokens from each file path."""
        self._file_tokens = {}
        for path in file_paths:
            tokens = self._extract_path_tokens(path)
            self._file_tokens[path] = tokens

    def _extract_path_tokens(self, path: str) -> Set[str]:
        """Extract searchable tokens from a file path."""
        tokens = set()
        parts = re.split(r'[/\\\\]', path)
        for part in parts:
            name = re.sub(r'\\.[^.]+$', '', part)
            tokens.add(name.lower())
            for subpart in name.split('_'):
                if len(subpart) >= 2:
                    tokens.add(subpart.lower())
            camel_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', name)
            for cp in camel_parts:
                if len(cp) >= 2:
                    tokens.add(cp.lower())
        return tokens

    def _match_files(self, code_tokens: List[str], identifiers: List[str], top_k: int = 3) -> Tuple[List[str], Dict[str, float]]:
        """Match tokens against available file paths."""
        if not self.available_files:
            return [], {}

        search_terms = set()
        for token in code_tokens[:10]:
            search_terms.add(token.lower().strip())
        for ident in identifiers[:10]:
            for part in ident.split('.'):
                if len(part) >= 2:
                    search_terms.add(part.lower())

        scores: Dict[str, float] = {}
        for path, path_tokens in self._file_tokens.items():
            score = 0.0
            matched_terms = search_terms & path_tokens
            if matched_terms:
                score = len(matched_terms)
                file_name = re.split(r'[/\\\\]', path)[-1]
                file_name_no_ext = re.sub(r'\\.[^.]+$', '', file_name).lower()
                if file_name_no_ext in search_terms:
                    score += 2.0
                dir_parts = re.split(r'[/\\\\]', path)[:-1]
                for dir_part in dir_parts:
                    if dir_part.lower() in search_terms:
                        score += 0.5
            if score > 0:
                scores[path] = score

        sorted_files = sorted(scores.items(), key=lambda x: -x[1])
        return [f for f, s in sorted_files[:top_k]], scores

    def _build_vocab_classification(self):
        """Classify all tokens in vocabulary."""
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
        print(f"    Vocab classification: {len(self.code_indices)} code, {len(self.language_indices)} language")
        print(f"    File paths for matching: {len(self.available_files)} files")

    def _compute_cce(self, logits: torch.Tensor) -> Tuple[float, float, float]:
        """Compute CCE = H_code - H_lang"""
        logits_np = logits.cpu().numpy()

        if len(self.code_indices) > 0:
            code_logits = logits_np[self.code_indices]
            code_logits_stable = code_logits - np.max(code_logits)
            code_probs = np.exp(code_logits_stable) / np.sum(np.exp(code_logits_stable))
            h_code = float(scipy_entropy(code_probs, base=2))
        else:
            h_code = 0.0

        if len(self.language_indices) > 0:
            lang_logits = logits_np[self.language_indices]
            lang_logits_stable = lang_logits - np.max(lang_logits)
            lang_probs = np.exp(lang_logits_stable) / np.sum(np.exp(lang_logits_stable))
            h_lang = float(scipy_entropy(lang_probs, base=2))
        else:
            h_lang = 0.0

        return h_code - h_lang, h_code, h_lang

    def _get_top_tokens_from_logits(self, logits: torch.Tensor) -> List[str]:
        """Get top-k tokens from ALL vocab."""
        logits_np = logits.cpu().numpy()
        top_indices = np.argsort(logits_np)[-self.top_k_tokens:][::-1]

        tokens = []
        for idx in top_indices:
            try:
                token = self.tokenizer.decode([idx]).strip()
                if token and len(token) >= 2:
                    tokens.append(token)
            except:
                continue
        return tokens

    def _filter_code_tokens(self, tokens: List[str]) -> List[str]:
        """Filter tokens to code-relevant ones."""
        filtered = []
        for token in tokens:
            clean = token.lower().strip()
            if clean in STOPWORDS:
                continue
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
                filtered.append(token)
            elif '.' in token:
                filtered.append(token)
            elif '/' in token or '::' in token:
                filtered.append(token)
        return filtered

    def _extract_identifiers(self, text: str) -> List[str]:
        """
        Extract identifiers from text (like TopicInferrer._extract_identifiers).

        This is called with GENERATED TEXT, not just file list.
        Extracts variable names, function names, class names, etc.
        """
        # Pattern for identifiers: foo, bar_baz, FooBar, foo123, _private
        identifier_pattern = r'\\b([a-zA-Z_][a-zA-Z0-9_]{2,})\\b'
        matches = re.findall(identifier_pattern, text)

        identifiers = []
        seen = set()
        for match in matches:
            if match.lower() not in STOPWORDS and match not in seen:
                seen.add(match)
                identifiers.append(match)

        # Also extract method chain patterns: foo.bar.baz
        chain_pattern = r'(\\w+(?:\\.\\w+)+)'
        chains = re.findall(chain_pattern, text)
        for chain in chains:
            if chain not in seen:
                seen.add(chain)
                identifiers.append(chain)

        return identifiers[:15]  # Return more identifiers

    def _build_retrieval_query(self, query: str, code_tokens: List[str], identifiers: List[str], matched_files: List[str]) -> str:
        """Build retrieval query combining all sources."""
        terms = []

        # Add file name parts from matched files
        for f in matched_files[:2]:
            name = re.split(r'[/\\\\]', f)[-1]
            name = re.sub(r'\\.[^.]+$', '', name)
            terms.append(name)

        # Add identifiers (from generated text)
        terms.extend(identifiers[:5])

        # Add code tokens (from logits)
        terms.extend(code_tokens[:5])

        # Deduplicate
        seen = set()
        unique_terms = []
        for t in terms:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique_terms.append(t)

        return f"{query} {' '.join(unique_terms[:8])}"

    def _count_tokens(self, content: str) -> int:
        return len(self.tokenizer.encode(content))

    def retrieve(self, query: str) -> MultiHopRetrievalResult:
        """Real CCE retrieval with dynamic identifier extraction."""
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
        initial_len = generated_ids.shape[-1]

        for i in range(self.max_gen_tokens):
            with torch.no_grad():
                outputs = self.model(generated_ids)
                logits = outputs.logits[0, -1, :]

            cce, h_code, h_lang = self._compute_cce(logits)

            if i < 3:
                print(f"    Token {i}: CCE={cce:.3f} (H_code={h_code:.2f}, H_lang={h_lang:.2f})")

            tokens_since_last = i - last_retrieval_pos
            in_cooldown = tokens_since_last < self.cooldown_tokens

            if cce > self.uncertainty_threshold and not in_cooldown and retrieval_count < self.max_retrievals:
                # Get top-k from ALL vocab, then filter
                top_tokens = self._get_top_tokens_from_logits(logits)
                code_tokens = self._filter_code_tokens(top_tokens)

                # DYNAMIC: Extract identifiers from GENERATED TEXT so far
                # Decode what we've generated (excluding initial prompt)
                generated_text = self.tokenizer.decode(generated_ids[0, initial_len:], skip_special_tokens=True)

                # Use last N characters of generated text + file list context
                recent_generated = generated_text[-self.context_window:] if generated_text else ""
                combined_context = recent_generated + "\\n" + self.file_list_context

                identifiers = self._extract_identifiers(combined_context)

                # Match against file paths
                matched_files, file_scores = self._match_files(code_tokens, identifiers)

                # Build query
                retrieval_query = self._build_retrieval_query(query, code_tokens, identifiers, matched_files)

                # Priority 1: Use matched files directly
                new_files = []
                if matched_files:
                    for fpath in matched_files:
                        if fpath not in seen_files:
                            if hasattr(self.retriever, 'documents') and fpath in self.retriever.documents:
                                new_files.append({
                                    'source': fpath,
                                    'content': self.retriever.documents[fpath],
                                    'score': file_scores.get(fpath, 1.0),
                                })

                # Priority 2: Fallback to embedding search
                if not new_files:
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
                    'top_tokens': top_tokens[:5],
                    'code_tokens': code_tokens[:5],
                    'identifiers': identifiers[:5],
                    'generated_context': recent_generated[-100:] if recent_generated else "(none yet)",
                    'matched_files': matched_files[:3],
                    'retrieval_query': retrieval_query[:80],
                    'new_files': [r['source'] for r in new_files],
                })

                retrieval_count += 1
                last_retrieval_pos = i

                print(f"    CODE Spike {retrieval_count} at token {i}: CCE={cce:.2f}")
                print(f"      Top tokens (all vocab): {top_tokens[:5]}")
                print(f"      Filtered code tokens: {code_tokens[:5]}")
                print(f"      Identifiers (from generated): {identifiers[:5]}")
                print(f"      Matched files: {matched_files[:3]}")

            next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token.to(self.model.device)], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        if not trace:
            trace.append({
                'hop': 0,
                'method': 'no_code_spike_detected',
                'reason': 'CCE never exceeded threshold',
            })

        content = "\\n\\n".join(retrieved_content)
        print(f"    Total retrievals: {retrieval_count}, files: {retrieved_files}")

        return MultiHopRetrievalResult(
            retrieved_files=retrieved_files,
            retrieved_content=content,
            scores=all_scores,
            num_hops=retrieval_count,
            total_tokens=self._count_tokens(content) if content else 0,
            trace=trace
        )


class CCETopKOnlyRetriever(RealCCEMultiHopRetriever):
    """Ablation: No original query in retrieval."""

    def _build_retrieval_query(self, query: str, code_tokens: List[str], identifiers: List[str], matched_files: List[str]) -> str:
        terms = []
        for f in matched_files[:2]:
            name = re.split(r'[/\\\\]', f)[-1]
            name = re.sub(r'\\.[^.]+$', '', name)
            terms.append(name)
        terms.extend(identifiers[:5])
        terms.extend(code_tokens[:5])

        seen = set()
        unique_terms = []
        for t in terms:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique_terms.append(t)

        return ' '.join(unique_terms[:8])  # NO original query


class CCEQueryPlusTopKRetriever(RealCCEMultiHopRetriever):
    """Uses original query + all extracted terms."""
    pass  # Inherits parent's _build_retrieval_query


print("Real CCE Multi-Hop with DYNAMIC identifier extraction:")
print("  - Extracts identifiers from GENERATED TEXT (not just file list)")
print("  - Uses context_window to analyze recent generation")
print("  - Combines: generated identifiers + logit tokens + file matching")
print("  - Threshold: 0.5 (CCE scale)")
'''

# Apply to notebook
print("Updating Cell 18 with dynamic identifier extraction...")
nb['cells'][18]['source'] = [line + '\n' for line in cell_18_new.split('\n')]
nb['cells'][18]['source'][-1] = nb['cells'][18]['source'][-1].rstrip('\n')
nb['cells'][18]['outputs'] = []

with open('C:/Users/rajka/reposynth/research/Week8_Comprehensive_Evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print()
print("=" * 60)
print("DYNAMIC IDENTIFIER EXTRACTION ADDED")
print("=" * 60)
print()
print("Key changes:")
print("  1. _extract_identifiers() now takes any text, not just file list")
print("  2. In retrieve(), we decode generated_ids to get generated text")
print("  3. Identifiers extracted from: recent_generated + file_list_context")
print("  4. Trace now includes 'generated_context' for debugging")
print()
print("Now Real CCE extracts identifiers DYNAMICALLY like cce_adaptive!")
