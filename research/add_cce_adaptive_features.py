"""
Add cce_adaptive features to Real CCE:
1. Top-k logit extraction from ALL vocab (not just CODE_KEYWORDS)
2. File path matching (TopicInferrer._match_files())

This makes Real CCE use the same identifier extraction as cce_adaptive.
"""

import json

# Load notebook
with open('C:/Users/rajka/reposynth/research/Week8_Comprehensive_Evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Adding cce_adaptive features to Real CCE...")
print("=" * 60)

# =============================================================================
# Updated Cell 18: Real CCE classes with cce_adaptive features
# =============================================================================

cell_18_new = '''# Cell 16.5: Real CCE Multi-Hop Retrieval Classes (FULL CCE: H_code - H_lang)
#
# ENHANCED with cce_adaptive features:
# - Top-k token extraction from ALL vocab (not just CODE_KEYWORDS)
# - File path matching (like TopicInferrer._match_files())
# - Stopwords filtering (like TopicInferrer._filter_code_tokens())
#
# NOTE ON BPE LIMITATION:
# CODE_KEYWORDS matching is approximate due to BPE tokenization. This version
# also extracts top-k from ALL vocab and filters, giving better tokens.

from typing import List, Tuple, Dict, Any, Set
from dataclasses import dataclass
import re
import numpy as np
from scipy.stats import entropy as scipy_entropy

# ============================================================================
# TOKEN CLASSIFICATION
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

# Stopwords for filtering top-k tokens (from cce_adaptive's TopicInferrer)
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
    - Top-k token extraction from ALL vocab (like TopicInferrer._get_top_tokens)
    - File path matching (like TopicInferrer._match_files)
    - Uses FULL CCE: H_code - H_lang
    """

    def __init__(self, base_retriever, tokenizer, model,
                 top_k: int = 2, max_retrievals: int = 5,
                 uncertainty_threshold: float = 0.5,
                 top_k_tokens: int = 20,  # Get more tokens, filter later
                 max_gen_tokens: int = 200,
                 cooldown_tokens: int = 5,
                 file_list_context: str = "",
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

        # File path matching (like cce_adaptive)
        self.available_files: List[str] = available_files or []
        self._file_tokens: Dict[str, Set[str]] = {}
        if available_files:
            self._preprocess_file_paths(available_files)
        elif file_list_context:
            # Extract file paths from context
            self._extract_files_from_context(file_list_context)

        # Build vocabulary classification (one-time)
        self._build_vocab_classification()

    def _extract_files_from_context(self, context: str):
        """Extract file paths from file list context."""
        # Match patterns like "- src/auth/jwt.py" or "src/auth/jwt.py"
        pattern = r'[\\-\\s]*([\\w/\\\\_]+\\.py)'
        matches = re.findall(pattern, context)
        self.available_files = matches
        self._preprocess_file_paths(matches)

    def _preprocess_file_paths(self, file_paths: List[str]):
        """Extract searchable tokens from each file path (like TopicInferrer)."""
        self._file_tokens = {}
        for path in file_paths:
            tokens = self._extract_path_tokens(path)
            self._file_tokens[path] = tokens

    def _extract_path_tokens(self, path: str) -> Set[str]:
        """Extract searchable tokens from a file path (like TopicInferrer)."""
        tokens = set()
        parts = re.split(r'[/\\\\]', path)

        for part in parts:
            name = re.sub(r'\\.[^.]+$', '', part)  # Remove extension
            tokens.add(name.lower())

            # Split on underscores
            for subpart in name.split('_'):
                if len(subpart) >= 2:
                    tokens.add(subpart.lower())

            # Split camelCase
            camel_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', name)
            for cp in camel_parts:
                if len(cp) >= 2:
                    tokens.add(cp.lower())

        return tokens

    def _match_files(self, code_tokens: List[str], identifiers: List[str], top_k: int = 3) -> Tuple[List[str], Dict[str, float]]:
        """
        Match confused tokens against available file paths (like TopicInferrer._match_files).

        Returns: (matched_files, score_dict)
        """
        if not self.available_files:
            return [], {}

        # Combine tokens and identifiers for matching
        search_terms = set()
        for token in code_tokens[:10]:
            search_terms.add(token.lower().strip())
        for ident in identifiers[:10]:
            for part in ident.split('.'):
                if len(part) >= 2:
                    search_terms.add(part.lower())

        # Score each file
        scores: Dict[str, float] = {}
        for path, path_tokens in self._file_tokens.items():
            score = 0.0
            matched_terms = search_terms & path_tokens

            if matched_terms:
                score = len(matched_terms)

                # Bonus for exact file name match
                file_name = re.split(r'[/\\\\]', path)[-1]
                file_name_no_ext = re.sub(r'\\.[^.]+$', '', file_name).lower()
                if file_name_no_ext in search_terms:
                    score += 2.0

                # Bonus for directory match
                dir_parts = re.split(r'[/\\\\]', path)[:-1]
                for dir_part in dir_parts:
                    if dir_part.lower() in search_terms:
                        score += 0.5

            if score > 0:
                scores[path] = score

        # Sort by score and return top-k
        sorted_files = sorted(scores.items(), key=lambda x: -x[1])
        matched = [f for f, s in sorted_files[:top_k]]

        return matched, scores

    def _build_vocab_classification(self):
        """Classify all tokens in vocabulary as code, language, or other."""
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
        print(f"    File paths for matching: {len(self.available_files)} files")

    def _compute_cce(self, logits: torch.Tensor) -> Tuple[float, float, float]:
        """Compute FULL CCE: H_code - H_lang"""
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
        """
        Get top-k tokens from ALL vocab (like TopicInferrer._get_top_tokens).
        This extracts what the model is actually considering, not just CODE_KEYWORDS.
        """
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
        """
        Filter tokens to code-relevant ones (like TopicInferrer._filter_code_tokens).
        Uses stopwords and identifier regex.
        """
        filtered = []
        for token in tokens:
            clean = token.lower().strip()
            if clean in STOPWORDS:
                continue
            # Keep if looks like code identifier
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
                filtered.append(token)
            elif '.' in token:  # Method chains
                filtered.append(token)
            elif '/' in token or '::' in token:  # Paths
                filtered.append(token)

        return filtered

    def _extract_identifiers_from_context(self, context: str) -> List[str]:
        """Extract identifiers from context (like TopicInferrer._extract_identifiers)."""
        identifier_pattern = r'\\b([a-zA-Z_][a-zA-Z0-9_]{2,})\\b'
        matches = re.findall(identifier_pattern, context)

        stopwords = {
            'def', 'class', 'import', 'from', 'return', 'if', 'else',
            'for', 'while', 'try', 'except', 'with', 'as', 'and', 'or',
            'not', 'true', 'false', 'none', 'null', 'the', 'is', 'are',
            'src', 'tests', 'api', 'utils', 'config', 'app', 'main',
            'Available', 'files', 'this', 'codebase', 'When', 'answering',
            'questions', 'about', 'refer', 'relevant', 'above',
        }

        identifiers = []
        seen = set()
        for match in matches:
            if match.lower() not in stopwords and match not in seen:
                seen.add(match)
                identifiers.append(match)

        return identifiers[:10]

    def _build_retrieval_query(self, query: str, code_tokens: List[str], identifiers: List[str], matched_files: List[str]) -> str:
        """Build retrieval query combining all sources."""
        # Priority: matched file names > identifiers > code tokens
        terms = []

        # Add file name parts from matched files
        for f in matched_files[:2]:
            name = re.split(r'[/\\\\]', f)[-1]
            name = re.sub(r'\\.[^.]+$', '', name)
            terms.append(name)

        # Add identifiers
        terms.extend(identifiers[:5])

        # Add code tokens
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
        """Real CCE retrieval with file matching."""
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
                # NEW: Get top-k from ALL vocab, then filter (like cce_adaptive)
                top_tokens = self._get_top_tokens_from_logits(logits)
                code_tokens = self._filter_code_tokens(top_tokens)

                # Extract identifiers from context
                identifiers = self._extract_identifiers_from_context(self.file_list_context)

                # NEW: Match against file paths (like cce_adaptive)
                matched_files, file_scores = self._match_files(code_tokens, identifiers)

                # Build query with all sources
                retrieval_query = self._build_retrieval_query(query, code_tokens, identifiers, matched_files)

                # Priority 1: Use matched files directly if available
                new_files = []
                if matched_files:
                    for fpath in matched_files:
                        if fpath not in seen_files:
                            # Try to get content from retriever
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
                    'matched_files': matched_files[:3],
                    'retrieval_query': retrieval_query[:80],
                    'new_files': [r['source'] for r in new_files],
                })

                retrieval_count += 1
                last_retrieval_pos = i

                print(f"    CODE Spike {retrieval_count} at token {i}: CCE={cce:.2f}")
                print(f"      Top tokens (all vocab): {top_tokens[:5]}")
                print(f"      Filtered code tokens: {code_tokens[:5]}")
                print(f"      Context identifiers: {identifiers[:5]}")
                print(f"      Matched files: {matched_files[:3]}")

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
    """Ablation: Uses identifiers + code tokens + file matching (no original query)."""

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
    """Uses original query + identifiers + code tokens + file matching."""
    pass  # Inherits _build_retrieval_query from parent which includes query


print("Real CCE Multi-Hop defined with cce_adaptive features:")
print("  - Top-k token extraction from ALL vocab (not just CODE_KEYWORDS)")
print("  - Stopwords filtering (like TopicInferrer)")
print("  - File path matching (like TopicInferrer._match_files)")
print("  - Priority: matched files > embedding search")
print("  - Threshold: 0.5 (CCE scale)")
'''

# Apply to notebook
print("Updating Cell 18 (Real CCE classes with cce_adaptive features)...")
nb['cells'][18]['source'] = [line + '\n' for line in cell_18_new.split('\n')]
nb['cells'][18]['source'][-1] = nb['cells'][18]['source'][-1].rstrip('\n')
nb['cells'][18]['outputs'] = []

# Save notebook
with open('C:/Users/rajka/reposynth/research/Week8_Comprehensive_Evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print()
print("=" * 60)
print("CCE_ADAPTIVE FEATURES ADDED TO REAL CCE")
print("=" * 60)
print()
print("New features in Real CCE:")
print("  1. _get_top_tokens_from_logits() - Gets top-k from ALL vocab")
print("  2. _filter_code_tokens() - Filters with stopwords + identifier regex")
print("  3. _preprocess_file_paths() - Preprocesses file paths for matching")
print("  4. _match_files() - Matches tokens against file paths")
print("  5. Priority retrieval: matched files first, then embedding search")
print()
print("Now Real CCE has the same identifier extraction as cce_adaptive!")
