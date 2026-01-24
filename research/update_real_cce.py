import json

with open('C:/Users/rajka/reposynth/research/Week8_Comprehensive_Evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_class = '''# Cell 16.5: Real CCE Multi-Hop Retrieval Classes (FULL CCE: H_code - H_lang)
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
    """

    def __init__(self, base_retriever, tokenizer, model,
                 top_k: int = 2, max_retrievals: int = 5,
                 uncertainty_threshold: float = 0.3,  # CCE threshold (not raw entropy!)
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
        """
        Extract identifiers from file list context (like cce_adaptive).

        This is the key insight: cce_adaptive gets domain-specific identifiers
        like 'JWTManager', 'User', 'create_access_token' from the context,
        NOT from the model's logits.
        """
        identifier_pattern = r'\\b([a-zA-Z_][a-zA-Z0-9_]{2,})\\b'
        matches = re.findall(identifier_pattern, context)

        # Filter out common keywords
        stopwords = {
            'def', 'class', 'import', 'from', 'return', 'if', 'else',
            'for', 'while', 'try', 'except', 'with', 'as', 'and', 'or',
            'not', 'true', 'false', 'none', 'null', 'the', 'is', 'are',
            'src', 'tests', 'api', 'utils', 'config', 'app', 'main',
        }

        identifiers = []
        seen = set()
        for match in matches:
            if match.lower() not in stopwords and match not in seen:
                seen.add(match)
                identifiers.append(match)

        return identifiers[:10]

    def _build_retrieval_query(self, query: str, confused_tokens: List[str]) -> str:
        # Extract identifiers from file list context (like cce_adaptive)
        context_identifiers = self._extract_identifiers_from_context(self.file_list_context)

        # Combine: query + context identifiers + code tokens
        all_terms = context_identifiers[:5] + confused_tokens[:5]
        return f"{query} {' '.join(all_terms)}"

    def retrieve(self, query: str) -> MultiHopRetrievalResult:
        """
        Real CCE retrieval using H_code - H_lang:
        - Only triggers on CODE uncertainty (CCE > 0)
        - Ignores language uncertainty (CCE < 0)
        """
        # Build initial prompt
        if self.file_list_context:
            prompt = f"{query}\\n\\n{self.file_list_context}\\n\\n"
        else:
            prompt = f"{query}\\n\\n"

        # Track state
        retrieved_files = []
        retrieved_content = []
        all_scores = []
        trace = []
        seen_files = set()

        retrieval_count = 0
        last_retrieval_pos = -100

        # Tokenize initial prompt
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        generated_ids = inputs['input_ids']

        # Generate tokens one by one, checking CCE
        for i in range(self.max_gen_tokens):
            with torch.no_grad():
                outputs = self.model(generated_ids)
                logits = outputs.logits[0, -1, :]

            # Compute FULL CCE (H_code - H_lang)
            cce, h_code, h_lang = self._compute_cce(logits)

            # Debug: print first few CCE values
            if i < 3:
                print(f"    Token {i}: CCE={cce:.3f} (H_code={h_code:.2f}, H_lang={h_lang:.2f}, threshold={self.uncertainty_threshold})")

            # Check for CODE uncertainty spike (CCE > threshold)
            tokens_since_last = i - last_retrieval_pos
            in_cooldown = tokens_since_last < self.cooldown_tokens

            if cce > self.uncertainty_threshold and not in_cooldown and retrieval_count < self.max_retrievals:
                # CODE UNCERTAINTY SPIKE - retrieve context
                confused_tokens = self._extract_code_tokens_from_logits(logits)
                retrieval_query = self._build_retrieval_query(query, confused_tokens)

                # Retrieve
                results = self.retriever.retrieve(retrieval_query, top_k=self.top_k, deduplicate=False)
                new_files = [r for r in results if r['source'] not in seen_files]

                if new_files:
                    for r in new_files:
                        seen_files.add(r['source'])
                        retrieved_files.append(r['source'])
                        retrieved_content.append(r['content'])
                        all_scores.append(r['score'])

                    # Add retrieved content to context
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

            # Sample next token
            next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token.to(self.model.device)], dim=-1)

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Build result
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
        # Extract identifiers from file list context
        context_identifiers = self._extract_identifiers_from_context(self.file_list_context)
        # Use identifiers + code tokens, but NOT the original query
        all_terms = context_identifiers[:5] + confused_tokens[:5]
        return ' '.join(all_terms)


class CCEQueryPlusTopKRetriever(RealCCEMultiHopRetriever):
    """Uses original query + context identifiers + confused CODE tokens."""

    def _build_retrieval_query(self, query: str, confused_tokens: List[str]) -> str:
        # Extract identifiers from file list context
        context_identifiers = self._extract_identifiers_from_context(self.file_list_context)
        # Combine: query + identifiers + code tokens
        all_terms = context_identifiers[:5] + confused_tokens[:5]
        return f"{query} {' '.join(all_terms)}"


print("Real CCE Multi-Hop defined (FULL CCE: H_code - H_lang)")
print("  - Computes Contrastive Code Entropy (H_code - H_lang)")
print("  - Only triggers on CODE uncertainty (CCE > 0)")
print("  - Ignores language uncertainty (CCE < 0)")
print("  - Extracts only CODE tokens from logits")
print("  - Threshold: 0.3 (CCE), not 3.0 (raw entropy)")
'''

# Update cell 18 with new implementation
nb['cells'][18]['source'] = [line + '\n' for line in new_class.split('\n')]
nb['cells'][18]['source'][-1] = nb['cells'][18]['source'][-1].rstrip('\n')

with open('C:/Users/rajka/reposynth/research/Week8_Comprehensive_Evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated Real CCE to work like cce_adaptive")
