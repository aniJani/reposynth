"""
Token Classification Module for CCE Computation

This module provides keyword-based and embedding-based classification of tokens
into 'code', 'language', or 'other' categories for Contrastive Code Entropy (CCE)
calculation.

Week 2, Day 3-5: Hybrid Token Classification
"""

from typing import Literal, Optional, Dict, Set, Tuple, List
import re


# ============================================================================
# PROGRAMMING KEYWORDS (GENERIC SYNTAX)
# ============================================================================

# Python keywords and built-ins
PYTHON_KEYWORDS = {
    # Control flow
    'if', 'else', 'elif', 'for', 'while', 'break', 'continue', 'pass',
    'return', 'yield', 'raise', 'try', 'except', 'finally', 'with', 'as',

    # Definitions
    'def', 'class', 'lambda', 'async', 'await',

    # Logic
    'and', 'or', 'not', 'in', 'is', 'None', 'True', 'False',

    # Imports
    'import', 'from', 'as',

    # Common built-ins
    'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
    'list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool',
    'open', 'read', 'write', 'close', 'append',
}

# JavaScript/TypeScript keywords
JS_KEYWORDS = {
    # Control flow
    'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default',
    'break', 'continue', 'return',

    # Definitions
    'function', 'const', 'let', 'var', 'class', 'async', 'await',

    # Logic
    'true', 'false', 'null', 'undefined', 'typeof', 'instanceof',

    # Imports
    'import', 'export', 'from', 'require', 'module', 'exports',

    # Common methods
    'console', 'log', 'push', 'pop', 'map', 'filter', 'reduce',
    'forEach', 'find', 'slice', 'splice',
}

# Java keywords
JAVA_KEYWORDS = {
    'public', 'private', 'protected', 'static', 'final', 'abstract',
    'class', 'interface', 'extends', 'implements', 'new', 'this', 'super',
    'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break',
    'return', 'void', 'int', 'String', 'boolean', 'true', 'false', 'null',
}

# Go keywords
GO_KEYWORDS = {
    'func', 'package', 'import', 'var', 'const', 'type', 'struct',
    'interface', 'if', 'else', 'for', 'range', 'switch', 'case',
    'return', 'defer', 'go', 'chan', 'select', 'map', 'make', 'new',
    'true', 'false', 'nil',
}

# Rust keywords
RUST_KEYWORDS = {
    'fn', 'let', 'mut', 'const', 'static', 'struct', 'enum', 'impl',
    'trait', 'if', 'else', 'for', 'while', 'loop', 'match', 'return',
    'pub', 'mod', 'use', 'crate', 'self', 'true', 'false', 'None', 'Some',
}

# Common operators and symbols (all languages)
CODE_OPERATORS = {
    '+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', '<=', '>=',
    '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '++', '--',
    '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=',
    '{', '}', '[', ']', '(', ')', ';', ':', ',', '.', '->', '=>',
}

# ============================================================================
# DOMAIN-SPECIFIC LIBRARY NAMES & APIS (CRITICAL FOR CCE)
# ============================================================================

# Python Data Science Libraries
PYTHON_DATA_LIBS = {
    # Libraries
    'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'sklearn',
    'tensorflow', 'torch', 'keras', 'plotly',

    # NumPy
    'np', 'array', 'ndarray', 'dtype', 'reshape', 'transpose', 'dot',
    'matmul', 'linspace', 'arange', 'zeros', 'ones', 'eye',

    # Pandas
    'pd', 'DataFrame', 'Series', 'read_csv', 'read_json', 'read_excel',
    'groupby', 'pivot', 'merge', 'concat', 'dropna', 'fillna', 'loc', 'iloc',

    # Scikit-learn
    'fit', 'transform', 'fit_transform', 'predict', 'score', 'train_test_split',
}

# Python Web Frameworks
PYTHON_WEB_LIBS = {
    # Libraries
    'flask', 'django', 'fastapi', 'requests', 'aiohttp', 'sqlalchemy',

    # FastAPI
    'FastAPI', 'APIRouter', 'HTTPException', 'Request', 'Response',
    'Depends', 'Query', 'Path', 'Body', 'get', 'post', 'put', 'delete',

    # Requests
    'requests', 'get', 'post', 'put', 'delete', 'patch', 'session',
    'json', 'headers', 'params', 'data', 'files', 'auth',

    # Flask
    'Flask', 'render_template', 'redirect', 'url_for', 'request', 'session',
    'jsonify', 'make_response', 'abort',
}

# JavaScript/React Libraries
JS_REACT_LIBS = {
    # Libraries
    'react', 'vue', 'angular', 'next', 'express', 'axios', 'lodash',

    # React Hooks
    'useState', 'useEffect', 'useContext', 'useReducer', 'useCallback',
    'useMemo', 'useRef', 'useImperativeHandle', 'useLayoutEffect',

    # React Core
    'React', 'Component', 'PureComponent', 'createElement', 'Fragment',
    'props', 'state', 'render', 'componentDidMount', 'componentDidUpdate',

    # JSX
    'className', 'onClick', 'onChange', 'onSubmit', 'onKeyDown', 'onKeyUp',
    'value', 'checked', 'disabled', 'placeholder',
}

# Firebase & Cloud Services
CLOUD_SERVICES = {
    # Firebase
    'firebase', 'Firebase', 'firestore', 'Firestore', 'auth', 'Auth',
    'database', 'storage', 'functions', 'messaging',
    'admin', 'initializeApp', 'getAuth', 'signInWithEmailAndPassword',
    'createUserWithEmailAndPassword', 'signOut', 'onAuthStateChanged',

    # AWS
    'aws', 'boto3', 's3', 'lambda', 'dynamodb', 'ec2', 'rds',

    # Other
    'mongodb', 'redis', 'postgres', 'mysql', 'elasticsearch',
}

# Testing Libraries
TESTING_LIBS = {
    'pytest', 'unittest', 'jest', 'mocha', 'chai', 'jasmine',
    'test', 'describe', 'it', 'expect', 'assert', 'mock', 'spy',
    'fixture', 'setUp', 'tearDown', 'beforeEach', 'afterEach',
}

# All code keywords combined
CODE_KEYWORDS = (
    PYTHON_KEYWORDS | JS_KEYWORDS | JAVA_KEYWORDS | GO_KEYWORDS | RUST_KEYWORDS |
    PYTHON_DATA_LIBS | PYTHON_WEB_LIBS | JS_REACT_LIBS | CLOUD_SERVICES | TESTING_LIBS
)

# ============================================================================
# LANGUAGE/DOCUMENTATION KEYWORDS
# ============================================================================

# Common English words for documentation and explanations
# NOTE: Exclude words that are also programming keywords (for, and, or, not, in, is, etc.)
# Programming keywords take precedence
LANGUAGE_WORDS = {
    # Question words
    'what', 'how', 'why', 'when', 'where', 'which', 'who', 'whom',

    # Action verbs (documentation) - excluding overlaps with code keywords
    'explain', 'show', 'tell', 'help', 'search',
    'generate', 'implement', 'refactor', 'optimize', 'improve',
    'summarize', 'analyze', 'review', 'check', 'validate',

    # Documentation keywords
    'comment', 'docstring', 'documentation', 'example', 'usage',
    'note', 'warning', 'todo', 'fixme', 'bug', 'issue',

    # Common words - excluding programming keywords
    'the', 'a', 'an', 'that', 'these', 'those',
    'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'does', 'did', 'will', 'would', 'should',
    'can', 'could', 'may', 'might', 'must', 'shall',
    'to', 'without', 'by', 'at', 'on',
    'of', 'off', 'up', 'down', 'over', 'under', 'above', 'below',
    'but', 'nor', 'so', 'yet',
    'my', 'your', 'his', 'her', 'its', 'our', 'their',
    'I', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',

    # Task-related - excluding overlaps
    'summary', 'debug',

    # Comparison/quality
    'better', 'worse', 'best', 'worst', 'good', 'bad', 'great', 'poor',
    'slow', 'quick', 'simple', 'complex', 'easy', 'hard',
}

# ============================================================================
# KEYWORD-BASED CLASSIFIER (FAST PATH)
# ============================================================================

class KeywordClassifier:
    """
    Fast keyword-based token classification.

    This is the first stage (fast path) of the hybrid classifier.
    Maps tokens to 'code', 'language', or None (for embedding fallback).
    """

    def __init__(self):
        """Initialize keyword sets."""
        self.code_keywords = CODE_KEYWORDS
        self.code_operators = CODE_OPERATORS
        self.language_words = LANGUAGE_WORDS

        # Precompute lowercase sets for case-insensitive matching
        self.code_keywords_lower = {k.lower() for k in self.code_keywords}
        self.language_words_lower = {w.lower() for w in self.language_words}

    def classify(self, token: str) -> Optional[Literal["code", "language"]]:
        """
        Classify a token using keyword lookup.

        Args:
            token: The token string to classify

        Returns:
            'code' if token is a programming keyword/operator
            'language' if token is a common English word
            None if token should be classified by embeddings (fallback)
        """
        # Clean token
        token_clean = token.strip()

        # Check exact match for operators
        if token_clean in self.code_operators:
            return 'code'

        # Check case-insensitive match for keywords
        token_lower = token_clean.lower()

        if token_lower in self.code_keywords_lower:
            return 'code'

        if token_lower in self.language_words_lower:
            return 'language'

        # Not found in keyword sets - return None for embedding fallback
        return None

    def get_coverage(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calculate classification coverage for a list of tokens.

        Args:
            tokens: List of token strings

        Returns:
            Dictionary with coverage statistics:
            - 'code': proportion classified as code
            - 'language': proportion classified as language
            - 'unknown': proportion requiring embedding fallback
        """
        if not tokens:
            return {'code': 0.0, 'language': 0.0, 'unknown': 0.0}

        counts = {'code': 0, 'language': 0, 'unknown': 0}

        for token in tokens:
            result = self.classify(token)
            if result is None:
                counts['unknown'] += 1
            else:
                counts[result] += 1

        total = len(tokens)
        return {
            'code': counts['code'] / total,
            'language': counts['language'] / total,
            'unknown': counts['unknown'] / total,
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def build_code_keyword_set() -> Set[str]:
    """Build the complete set of code keywords."""
    return CODE_KEYWORDS.copy()


def build_language_keyword_set() -> Set[str]:
    """Build the complete set of language keywords."""
    return LANGUAGE_WORDS.copy()


def get_keyword_stats() -> Dict[str, int]:
    """Get statistics about keyword sets."""
    return {
        'code_keywords': len(CODE_KEYWORDS),
        'code_operators': len(CODE_OPERATORS),
        'language_words': len(LANGUAGE_WORDS),
        'total_keywords': len(CODE_KEYWORDS) + len(CODE_OPERATORS) + len(LANGUAGE_WORDS),
        'python_specific': len(PYTHON_KEYWORDS) + len(PYTHON_DATA_LIBS) + len(PYTHON_WEB_LIBS),
        'js_specific': len(JS_KEYWORDS) + len(JS_REACT_LIBS),
        'domain_specific': len(PYTHON_DATA_LIBS) + len(PYTHON_WEB_LIBS) + len(JS_REACT_LIBS) + len(CLOUD_SERVICES),
    }


# ============================================================================
# EMBEDDING-BASED CLASSIFIER (SLOW PATH / FALLBACK)
# ============================================================================

class EmbeddingClassifier:
    """
    Embedding-based token classification using sentence-transformers.

    This is the second stage (slow path) of the hybrid classifier.
    Uses semantic similarity to code/language prototypes for classification.
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        margin: float = 0.05,
        use_cache: bool = True
    ):
        """
        Initialize embedding classifier.

        Args:
            model_name: SentenceTransformer model name
            margin: Similarity margin for classification (default 0.05)
            use_cache: Whether to cache token embeddings
        """
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            raise ImportError(
                "sentence-transformers and numpy required for EmbeddingClassifier. "
                "Install with: pip install sentence-transformers numpy"
            )

        self.model = SentenceTransformer(model_name)
        self.margin = margin
        self.use_cache = use_cache

        # Embedding cache for tokens
        self.embedding_cache: Dict[str, 'np.ndarray'] = {}

        # Build prototypes
        self.code_prototype = self._build_code_prototype()
        self.language_prototype = self._build_language_prototype()

    def _build_code_prototype(self) -> 'np.ndarray':
        """
        Build code prototype from examples.

        Important: 50% generic syntax, 50% domain-specific terms
        to address Week 1 POC issues.
        """
        import numpy as np

        code_examples = [
            # Generic syntax (25 examples - 50%)
            'function', 'class', 'import', 'return', 'if', 'else', 'for', 'while',
            'const', 'let', 'var', 'def', 'async', 'await', 'try', 'catch',
            'public', 'private', 'static', 'void', 'int', 'string', 'boolean',
            'array', 'object',

            # Domain-specific libraries & APIs (25 examples - 50%)
            'pandas', 'numpy', 'requests', 'firebase', 'react', 'useState',
            'useEffect', 'FastAPI', 'express', 'axios', 'flask', 'django',
            'DataFrame', 'read_csv', 'get', 'post', 'auth', 'database',
            'query', 'model', 'router', 'component', 'props', 'state',
            'render',
        ]

        embeddings = self.model.encode(code_examples)
        prototype = np.mean(embeddings, axis=0).reshape(1, -1)
        return prototype

    def _build_language_prototype(self) -> 'np.ndarray':
        """Build language prototype from examples."""
        import numpy as np

        language_examples = [
            'explain', 'describe', 'summarize', 'how', 'what', 'why', 'when',
            'where', 'which', 'show', 'tell', 'help', 'write', 'create',
            'the', 'a', 'an', 'this', 'that', 'these', 'those', 'is', 'are',
            'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'can', 'could', 'may', 'might',
            'understand', 'implement', 'improve', 'refactor', 'optimize',
            'better', 'good', 'bad', 'simple', 'complex', 'easy', 'hard',
            'question', 'answer', 'comment', 'documentation',
        ]

        embeddings = self.model.encode(language_examples)
        prototype = np.mean(embeddings, axis=0).reshape(1, -1)
        return prototype

    def _get_embedding(self, token: str) -> 'np.ndarray':
        """Get embedding for a token (with caching)."""
        if self.use_cache and token in self.embedding_cache:
            return self.embedding_cache[token]

        embedding = self.model.encode([token])[0]

        if self.use_cache:
            self.embedding_cache[token] = embedding

        return embedding

    def classify(self, token: str) -> Optional[Literal["code", "language"]]:
        """
        Classify a token using embedding similarity.

        Args:
            token: The token string to classify

        Returns:
            'code' if more similar to code prototype
            'language' if more similar to language prototype
            'other' if ambiguous (within margin)
        """
        from sklearn.metrics.pairwise import cosine_similarity

        # Get token embedding
        token_emb = self._get_embedding(token).reshape(1, -1)

        # Compute similarities to prototypes
        sim_code = cosine_similarity(token_emb, self.code_prototype)[0][0]
        sim_lang = cosine_similarity(token_emb, self.language_prototype)[0][0]

        # Classification with margin
        diff = sim_code - sim_lang

        if diff > self.margin:
            return 'code'
        elif diff < -self.margin:
            return 'language'
        else:
            # Too ambiguous - return 'other'
            return None

    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()


# ============================================================================
# HYBRID CLASSIFIER (KEYWORD + EMBEDDING)
# ============================================================================

class HybridClassifier:
    """
    Two-stage hybrid token classifier.

    Stage 1 (fast path): Keyword lookup
    Stage 2 (slow path): Embedding similarity

    This approach balances speed and coverage.
    """

    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        embedding_margin: float = 0.05,
        use_embedding_cache: bool = True
    ):
        """
        Initialize hybrid classifier.

        Args:
            embedding_model: SentenceTransformer model name
            embedding_margin: Similarity margin for embedding classification
            use_embedding_cache: Whether to cache embeddings
        """
        self.keyword_classifier = KeywordClassifier()
        self.embedding_classifier = EmbeddingClassifier(
            model_name=embedding_model,
            margin=embedding_margin,
            use_cache=use_embedding_cache
        )

        # Diagnostic counters
        self.stats = {
            'keyword_hits': 0,
            'embedding_hits': 0,
            'other': 0,
        }

    def classify(self, token: str) -> Literal["code", "language", "other"]:
        """
        Classify a token using two-stage lookup.

        Args:
            token: The token string to classify

        Returns:
            'code', 'language', or 'other'
        """
        # Stage 1: Fast keyword lookup
        keyword_result = self.keyword_classifier.classify(token)

        if keyword_result is not None:
            self.stats['keyword_hits'] += 1
            return keyword_result

        # Stage 2: Embedding similarity
        embedding_result = self.embedding_classifier.classify(token)

        if embedding_result is not None:
            self.stats['embedding_hits'] += 1
            return embedding_result

        # No classification possible
        self.stats['other'] += 1
        return 'other'

    def get_stats(self) -> Dict[str, int]:
        """Get diagnostic statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset diagnostic statistics."""
        self.stats = {
            'keyword_hits': 0,
            'embedding_hits': 0,
            'other': 0,
        }

    def get_coverage(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calculate classification coverage for a list of tokens.

        Returns:
            Dictionary with coverage breakdown:
            - 'code': proportion classified as code
            - 'language': proportion classified as language
            - 'other': proportion classified as other
            - 'keyword_coverage': proportion resolved by keywords
            - 'embedding_coverage': proportion resolved by embeddings
        """
        if not tokens:
            return {
                'code': 0.0,
                'language': 0.0,
                'other': 0.0,
                'keyword_coverage': 0.0,
                'embedding_coverage': 0.0,
            }

        # Reset stats
        self.reset_stats()

        # Classify all tokens
        counts = {'code': 0, 'language': 0, 'other': 0}
        for token in tokens:
            result = self.classify(token)
            counts[result] += 1

        total = len(tokens)

        return {
            'code': counts['code'] / total,
            'language': counts['language'] / total,
            'other': counts['other'] / total,
            'keyword_coverage': self.stats['keyword_hits'] / total,
            'embedding_coverage': self.stats['embedding_hits'] / total,
        }


# ============================================================================
# FUNCTIONAL INTERFACE
# ============================================================================

# Module-level classifier instance for functional interface
_default_classifier = None


def classify_token(token: str) -> Optional[Literal["code", "language"]]:
    """
    Classify a single token using keyword lookup.

    This is a convenience function that uses a module-level classifier instance.

    Args:
        token: The token string to classify

    Returns:
        'code', 'language', or None (unknown)
    """
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = KeywordClassifier()
    return _default_classifier.classify(token)
