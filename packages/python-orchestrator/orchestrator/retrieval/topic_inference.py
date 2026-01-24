"""
Topic Inference Module for Adaptive Context Retrieval.

This module infers WHAT the model is uncertain about by analyzing:
1. Top-k predicted tokens (what the model is considering)
2. Recent context (what identifiers/symbols are being referenced)
3. Uncertainty pattern (code vs language uncertainty)

The inferred topic is used to construct a search query for retrieval.

Phase 3, Week 6: Topic Inference & Retrieval
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple
import numpy as np


@dataclass
class TopicResult:
    """Result of topic inference."""
    query: str                           # Search query for retrieval
    confidence: float                    # Confidence in inferred topic [0, 1]
    top_tokens: List[str]                # Top predicted tokens
    extracted_identifiers: List[str]     # Identifiers from context
    inferred_type: str                   # 'api', 'library', 'function', 'variable', 'unknown'
    context_snippet: str                 # Relevant context used for inference

    # File path matching (new!)
    matched_files: List[str] = field(default_factory=list)  # Files matching confused tokens
    file_match_scores: Dict[str, float] = field(default_factory=dict)  # Scores for each file

    # Optional metadata
    metadata: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"TopicResult(query='{self.query}', type={self.inferred_type}, conf={self.confidence:.2f}, matched={self.matched_files[:2]})"


class TopicInferrer:
    """
    Infer what the model is uncertain about from logits and context.

    The inferred topic is used to construct a retrieval query when
    uncertainty is detected during generation.

    Methods:
    1. Top-k tokens: Use top predictions as search terms
    2. Context extraction: Parse recent context for identifiers
    3. Symbol extraction: Identify specific identifiers being referenced
    4. Pattern matching: Detect common uncertainty patterns (imports, APIs, etc.)
    5. File path matching: Match confused tokens against available file paths

    Example:
        >>> inferrer = TopicInferrer(tokenizer)
        >>> inferrer.set_available_files(["src/auth/jwt.py", "src/api/routes.py"])
        >>> result = inferrer.infer_topic(
        ...     logits=model_output.logits,
        ...     context="In our Flask app, the database we use is",
        ...     position=42
        ... )
        >>> print(result.query)  # "Flask database SQLAlchemy PostgreSQL"
        >>> print(result.matched_files)  # ["src/db/models.py"]
    """

    # Common code patterns that indicate what type of context is needed
    IMPORT_PATTERNS = [
        r'import\s+(\w+)',
        r'from\s+(\w+)',
        r'require\s*\(\s*[\'"](\w+)',
    ]

    FUNCTION_CALL_PATTERNS = [
        r'(\w+)\s*\(',
        r'(\w+)\.\w+\s*\(',
    ]

    METHOD_ACCESS_PATTERNS = [
        r'(\w+)\.(\w+)',
    ]

    # Common library/framework indicators
    FRAMEWORK_INDICATORS = {
        'flask': ['Flask', 'route', 'Blueprint', 'request', 'jsonify'],
        'django': ['Django', 'models', 'views', 'urls', 'admin'],
        'react': ['React', 'useState', 'useEffect', 'component', 'props'],
        'express': ['express', 'app.get', 'app.post', 'router', 'middleware'],
        'fastapi': ['FastAPI', 'Depends', 'HTTPException', 'router'],
        'sqlalchemy': ['SQLAlchemy', 'Session', 'Column', 'relationship'],
        'pandas': ['pandas', 'DataFrame', 'Series', 'read_csv'],
        'numpy': ['numpy', 'array', 'ndarray', 'zeros', 'ones'],
        'tensorflow': ['tensorflow', 'keras', 'model', 'layer'],
        'pytorch': ['torch', 'nn', 'tensor', 'Module'],
    }

    def __init__(
        self,
        tokenizer=None,
        top_k: int = 20,
        context_window: int = 300,
        min_token_length: int = 2,
        code_filter_threshold: float = 0.3,
        available_files: Optional[List[str]] = None,
        embedder=None,
        vector_similarity_threshold: float = 0.5,
    ):
        """
        Initialize topic inferrer.

        Args:
            tokenizer: HuggingFace tokenizer for decoding tokens
            top_k: Number of top tokens to consider
            context_window: Characters of context to analyze
            min_token_length: Minimum length for token to be considered
            code_filter_threshold: Threshold for filtering non-code tokens
            available_files: List of file paths in the codebase (for matching)
            embedder: SentenceTransformer model for vector-based file matching
            vector_similarity_threshold: Minimum cosine similarity for vector matching
        """
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.context_window = context_window
        self.min_token_length = min_token_length
        self.code_filter_threshold = code_filter_threshold

        # File path matching
        self.available_files: List[str] = available_files or []
        self._file_tokens: Dict[str, Set[str]] = {}  # Cached tokens for each file (string matching)
        self._file_embeddings: Optional[np.ndarray] = None  # Cached embeddings for vector matching
        self._file_path_texts: List[str] = []  # Human-readable file path texts for embedding
        if available_files:
            self._preprocess_file_paths(available_files)

        # Vector-based matching
        self.embedder = embedder
        self.vector_similarity_threshold = vector_similarity_threshold
        self._use_vector_matching = embedder is not None

        # Cache for recent inferences
        self.last_result: Optional[TopicResult] = None
        self.history: List[TopicResult] = []

    def set_available_files(self, file_paths: List[str]):
        """
        Set the available file paths for matching.

        This enables the inferrer to match confused tokens against
        actual file names in the project, improving retrieval accuracy.

        Args:
            file_paths: List of file paths (e.g., ["src/auth/jwt.py", "src/api/routes.py"])
        """
        self.available_files = file_paths
        self._file_embeddings = None  # Invalidate cache
        self._preprocess_file_paths(file_paths)

    def set_embedder(self, embedder):
        """
        Set the embedder for vector-based file matching.

        Args:
            embedder: SentenceTransformer model or compatible encoder
        """
        self.embedder = embedder
        self._use_vector_matching = embedder is not None
        self._file_embeddings = None  # Invalidate cache

    def _preprocess_file_paths(self, file_paths: List[str]):
        """Extract searchable tokens from each file path."""
        self._file_tokens = {}
        self._file_path_texts = []

        for path in file_paths:
            tokens = self._extract_path_tokens(path)
            self._file_tokens[path] = tokens
            # Create human-readable text for embedding (e.g., "src auth jwt" from "src/auth/jwt.py")
            readable_text = self._path_to_readable_text(path)
            self._file_path_texts.append(readable_text)

        # Invalidate embeddings cache
        self._file_embeddings = None

    def _path_to_readable_text(self, path: str) -> str:
        """Convert file path to human-readable text for embedding."""
        # Split path into components
        parts = re.split(r'[/\\]', path)
        words = []

        for part in parts:
            # Remove extension
            name = re.sub(r'\.[^.]+$', '', part)

            # Split on underscores
            for subpart in name.split('_'):
                if len(subpart) >= 2:
                    words.append(subpart.lower())

            # Split camelCase
            camel_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', name)
            for cp in camel_parts:
                if len(cp) >= 2 and cp.lower() not in words:
                    words.append(cp.lower())

        return ' '.join(words)

    def _extract_path_tokens(self, path: str) -> Set[str]:
        """Extract searchable tokens from a file path."""
        tokens = set()

        # Split path into components
        parts = re.split(r'[/\\]', path)

        for part in parts:
            # Remove extension
            name = re.sub(r'\.[^.]+$', '', part)

            # Add the full name
            tokens.add(name.lower())

            # Split on underscores and add parts
            for subpart in name.split('_'):
                if len(subpart) >= 2:
                    tokens.add(subpart.lower())

            # Split camelCase and add parts
            camel_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', name)
            for cp in camel_parts:
                if len(cp) >= 2:
                    tokens.add(cp.lower())

        return tokens

    def _match_files(
        self,
        code_tokens: List[str],
        identifiers: List[str],
        top_k: int = 3,
        original_query: str = "",
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Match confused tokens against available file paths.

        Uses vector similarity when embedder is available, falls back to
        string matching otherwise. Vector matching handles semantic similarity
        (e.g., "auth" matches "authenticate.py").

        IMPORTANT: Only uses code_tokens that are RELEVANT to the original query.
        This filters out tangential mentions (e.g., model mentions "auth" while
        answering a question about "app structure" - "auth" is noise).

        Args:
            code_tokens: Tokens from model's uncertain predictions (CCE)
            identifiers: Identifiers extracted from context (NOT used for vector matching)
            top_k: Number of top matches to return
            original_query: The user's original query (for relevance filtering)

        Returns:
            Tuple of (matched_files, score_dict)
        """
        if not self.available_files:
            return [], {}

        # ONLY use code tokens from CCE - these are what the model is confused about
        # Filter aggressively to keep only meaningful code tokens
        search_terms = []
        for token in code_tokens[:10]:
            clean = token.lower().strip()
            # Skip short tokens, special tokens, and common noise
            if len(clean) >= 3 and self._is_meaningful_token(clean):
                search_terms.append(clean)

        if not search_terms:
            return [], {}

        # CRITICAL: Filter tokens by relevance to original query
        # Only use confused tokens that are semantically related to the question
        if original_query and self._use_vector_matching and self.embedder is not None:
            search_terms = self._filter_tokens_by_query_relevance(
                search_terms, original_query, threshold=0.25
            )
            if not search_terms:
                return [], {}

        # Use vector matching if embedder is available
        if self._use_vector_matching and self.embedder is not None:
            return self._match_files_vector(search_terms, top_k)
        else:
            # For string matching, also include identifiers as fallback
            string_terms = set(search_terms)
            for ident in identifiers[:5]:
                for part in ident.split('.'):
                    if len(part) >= 3:
                        string_terms.add(part.lower())
            return self._match_files_string(string_terms, top_k)

    def _filter_tokens_by_query_relevance(
        self,
        tokens: List[str],
        query: str,
        threshold: float = 0.25,
    ) -> List[str]:
        """
        Filter confused tokens to only those relevant to the original query.

        This prevents retrieving files based on tangential mentions.
        Example:
            Query: "What is the app structure?"
            Tokens: ["auth", "jwt", "blueprint", "factory"]
            After filtering: ["blueprint", "factory"] (relevant to structure)
            Filtered out: ["auth", "jwt"] (not relevant to structure question)

        Args:
            tokens: Confused tokens from CCE
            query: Original user query
            threshold: Minimum similarity to keep token (0.25 = loosely related)

        Returns:
            Filtered list of tokens relevant to the query
        """
        if not tokens or not query:
            return tokens

        # Embed the query
        query_embedding = self.embedder.encode([query])[0]
        query_norm = np.linalg.norm(query_embedding) + 1e-8

        # Embed all tokens at once (batch)
        token_embeddings = self.embedder.encode(tokens)

        # Keep tokens that are similar to the query
        relevant_tokens = []
        for i, token in enumerate(tokens):
            token_emb = token_embeddings[i]
            token_norm = np.linalg.norm(token_emb) + 1e-8

            # Cosine similarity
            similarity = np.dot(query_embedding, token_emb) / (query_norm * token_norm)

            if similarity >= threshold:
                relevant_tokens.append(token)

        return relevant_tokens

    def _is_meaningful_token(self, token: str) -> bool:
        """Check if a token is meaningful for file matching (not noise)."""
        # Skip special tokens
        if token.startswith('<') or token.startswith('</') or token.endswith('>'):
            return False

        # Skip common Python/code noise
        noise_tokens = {
            'self', 'def', 'class', 'import', 'from', 'return', 'none',
            'true', 'false', 'null', 'undefined', 'function', 'const',
            'let', 'var', 'async', 'await', 'if', 'else', 'for', 'while',
            'try', 'except', 'with', 'lambda', 'yield', 'raise', 'pass',
            'break', 'continue', 'global', 'nonlocal', 'assert', 'del',
            'print', 'input', 'open', 'close', 'read', 'write', 'len',
            'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
            'type', 'isinstance', 'hasattr', 'getattr', 'setattr',
        }
        if token in noise_tokens:
            return False

        # Skip common English words that aren't code-related
        english_noise = {
            'the', 'this', 'that', 'what', 'which', 'where', 'when', 'how',
            'why', 'who', 'also', 'just', 'only', 'more', 'most', 'some',
            'any', 'all', 'each', 'every', 'both', 'few', 'many', 'much',
            'other', 'another', 'such', 'same', 'different', 'new', 'old',
            'first', 'last', 'next', 'previous', 'current', 'following',
            'use', 'used', 'using', 'make', 'made', 'making', 'get', 'got',
            'getting', 'set', 'setting', 'can', 'could', 'would', 'should',
            'will', 'shall', 'may', 'might', 'must', 'need', 'want', 'like',
            'know', 'think', 'see', 'look', 'find', 'give', 'take', 'come',
            'going', 'have', 'has', 'had', 'having', 'been', 'being', 'was',
            'were', 'are', 'does', 'did', 'doing', 'done', 'please', 'thank',
            'thanks', 'hello', 'yes', 'not', 'and', 'but', 'then', 'now',
            'here', 'there', 'very', 'really', 'actually', 'basically',
        }
        if token in english_noise:
            return False

        return True

    def _match_files_vector(
        self,
        search_terms: List[str],
        top_k: int = 3
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Match files using vector similarity (semantic matching).

        Matches EACH token individually against file paths, then takes
        the maximum similarity. This prevents signal dilution from
        combining multiple tokens into one query.

        This handles cases like:
        - "auth" matching "authenticate.py"
        - "jwt" matching "token_handler.py"
        - "config" matching "settings.py"
        """
        # Ensure file embeddings are computed
        if self._file_embeddings is None:
            self._file_embeddings = self.embedder.encode(self._file_path_texts)

        # Embed each search term individually (batch for efficiency)
        term_embeddings = self.embedder.encode(search_terms)

        # Score each file by MAX similarity to any search term
        scores: Dict[str, float] = {}
        best_matches: Dict[str, str] = {}  # Track which term matched best

        for i, path in enumerate(self.available_files):
            file_emb = self._file_embeddings[i]
            file_norm = np.linalg.norm(file_emb) + 1e-8

            max_sim = 0.0
            best_term = ""

            # Check similarity with EACH term individually
            for j, term in enumerate(search_terms):
                term_emb = term_embeddings[j]
                term_norm = np.linalg.norm(term_emb) + 1e-8

                # Cosine similarity
                sim = np.dot(term_emb, file_emb) / (term_norm * file_norm)

                if sim > max_sim:
                    max_sim = sim
                    best_term = term

            # Only include if above threshold
            if max_sim >= self.vector_similarity_threshold:
                scores[path] = float(max_sim)
                best_matches[path] = best_term

        # Sort by score and return top-k
        sorted_files = sorted(scores.items(), key=lambda x: -x[1])
        matched = [f for f, s in sorted_files[:top_k]]

        return matched, scores

    def _match_files_string(
        self,
        search_terms: Set[str],
        top_k: int = 3
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Match files using string/token matching (exact matching fallback).

        Original implementation - used when no embedder is available.
        """
        scores: Dict[str, float] = {}
        for path, path_tokens in self._file_tokens.items():
            score = 0.0
            matched_terms = search_terms & path_tokens

            if matched_terms:
                # Base score: number of matching tokens
                score = len(matched_terms)

                # Bonus for exact file name match
                file_name = re.split(r'[/\\]', path)[-1]
                file_name_no_ext = re.sub(r'\.[^.]+$', '', file_name).lower()
                if file_name_no_ext in search_terms:
                    score += 2.0

                # Bonus for directory match
                dir_parts = re.split(r'[/\\]', path)[:-1]
                for dir_part in dir_parts:
                    if dir_part.lower() in search_terms:
                        score += 0.5

            if score > 0:
                scores[path] = score

        # Sort by score and return top-k
        sorted_files = sorted(scores.items(), key=lambda x: -x[1])
        matched = [f for f, s in sorted_files[:top_k]]

        return matched, scores

    def infer_topic(
        self,
        logits: np.ndarray,
        context: str,
        position: int,
        uncertainty_value: Optional[float] = None,
        original_query: str = "",
    ) -> TopicResult:
        """
        Infer the topic of uncertainty from logits and context.

        Args:
            logits: Model output logits [vocab_size]
            context: Generated text up to this point
            position: Token position in generation
            uncertainty_value: Optional uncertainty score for confidence weighting
            original_query: Original user query (for filtering relevant tokens)

        Returns:
            TopicResult with search query and metadata
        """
        # 1. Extract top-k tokens
        top_tokens = self._get_top_tokens(logits)

        # 2. Filter to code-relevant tokens
        code_tokens = self._filter_code_tokens(top_tokens)

        # 3. Extract identifiers from context
        recent_context = context[-self.context_window:]
        identifiers = self._extract_identifiers(recent_context)

        # 4. Detect framework/library context
        framework = self._detect_framework(recent_context, identifiers)

        # 5. Infer uncertainty type
        uncertainty_type = self._infer_type(recent_context, code_tokens, identifiers)

        # 6. Build search query
        query = self._build_query(code_tokens, identifiers, framework, uncertainty_type)

        # 7. Match against available file paths
        # Pass original_query to filter confused tokens by relevance
        matched_files, file_scores = self._match_files(
            code_tokens, identifiers, original_query=original_query
        )

        # 8. Calculate confidence
        confidence = self._calculate_confidence(
            code_tokens, identifiers, framework, uncertainty_value
        )

        # Boost confidence if we have file matches
        if matched_files:
            confidence = min(1.0, confidence + 0.1)

        result = TopicResult(
            query=query,
            confidence=confidence,
            top_tokens=top_tokens[:10],
            extracted_identifiers=identifiers,
            inferred_type=uncertainty_type,
            context_snippet=recent_context[-100:],
            matched_files=matched_files,
            file_match_scores=file_scores,
            metadata={
                'position': position,
                'framework': framework,
                'uncertainty_value': uncertainty_value,
                'search_terms': list(set(t.lower() for t in code_tokens[:5] + identifiers[:5])),
            }
        )

        self.last_result = result
        self.history.append(result)

        return result

    def _get_top_tokens(self, logits: np.ndarray) -> List[str]:
        """Get top-k predicted tokens from logits."""
        if self.tokenizer is None:
            return []

        # Get top-k indices
        top_indices = np.argsort(logits)[-self.top_k:][::-1]

        # Decode tokens
        tokens = []
        for idx in top_indices:
            try:
                token = self.tokenizer.decode([idx])
                if token and len(token.strip()) >= self.min_token_length:
                    tokens.append(token.strip())
            except Exception:
                continue

        return tokens

    def _filter_code_tokens(self, tokens: List[str]) -> List[str]:
        """Filter tokens to keep only code-relevant ones."""
        # Common English words to filter out
        stopwords = {
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
        }

        filtered = []
        for token in tokens:
            clean = token.lower().strip()
            # Keep if not a stopword and looks code-like
            if clean not in stopwords:
                # Check if it looks like code (has letters, underscores, etc.)
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', clean):
                    filtered.append(token)
                # Keep tokens with dots (method chains)
                elif '.' in token:
                    filtered.append(token)
                # Keep tokens that look like imports/paths
                elif '/' in token or '::' in token:
                    filtered.append(token)

        return filtered

    def _extract_identifiers(self, context: str) -> List[str]:
        """Extract variable/function/class names from context."""
        identifiers = set()

        # Pattern for identifiers (variable names, function names, etc.)
        # Matches: foo, bar_baz, FooBar, foo123, _private
        identifier_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]{2,})\b'
        matches = re.findall(identifier_pattern, context)

        for match in matches:
            # Filter out common keywords
            if match.lower() not in {
                'def', 'class', 'import', 'from', 'return', 'if', 'else',
                'for', 'while', 'try', 'except', 'with', 'as', 'and', 'or',
                'not', 'true', 'false', 'none', 'null', 'undefined',
                'function', 'const', 'let', 'var', 'async', 'await',
            }:
                identifiers.add(match)

        # Extract method chain patterns: foo.bar.baz
        chain_pattern = r'(\w+(?:\.\w+)+)'
        chains = re.findall(chain_pattern, context)
        identifiers.update(chains)

        # Extract import statements
        for pattern in self.IMPORT_PATTERNS:
            imports = re.findall(pattern, context)
            identifiers.update(imports)

        return list(identifiers)[-10:]  # Keep most recent 10

    def _detect_framework(
        self,
        context: str,
        identifiers: List[str]
    ) -> Optional[str]:
        """Detect which framework/library is being used."""
        context_lower = context.lower()
        identifier_set = set(id.lower() for id in identifiers)

        best_match = None
        best_score = 0

        for framework, indicators in self.FRAMEWORK_INDICATORS.items():
            score = 0
            for indicator in indicators:
                if indicator.lower() in context_lower:
                    score += 2
                if indicator.lower() in identifier_set:
                    score += 1

            if score > best_score:
                best_score = score
                best_match = framework

        return best_match if best_score >= 2 else None

    def _infer_type(
        self,
        context: str,
        code_tokens: List[str],
        identifiers: List[str]
    ) -> str:
        """Infer the type of uncertainty (what kind of context is needed)."""
        # Check for import uncertainty
        if re.search(r'import\s*$|from\s+\w+\s+import\s*$', context):
            return 'import'

        # Check for API/library call
        if re.search(r'\.\s*$|\(\s*$', context):
            return 'api'

        # Check for function/method name
        if re.search(r'def\s+\w*$|function\s+\w*$', context):
            return 'function'

        # Check for variable assignment
        if re.search(r'=\s*$|:\s*$', context):
            return 'variable'

        # Check for type annotation
        if re.search(r':\s*$|->\s*$', context):
            return 'type'

        # Default based on tokens
        if code_tokens:
            return 'api'

        return 'unknown'

    def _build_query(
        self,
        code_tokens: List[str],
        identifiers: List[str],
        framework: Optional[str],
        uncertainty_type: str
    ) -> str:
        """Build search query from extracted information."""
        query_parts = []

        # Add framework context
        if framework:
            query_parts.append(framework)

        # Add type-specific prefix
        type_prefixes = {
            'import': 'import',
            'api': 'api usage',
            'function': 'function implementation',
            'variable': 'variable',
            'type': 'type annotation',
        }
        if uncertainty_type in type_prefixes:
            query_parts.append(type_prefixes[uncertainty_type])

        # Add identifiers (most important)
        query_parts.extend(identifiers[:5])

        # Add code tokens
        query_parts.extend(code_tokens[:5])

        # Deduplicate while preserving order
        seen = set()
        unique_parts = []
        for part in query_parts:
            if part.lower() not in seen:
                seen.add(part.lower())
                unique_parts.append(part)

        return ' '.join(unique_parts[:8])  # Limit query length

    def _calculate_confidence(
        self,
        code_tokens: List[str],
        identifiers: List[str],
        framework: Optional[str],
        uncertainty_value: Optional[float]
    ) -> float:
        """Calculate confidence in the inferred topic."""
        confidence = 0.5  # Base confidence

        # More code tokens = higher confidence
        if len(code_tokens) >= 3:
            confidence += 0.15
        elif len(code_tokens) >= 1:
            confidence += 0.05

        # Identifiers help
        if len(identifiers) >= 3:
            confidence += 0.15
        elif len(identifiers) >= 1:
            confidence += 0.05

        # Framework detection helps
        if framework:
            confidence += 0.1

        # High uncertainty = more confident we need retrieval
        if uncertainty_value is not None and uncertainty_value > 0.5:
            confidence += 0.1

        return min(1.0, confidence)

    def reset(self):
        """Reset inferrer state."""
        self.last_result = None
        self.history = []

    def get_history(self) -> List[TopicResult]:
        """Get inference history."""
        return self.history.copy()


# ============================================================================
# SPECIALIZED INFERRERS
# ============================================================================

class ImportTopicInferrer(TopicInferrer):
    """Specialized inferrer for import statement uncertainty."""

    def infer_topic(
        self,
        logits: np.ndarray,
        context: str,
        position: int,
        uncertainty_value: Optional[float] = None,
    ) -> TopicResult:
        """Specialized inference for imports."""
        result = super().infer_topic(logits, context, position, uncertainty_value)

        # Add import-specific query terms
        if 'import' not in result.query.lower():
            result.query = f"import {result.query}"

        result.inferred_type = 'import'
        return result


class APITopicInferrer(TopicInferrer):
    """Specialized inferrer for API call uncertainty."""

    def infer_topic(
        self,
        logits: np.ndarray,
        context: str,
        position: int,
        uncertainty_value: Optional[float] = None,
    ) -> TopicResult:
        """Specialized inference for API calls."""
        result = super().infer_topic(logits, context, position, uncertainty_value)

        # Extract the object being called
        method_match = re.search(r'(\w+)\.\s*$', context)
        if method_match:
            obj_name = method_match.group(1)
            if obj_name not in result.query:
                result.query = f"{obj_name} methods {result.query}"

        result.inferred_type = 'api'
        return result


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_topic_inferrer(
    tokenizer=None,
    specialized: Optional[str] = None,
    **kwargs
) -> TopicInferrer:
    """
    Factory function to create topic inferrers.

    Args:
        tokenizer: HuggingFace tokenizer
        specialized: Optional specialization ('import', 'api')
        **kwargs: Additional arguments

    Returns:
        Configured TopicInferrer
    """
    if specialized == 'import':
        return ImportTopicInferrer(tokenizer=tokenizer, **kwargs)
    elif specialized == 'api':
        return APITopicInferrer(tokenizer=tokenizer, **kwargs)
    else:
        return TopicInferrer(tokenizer=tokenizer, **kwargs)
