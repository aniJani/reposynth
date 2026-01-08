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

    # Optional metadata
    metadata: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"TopicResult(query='{self.query}', type={self.inferred_type}, conf={self.confidence:.2f})"


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

    Example:
        >>> inferrer = TopicInferrer(tokenizer)
        >>> result = inferrer.infer_topic(
        ...     logits=model_output.logits,
        ...     context="In our Flask app, the database we use is",
        ...     position=42
        ... )
        >>> print(result.query)  # "Flask database SQLAlchemy PostgreSQL"
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
    ):
        """
        Initialize topic inferrer.

        Args:
            tokenizer: HuggingFace tokenizer for decoding tokens
            top_k: Number of top tokens to consider
            context_window: Characters of context to analyze
            min_token_length: Minimum length for token to be considered
            code_filter_threshold: Threshold for filtering non-code tokens
        """
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.context_window = context_window
        self.min_token_length = min_token_length
        self.code_filter_threshold = code_filter_threshold

        # Cache for recent inferences
        self.last_result: Optional[TopicResult] = None
        self.history: List[TopicResult] = []

    def infer_topic(
        self,
        logits: np.ndarray,
        context: str,
        position: int,
        uncertainty_value: Optional[float] = None,
    ) -> TopicResult:
        """
        Infer the topic of uncertainty from logits and context.

        Args:
            logits: Model output logits [vocab_size]
            context: Generated text up to this point
            position: Token position in generation
            uncertainty_value: Optional uncertainty score for confidence weighting

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

        # 7. Calculate confidence
        confidence = self._calculate_confidence(
            code_tokens, identifiers, framework, uncertainty_value
        )

        result = TopicResult(
            query=query,
            confidence=confidence,
            top_tokens=top_tokens[:10],
            extracted_identifiers=identifiers,
            inferred_type=uncertainty_type,
            context_snippet=recent_context[-100:],
            metadata={
                'position': position,
                'framework': framework,
                'uncertainty_value': uncertainty_value,
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
