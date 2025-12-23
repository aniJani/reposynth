"""
Entropy calculation modules for Contrastive Code Entropy (CCE).

This package provides uncertainty measurement for LLM code generation through:
- Entropy calculation (Shannon, normalized, probability differential)
- Token classification (code vs language vs other)
- Contrastive Code Entropy (CCE) computation
- Uncertainty monitoring and spike detection
"""

# Lazy imports to avoid requiring all dependencies at import time
def __getattr__(name):
    if name in ['EntropyCalculator', 'shannon_entropy', 'normalized_entropy',
                'probability_differential', 'top_k_entropy']:
        from .calculator import (
            EntropyCalculator,
            shannon_entropy,
            normalized_entropy,
            probability_differential,
            top_k_entropy
        )
        globals().update({
            'EntropyCalculator': EntropyCalculator,
            'shannon_entropy': shannon_entropy,
            'normalized_entropy': normalized_entropy,
            'probability_differential': probability_differential,
            'top_k_entropy': top_k_entropy,
        })
        return globals()[name]

    if name in ['KeywordClassifier', 'classify_token', 'build_code_keyword_set',
                'build_language_keyword_set', 'get_keyword_stats',
                'EmbeddingClassifier', 'HybridClassifier']:
        from .token_classifier import (
            KeywordClassifier,
            classify_token,
            build_code_keyword_set,
            build_language_keyword_set,
            get_keyword_stats,
            EmbeddingClassifier,
            HybridClassifier,
        )
        globals().update({
            'KeywordClassifier': KeywordClassifier,
            'classify_token': classify_token,
            'build_code_keyword_set': build_code_keyword_set,
            'build_language_keyword_set': build_language_keyword_set,
            'get_keyword_stats': get_keyword_stats,
            'EmbeddingClassifier': EmbeddingClassifier,
            'HybridClassifier': HybridClassifier,
        })
        return globals()[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Calculator
    'EntropyCalculator',
    'shannon_entropy',
    'normalized_entropy',
    'probability_differential',
    'top_k_entropy',
    # Token Classifier
    'KeywordClassifier',
    'EmbeddingClassifier',
    'HybridClassifier',
    'classify_token',
    'build_code_keyword_set',
    'build_language_keyword_set',
    'get_keyword_stats',
]

__version__ = '0.1.0'
