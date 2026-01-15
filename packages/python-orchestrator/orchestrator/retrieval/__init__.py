"""
Retrieval module for Adaptive Context Retrieval.

This package provides:
- TopicInferrer: Infer WHAT to retrieve from uncertainty signals
- AdaptiveContextRetriever: Retrieve context when uncertainty is detected
- ContextManager: Manage context window budget

Phase 3: Weeks 6-7 of CCE Research Implementation
"""

__all__ = [
    # Topic Inference
    'TopicInferrer',
    'TopicResult',

    # Adaptive Retrieval
    'AdaptiveContextRetriever',
    'RetrievalResult',
    'EmbeddingRetriever',
    'SimpleRetriever',

    # Context Management
    'ContextManager',
    'ContextSegment',
]

__version__ = '0.1.0'

# Lazy imports
def __getattr__(name):
    if name in ['TopicInferrer', 'TopicResult']:
        from .topic_inference import TopicInferrer, TopicResult
        globals().update({
            'TopicInferrer': TopicInferrer,
            'TopicResult': TopicResult,
        })
        return globals()[name]

    if name in ['AdaptiveContextRetriever', 'RetrievalResult', 'EmbeddingRetriever', 'SimpleRetriever']:
        from .adaptive import AdaptiveContextRetriever, RetrievalResult, EmbeddingRetriever, SimpleRetriever
        globals().update({
            'AdaptiveContextRetriever': AdaptiveContextRetriever,
            'RetrievalResult': RetrievalResult,
            'EmbeddingRetriever': EmbeddingRetriever,
            'SimpleRetriever': SimpleRetriever,
        })
        return globals()[name]

    if name in ['ContextManager', 'ContextSegment']:
        from .context_manager import ContextManager, ContextSegment
        globals().update({
            'ContextManager': ContextManager,
            'ContextSegment': ContextSegment,
        })
        return globals()[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
