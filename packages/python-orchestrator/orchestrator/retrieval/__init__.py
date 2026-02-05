"""
Retrieval module for Adaptive Context Retrieval.

This package provides:
- TopicInferrer: Infer WHAT to retrieve from uncertainty signals
- LearnedTopicInferrer: Learned cross-attention replacement for TopicInferrer
- AdaptiveContextRetriever: Retrieve context when uncertainty is detected
- ContextManager: Manage context window budget

Phase 3: Weeks 6-7 of CCE Research Implementation
Week 10: Learned Query Pooling
"""

__all__ = [
    # Topic Inference
    'TopicInferrer',
    'TopicResult',

    # Learned Query Pooling (Week 10)
    'LearnedTopicInferrer',
    'LearnedQueryModule',
    'LearnedQueryPooler',
    'QueryPoolerLoss',

    # Adaptive Retrieval
    'AdaptiveContextRetriever',
    'RetrievalResult',
    'EmbeddingRetriever',
    'SimpleRetriever',

    # RepoSynth Integration
    'RepoSynthRetriever',

    # Context Management
    'ContextManager',
    'ContextSegment',
]

__version__ = '0.2.0'

# Lazy imports
def __getattr__(name):
    if name in ['TopicInferrer', 'TopicResult']:
        from .topic_inference import TopicInferrer, TopicResult
        globals().update({
            'TopicInferrer': TopicInferrer,
            'TopicResult': TopicResult,
        })
        return globals()[name]

    if name == 'LearnedTopicInferrer':
        from .learned_inferrer import LearnedTopicInferrer
        globals()['LearnedTopicInferrer'] = LearnedTopicInferrer
        return LearnedTopicInferrer

    if name in ['LearnedQueryModule', 'LearnedQueryPooler', 'QueryPoolerLoss']:
        from .learned_query import LearnedQueryModule, LearnedQueryPooler, QueryPoolerLoss
        globals().update({
            'LearnedQueryModule': LearnedQueryModule,
            'LearnedQueryPooler': LearnedQueryPooler,
            'QueryPoolerLoss': QueryPoolerLoss,
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

    if name == 'RepoSynthRetriever':
        from .reposynth_retriever import RepoSynthRetriever
        globals()['RepoSynthRetriever'] = RepoSynthRetriever
        return RepoSynthRetriever

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
