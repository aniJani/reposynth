"""
Generation module for Adaptive Code Generation.

This package provides:
- AdaptiveGenerator: Main generation loop with uncertainty-triggered retrieval
- GenerationResult: Result container with traces and metadata

Phase 3, Week 7: Generation Loop & Context Management
"""

__all__ = [
    'AdaptiveGenerator',
    'GenerationResult',
    'GenerationConfig',
]

__version__ = '0.1.0'

# Lazy imports
def __getattr__(name):
    if name in ['AdaptiveGenerator', 'GenerationResult', 'GenerationConfig']:
        from .adaptive_generator import AdaptiveGenerator, GenerationResult, GenerationConfig
        globals().update({
            'AdaptiveGenerator': AdaptiveGenerator,
            'GenerationResult': GenerationResult,
            'GenerationConfig': GenerationConfig,
        })
        return globals()[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
