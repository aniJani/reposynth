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
    'create_adaptive_generator',
    'create_reposynth_generator',
]

__version__ = '0.1.0'

# Lazy imports
def __getattr__(name):
    if name in ['AdaptiveGenerator', 'GenerationResult', 'GenerationConfig',
                'create_adaptive_generator', 'create_reposynth_generator']:
        from .adaptive_generator import (
            AdaptiveGenerator,
            GenerationResult,
            GenerationConfig,
            create_adaptive_generator,
            create_reposynth_generator,
        )
        globals().update({
            'AdaptiveGenerator': AdaptiveGenerator,
            'GenerationResult': GenerationResult,
            'GenerationConfig': GenerationConfig,
            'create_adaptive_generator': create_adaptive_generator,
            'create_reposynth_generator': create_reposynth_generator,
        })
        return globals()[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
