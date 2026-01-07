"""
Entropy calculation modules for Contrastive Code Entropy (CCE).

This package provides uncertainty measurement for LLM code generation through:
- Entropy calculation (Shannon, normalized, probability differential)
- Token classification (code vs language vs other)
- Contrastive Code Entropy (CCE) computation
- Uncertainty monitoring and spike detection
- Measurement strategies (when to measure entropy)

Week 4-5: Complete uncertainty monitoring system
"""

# Lazy imports to avoid requiring all dependencies at import time
def __getattr__(name):
    # Calculator imports
    if name in ['EntropyCalculator', 'shannon_entropy', 'normalized_entropy',
                'probability_differential', 'top_k_entropy', 'softmax',
                'get_top_k_predictions']:
        from .calculator import (
            EntropyCalculator,
            shannon_entropy,
            normalized_entropy,
            probability_differential,
            top_k_entropy,
            softmax,
            get_top_k_predictions,
        )
        globals().update({
            'EntropyCalculator': EntropyCalculator,
            'shannon_entropy': shannon_entropy,
            'normalized_entropy': normalized_entropy,
            'probability_differential': probability_differential,
            'top_k_entropy': top_k_entropy,
            'softmax': softmax,
            'get_top_k_predictions': get_top_k_predictions,
        })
        return globals()[name]

    # Token classifier imports
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

    # CCE computer imports
    if name in ['CCEComputer', 'compute_cce_from_logits']:
        from .cce_computer import CCEComputer, compute_cce_from_logits
        globals().update({
            'CCEComputer': CCEComputer,
            'compute_cce_from_logits': compute_cce_from_logits,
        })
        return globals()[name]

    # Measurement strategy imports
    if name in ['MeasurementStrategy', 'MeasurementContext', 'EveryTokenStrategy',
                'NthTokenStrategy', 'LineStartStrategy', 'SemanticBoundaryStrategy',
                'CompositeStrategy', 'create_strategy', 'get_recommended_strategy']:
        from .measurement import (
            MeasurementStrategy,
            MeasurementContext,
            EveryTokenStrategy,
            NthTokenStrategy,
            LineStartStrategy,
            SemanticBoundaryStrategy,
            CompositeStrategy,
            create_strategy,
            get_recommended_strategy,
        )
        globals().update({
            'MeasurementStrategy': MeasurementStrategy,
            'MeasurementContext': MeasurementContext,
            'EveryTokenStrategy': EveryTokenStrategy,
            'NthTokenStrategy': NthTokenStrategy,
            'LineStartStrategy': LineStartStrategy,
            'SemanticBoundaryStrategy': SemanticBoundaryStrategy,
            'CompositeStrategy': CompositeStrategy,
            'create_strategy': create_strategy,
            'get_recommended_strategy': get_recommended_strategy,
        })
        return globals()[name]

    # Monitor imports
    if name in ['UncertaintyMonitor', 'UncertaintyResult', 'UncertaintyMethod',
                'create_monitor', 'create_probe_monitor']:
        from .monitor import (
            UncertaintyMonitor,
            UncertaintyResult,
            UncertaintyMethod,
            create_monitor,
            create_probe_monitor,
        )
        globals().update({
            'UncertaintyMonitor': UncertaintyMonitor,
            'UncertaintyResult': UncertaintyResult,
            'UncertaintyMethod': UncertaintyMethod,
            'create_monitor': create_monitor,
            'create_probe_monitor': create_probe_monitor,
        })
        return globals()[name]

    # Spike detector imports
    if name in ['SpikeDetector', 'SpikeInfo', 'SpikeMethod',
                'MonitoredSpikeDetector', 'create_spike_detector',
                'create_monitored_detector']:
        from .spike_detector import (
            SpikeDetector,
            SpikeInfo,
            SpikeMethod,
            MonitoredSpikeDetector,
            create_spike_detector,
            create_monitored_detector,
        )
        globals().update({
            'SpikeDetector': SpikeDetector,
            'SpikeInfo': SpikeInfo,
            'SpikeMethod': SpikeMethod,
            'MonitoredSpikeDetector': MonitoredSpikeDetector,
            'create_spike_detector': create_spike_detector,
            'create_monitored_detector': create_monitored_detector,
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
    'softmax',
    'get_top_k_predictions',

    # Token Classifier
    'KeywordClassifier',
    'EmbeddingClassifier',
    'HybridClassifier',
    'classify_token',
    'build_code_keyword_set',
    'build_language_keyword_set',
    'get_keyword_stats',

    # CCE Computer
    'CCEComputer',
    'compute_cce_from_logits',

    # Measurement Strategies
    'MeasurementStrategy',
    'MeasurementContext',
    'EveryTokenStrategy',
    'NthTokenStrategy',
    'LineStartStrategy',
    'SemanticBoundaryStrategy',
    'CompositeStrategy',
    'create_strategy',
    'get_recommended_strategy',

    # Uncertainty Monitor
    'UncertaintyMonitor',
    'UncertaintyResult',
    'UncertaintyMethod',
    'create_monitor',
    'create_probe_monitor',

    # Spike Detector
    'SpikeDetector',
    'SpikeInfo',
    'SpikeMethod',
    'MonitoredSpikeDetector',
    'create_spike_detector',
    'create_monitored_detector',
]

__version__ = '0.2.0'  # Week 4-5 additions
