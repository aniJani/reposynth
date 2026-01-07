"""
Spike Detector Module for Uncertainty Analysis

This module provides methods for detecting entropy spikes during generation.
Rather than using a fixed threshold, spike detection compares current entropy
to a dynamic baseline, which can be more robust across different contexts.

Detection Methods:
- threshold: Simple fixed threshold (baseline approach)
- relative: Spike if value > baseline * factor
- statistical: Spike if value > mean + k * std
- percentile: Spike if value > percentile threshold of history
- adaptive: Combines multiple methods with voting

Week 5: Spike Detection & Integration
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal, Tuple
from collections import deque
from enum import Enum
import numpy as np


class SpikeMethod(Enum):
    """Available spike detection methods."""
    THRESHOLD = "threshold"
    RELATIVE = "relative"
    STATISTICAL = "statistical"
    PERCENTILE = "percentile"
    ADAPTIVE = "adaptive"


@dataclass
class SpikeInfo:
    """Information about a detected spike."""
    position: int             # Position where spike detected
    value: float             # Entropy value at spike
    baseline: float          # Baseline value used for comparison
    deviation: float         # How much value exceeds baseline
    severity: str            # 'low', 'medium', 'high'
    method: str              # Detection method that flagged it
    confidence: float        # Confidence in spike detection [0, 1]

    def __repr__(self) -> str:
        return (
            f"SpikeInfo(pos={self.position}, value={self.value:.3f}, "
            f"baseline={self.baseline:.3f}, severity={self.severity})"
        )


@dataclass
class DetectorState:
    """Internal state of spike detector."""
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    baseline: Optional[float] = None
    baseline_std: Optional[float] = None
    spike_count: int = 0
    last_spike_position: Optional[int] = None
    warmup_complete: bool = False


class SpikeDetector:
    """
    Detects entropy spikes indicating code uncertainty.

    The detector maintains a rolling history of entropy values and
    compares new values against a dynamic baseline to detect spikes.

    Example:
        >>> detector = SpikeDetector(method='statistical', k_sigma=2.0)
        >>> for i, entropy in enumerate(entropy_values):
        ...     detector.update(entropy)
        ...     if detector.is_spike(entropy):
        ...         spike = detector.get_spike_info(i, entropy)
        ...         print(f"Spike detected at position {i}")
    """

    def __init__(
        self,
        method: str = "threshold",
        threshold: float = 0.3,
        relative_factor: float = 1.5,
        k_sigma: float = 2.0,
        percentile: float = 90.0,
        window_size: int = 100,
        warmup_samples: int = 10,
        min_spike_gap: int = 5,
    ):
        """
        Initialize spike detector.

        Args:
            method: Detection method ('threshold', 'relative', 'statistical',
                   'percentile', 'adaptive')
            threshold: Fixed threshold for 'threshold' method
            relative_factor: Multiplier for 'relative' method (spike if > baseline * factor)
            k_sigma: Number of std deviations for 'statistical' method
            percentile: Percentile threshold for 'percentile' method
            window_size: Size of rolling history window
            warmup_samples: Number of samples before baseline is established
            min_spike_gap: Minimum positions between spikes (debouncing)
        """
        self.method = SpikeMethod(method)
        self.threshold = threshold
        self.relative_factor = relative_factor
        self.k_sigma = k_sigma
        self.percentile = percentile
        self.window_size = window_size
        self.warmup_samples = warmup_samples
        self.min_spike_gap = min_spike_gap

        # Initialize state
        self.state = DetectorState(
            history=deque(maxlen=window_size)
        )

    def reset(self):
        """Reset detector state."""
        self.state = DetectorState(
            history=deque(maxlen=self.window_size)
        )

    def update(self, value: float, position: Optional[int] = None):
        """
        Update detector with new entropy value.

        Args:
            value: Entropy value to add to history
            position: Optional position index
        """
        self.state.history.append(value)

        # Update baseline after warmup
        if len(self.state.history) >= self.warmup_samples:
            self.state.warmup_complete = True
            self._update_baseline()

    def _update_baseline(self):
        """Update baseline statistics from history."""
        if not self.state.history:
            return

        values = np.array(self.state.history)
        self.state.baseline = float(np.mean(values))
        self.state.baseline_std = float(np.std(values))

    def is_spike(self, value: float, position: Optional[int] = None) -> bool:
        """
        Check if value is a spike.

        Args:
            value: Entropy value to check
            position: Position index (for debouncing)

        Returns:
            True if value is detected as a spike
        """
        # Debouncing: skip if too close to last spike
        if position is not None and self.state.last_spike_position is not None:
            if position - self.state.last_spike_position < self.min_spike_gap:
                return False

        # Use appropriate detection method
        if self.method == SpikeMethod.THRESHOLD:
            is_spike = self._check_threshold(value)
        elif self.method == SpikeMethod.RELATIVE:
            is_spike = self._check_relative(value)
        elif self.method == SpikeMethod.STATISTICAL:
            is_spike = self._check_statistical(value)
        elif self.method == SpikeMethod.PERCENTILE:
            is_spike = self._check_percentile(value)
        elif self.method == SpikeMethod.ADAPTIVE:
            is_spike = self._check_adaptive(value)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Update last spike position
        if is_spike and position is not None:
            self.state.last_spike_position = position
            self.state.spike_count += 1

        return is_spike

    def _check_threshold(self, value: float) -> bool:
        """Simple fixed threshold check."""
        return value > self.threshold

    def _check_relative(self, value: float) -> bool:
        """Check if value exceeds baseline by factor."""
        if not self.state.warmup_complete or self.state.baseline is None:
            return self._check_threshold(value)

        return value > self.state.baseline * self.relative_factor

    def _check_statistical(self, value: float) -> bool:
        """Check if value is k standard deviations above mean."""
        if not self.state.warmup_complete:
            return self._check_threshold(value)

        if self.state.baseline is None or self.state.baseline_std is None:
            return self._check_threshold(value)

        # Handle zero std (constant values)
        if self.state.baseline_std < 1e-6:
            return value > self.state.baseline

        threshold = self.state.baseline + self.k_sigma * self.state.baseline_std
        return value > threshold

    def _check_percentile(self, value: float) -> bool:
        """Check if value exceeds percentile of history."""
        if not self.state.warmup_complete or len(self.state.history) < self.warmup_samples:
            return self._check_threshold(value)

        values = np.array(self.state.history)
        percentile_threshold = np.percentile(values, self.percentile)
        return value > percentile_threshold

    def _check_adaptive(self, value: float) -> bool:
        """
        Adaptive method: combine multiple detection methods with voting.

        A spike is detected if at least 2 out of 3 methods agree.
        """
        votes = 0

        # Statistical vote
        if self._check_statistical(value):
            votes += 1

        # Relative vote
        if self._check_relative(value):
            votes += 1

        # Percentile vote (if enough history)
        if len(self.state.history) >= self.warmup_samples:
            if self._check_percentile(value):
                votes += 1
        else:
            # Fall back to threshold during warmup
            if self._check_threshold(value):
                votes += 1

        return votes >= 2

    def get_spike_info(self, position: int, value: float) -> SpikeInfo:
        """
        Get detailed information about a spike.

        Args:
            position: Position where spike detected
            value: Entropy value at spike

        Returns:
            SpikeInfo with spike details
        """
        baseline = self.state.baseline or self.threshold
        deviation = value - baseline

        # Calculate severity
        if self.state.baseline_std and self.state.baseline_std > 1e-6:
            z_score = deviation / self.state.baseline_std
            if z_score > 3:
                severity = 'high'
            elif z_score > 2:
                severity = 'medium'
            else:
                severity = 'low'
        else:
            # Fallback severity calculation
            relative = value / baseline if baseline > 0 else float('inf')
            if relative > 2.0:
                severity = 'high'
            elif relative > 1.5:
                severity = 'medium'
            else:
                severity = 'low'

        # Calculate confidence
        confidence = self._calculate_confidence(value)

        return SpikeInfo(
            position=position,
            value=value,
            baseline=baseline,
            deviation=deviation,
            severity=severity,
            method=self.method.value,
            confidence=confidence,
        )

    def _calculate_confidence(self, value: float) -> float:
        """
        Calculate confidence in spike detection.

        Higher confidence when multiple methods agree and
        deviation is large.
        """
        if not self.state.warmup_complete:
            return 0.5  # Low confidence during warmup

        # Count method agreement
        methods_triggered = 0
        if self._check_threshold(value):
            methods_triggered += 1
        if self._check_relative(value):
            methods_triggered += 1
        if self._check_statistical(value):
            methods_triggered += 1
        if len(self.state.history) >= self.warmup_samples:
            if self._check_percentile(value):
                methods_triggered += 1

        # Base confidence on method agreement
        base_confidence = methods_triggered / 4.0

        # Boost confidence for large deviations
        if self.state.baseline_std and self.state.baseline_std > 1e-6:
            z_score = (value - self.state.baseline) / self.state.baseline_std
            deviation_boost = min(0.3, z_score * 0.1)  # Max 0.3 boost
        else:
            deviation_boost = 0.0

        return min(1.0, base_confidence + deviation_boost)

    def get_baseline(self) -> Optional[float]:
        """Get current baseline value."""
        return self.state.baseline

    def get_statistics(self) -> Dict:
        """
        Get detector statistics.

        Returns:
            Dictionary with detector statistics
        """
        values = np.array(self.state.history) if self.state.history else np.array([])

        return {
            'method': self.method.value,
            'warmup_complete': self.state.warmup_complete,
            'history_size': len(self.state.history),
            'baseline': self.state.baseline,
            'baseline_std': self.state.baseline_std,
            'spike_count': self.state.spike_count,
            'history_mean': float(np.mean(values)) if len(values) > 0 else None,
            'history_max': float(np.max(values)) if len(values) > 0 else None,
            'history_min': float(np.min(values)) if len(values) > 0 else None,
        }


# ============================================================================
# COMBINED MONITOR + DETECTOR
# ============================================================================

class MonitoredSpikeDetector:
    """
    Combines UncertaintyMonitor with SpikeDetector for complete pipeline.

    This class wraps both components and provides a unified interface
    for uncertainty-triggered retrieval decisions.

    Example:
        >>> detector = MonitoredSpikeDetector(
        ...     method='cce',
        ...     spike_method='statistical',
        ...     tokenizer=tokenizer
        ... )
        >>> for logits, token in generation:
        ...     result = detector.step(logits, token)
        ...     if result.should_retrieve:
        ...         context = retrieve_context()
    """

    def __init__(
        self,
        method: str = "cce",
        spike_method: str = "statistical",
        tokenizer = None,
        measurement_strategy: str = "semantic_boundary",
        threshold: float = 0.3,
        k_sigma: float = 2.0,
        max_retrievals: int = 3,
        **kwargs
    ):
        """
        Initialize combined monitor and detector.

        Args:
            method: Uncertainty measurement method
            spike_method: Spike detection method
            tokenizer: HuggingFace tokenizer
            measurement_strategy: When to measure
            threshold: Base threshold
            k_sigma: Statistical detection parameter
            max_retrievals: Maximum retrievals
            **kwargs: Additional arguments passed to monitor
        """
        from .monitor import UncertaintyMonitor

        self.monitor = UncertaintyMonitor(
            method=method,
            threshold=threshold,
            tokenizer=tokenizer,
            measurement_strategy=measurement_strategy,
            max_retrievals=max_retrievals,
            **kwargs
        )

        self.detector = SpikeDetector(
            method=spike_method,
            threshold=threshold,
            k_sigma=k_sigma,
        )

        self.use_spike_detection = spike_method != "threshold"

    def reset(self):
        """Reset both monitor and detector."""
        self.monitor.reset()
        self.detector.reset()

    def step(
        self,
        logits: np.ndarray,
        token: str,
        hidden_states: Optional[np.ndarray] = None,
    ):
        """
        Process one generation step.

        Args:
            logits: Model output logits
            token: Generated token
            hidden_states: Optional hidden states for probe method

        Returns:
            UncertaintyResult with spike-adjusted retrieval decision
        """
        result = self.monitor.step(logits, token, hidden_states)

        if result is None:
            return None

        # Update spike detector with measured value
        self.detector.update(result.value, result.position)

        # Override retrieval decision based on spike detection
        if self.use_spike_detection:
            is_spike = self.detector.is_spike(result.value, result.position)
            result.should_retrieve = is_spike
            result.details['is_spike'] = is_spike

            if is_spike:
                spike_info = self.detector.get_spike_info(result.position, result.value)
                result.details['spike_info'] = {
                    'severity': spike_info.severity,
                    'baseline': spike_info.baseline,
                    'deviation': spike_info.deviation,
                    'confidence': spike_info.confidence,
                }

        return result

    def get_summary(self) -> Dict:
        """Get combined summary from both components."""
        monitor_summary = self.monitor.get_summary()
        detector_stats = self.detector.get_statistics()

        return {
            **monitor_summary,
            'detector': detector_stats,
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_spike_detector(
    method: str = "statistical",
    **kwargs
) -> SpikeDetector:
    """
    Factory function to create spike detectors.

    Args:
        method: Detection method
        **kwargs: Method-specific parameters

    Returns:
        Configured SpikeDetector
    """
    return SpikeDetector(method=method, **kwargs)


def create_monitored_detector(
    uncertainty_method: str = "cce",
    spike_method: str = "statistical",
    tokenizer = None,
    **kwargs
) -> MonitoredSpikeDetector:
    """
    Factory function to create combined monitor + detector.

    Args:
        uncertainty_method: Method for measuring uncertainty
        spike_method: Method for detecting spikes
        tokenizer: HuggingFace tokenizer
        **kwargs: Additional parameters

    Returns:
        Configured MonitoredSpikeDetector
    """
    return MonitoredSpikeDetector(
        method=uncertainty_method,
        spike_method=spike_method,
        tokenizer=tokenizer,
        **kwargs
    )
