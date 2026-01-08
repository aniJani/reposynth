"""
Uncertainty Monitor Module

This module provides real-time uncertainty monitoring during LLM generation.
It wraps entropy calculation, measurement strategies, and provides an interface
for detecting when context retrieval should be triggered.

Supports multiple uncertainty methods:
- raw_entropy: Standard Shannon entropy
- normalized_entropy: Entropy normalized to [0, 1]
- prob_differential: 1 - max(P) (UnCert-CoT style)
- cce: Contrastive Code Entropy (our method)
- probe: Learned probe on hidden states (Week 3 best)

Week 4: Uncertainty Monitoring System
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Any, Callable
from enum import Enum
import numpy as np

from .measurement import (
    MeasurementStrategy,
    MeasurementContext,
    SemanticBoundaryStrategy,
    create_strategy,
)
from .calculator import EntropyCalculator


class UncertaintyMethod(Enum):
    """Available uncertainty measurement methods."""
    RAW_ENTROPY = "raw_entropy"
    NORMALIZED_ENTROPY = "normalized_entropy"
    PROB_DIFFERENTIAL = "prob_differential"
    CCE = "cce"
    PROBE = "probe"  # Week 3 learned probe


@dataclass
class UncertaintyResult:
    """Result of a single uncertainty measurement."""
    position: int                 # Token position where measured
    value: float                  # Uncertainty value
    method: str                   # Method used
    should_retrieve: bool         # Whether to trigger retrieval
    token: str                    # Token at this position
    generated_context: str        # Text generated up to this point

    # Optional detailed metrics
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        retrieve_str = "RETRIEVE" if self.should_retrieve else "continue"
        return (
            f"UncertaintyResult(pos={self.position}, "
            f"value={self.value:.4f}, {retrieve_str})"
        )


@dataclass
class MonitorState:
    """Internal state of the uncertainty monitor."""
    position: int = 0
    generated_tokens: List[str] = field(default_factory=list)
    generated_text: str = ""
    measurements: List[UncertaintyResult] = field(default_factory=list)
    retrieval_events: List[Dict] = field(default_factory=list)
    total_retrievals: int = 0
    last_retrieval_position: int = -100  # For cooldown tracking


class UncertaintyMonitor:
    """
    Real-time uncertainty monitor for LLM generation.

    Monitors token generation and detects when uncertainty indicates
    that additional context should be retrieved.

    Example:
        >>> monitor = UncertaintyMonitor(
        ...     method="cce",
        ...     threshold=0.3,
        ...     tokenizer=tokenizer
        ... )
        >>> for i, logits in enumerate(generation_logits):
        ...     result = monitor.step(logits, generated_token)
        ...     if result and result.should_retrieve:
        ...         # Trigger context retrieval
        ...         context = retrieve_context(result)
    """

    def __init__(
        self,
        method: str = "cce",
        threshold: float = 0.3,
        tokenizer = None,
        measurement_strategy: str = "semantic_boundary",
        max_retrievals: int = 3,
        cooldown_tokens: int = 0,
        cce_use_hybrid: bool = True,
        cce_embedding_model: str = 'all-MiniLM-L6-v2',
        probe_model = None,
        probe_scaler = None,
        probe_layers: List[int] = None,
    ):
        """
        Initialize uncertainty monitor.

        Args:
            method: Uncertainty method ('raw_entropy', 'normalized_entropy',
                   'prob_differential', 'cce', 'probe')
            threshold: Threshold for triggering retrieval
            tokenizer: HuggingFace tokenizer (required for CCE)
            measurement_strategy: When to measure ('every_token', 'semantic_boundary',
                                 'line_start', 'every_n')
            max_retrievals: Maximum number of retrievals per generation
            cooldown_tokens: Minimum tokens between retrievals (0 = no cooldown)
            cce_use_hybrid: Use hybrid classifier for CCE
            cce_embedding_model: Embedding model for CCE
            probe_model: Trained probe model (for 'probe' method)
            probe_scaler: Feature scaler for probe
            probe_layers: Hidden state layers for probe
        """
        self.method = UncertaintyMethod(method)
        self.threshold = threshold
        self.tokenizer = tokenizer
        self.max_retrievals = max_retrievals
        self.cooldown_tokens = cooldown_tokens

        # Set up measurement strategy
        if isinstance(measurement_strategy, str):
            self.measurement_strategy = create_strategy(measurement_strategy)
        else:
            self.measurement_strategy = measurement_strategy

        # Set up entropy calculator
        self.entropy_calc = EntropyCalculator()

        # Set up CCE computer if needed
        self.cce_computer = None
        if self.method == UncertaintyMethod.CCE:
            if tokenizer is None:
                raise ValueError("Tokenizer required for CCE method")
            from .cce_computer import CCEComputer
            self.cce_computer = CCEComputer(
                tokenizer=tokenizer,
                use_hybrid=cce_use_hybrid,
                embedding_model=cce_embedding_model,
            )

        # Set up probe if needed
        self.probe_model = probe_model
        self.probe_scaler = probe_scaler
        self.probe_layers = probe_layers or [16, 24, 31]

        # Initialize state
        self.state = MonitorState()

        # History for analysis
        self.history: List[UncertaintyResult] = []

    def reset(self):
        """Reset monitor state for new generation."""
        self.state = MonitorState()

    def step(
        self,
        logits: np.ndarray,
        token: str,
        hidden_states: Optional[np.ndarray] = None,
    ) -> Optional[UncertaintyResult]:
        """
        Process one generation step.

        Args:
            logits: Logits from model output [vocab_size]
            token: The token generated at this position
            hidden_states: Hidden states (required for probe method)
                          Shape: [n_layers, hidden_dim] or concatenated [total_dim]

        Returns:
            UncertaintyResult if measurement was taken, None otherwise
        """
        # Update state
        self.state.generated_tokens.append(token)
        self.state.generated_text += token

        # Create measurement context
        context = MeasurementContext(
            token=token,
            position=self.state.position,
            generated_so_far=self.state.generated_text,
            previous_tokens=self.state.generated_tokens[:-1],
        )

        # Check if we should measure at this position
        if not self.measurement_strategy.should_measure(context):
            self.state.position += 1
            return None

        # Compute uncertainty
        result = self._compute_uncertainty(
            logits=logits,
            token=token,
            hidden_states=hidden_states,
        )

        # Store result
        self.state.measurements.append(result)
        self.history.append(result)

        # Check if retrieval should be triggered
        if result.should_retrieve:
            # Check cooldown
            tokens_since_last = result.position - self.state.last_retrieval_position
            in_cooldown = tokens_since_last < self.cooldown_tokens

            if in_cooldown:
                result.should_retrieve = False
                result.details['in_cooldown'] = True
                result.details['tokens_until_cooldown_end'] = self.cooldown_tokens - tokens_since_last
            elif self.state.total_retrievals >= self.max_retrievals:
                # Max retrievals reached, override decision
                result.should_retrieve = False
                result.details['max_retrievals_reached'] = True
            else:
                # Proceed with retrieval
                self.state.retrieval_events.append({
                    'position': result.position,
                    'value': result.value,
                    'token': result.token,
                    'context': result.generated_context[-200:],  # Last 200 chars
                })
                self.state.total_retrievals += 1
                self.state.last_retrieval_position = result.position

        self.state.position += 1
        return result

    def _compute_uncertainty(
        self,
        logits: np.ndarray,
        token: str,
        hidden_states: Optional[np.ndarray] = None,
    ) -> UncertaintyResult:
        """Compute uncertainty using the configured method."""

        if self.method == UncertaintyMethod.RAW_ENTROPY:
            value = self.entropy_calc.shannon_entropy(logits)
            should_retrieve = value > self.threshold
            details = {'entropy_bits': value}

        elif self.method == UncertaintyMethod.NORMALIZED_ENTROPY:
            value = self.entropy_calc.normalized_entropy(logits)
            should_retrieve = value > self.threshold
            details = {'normalized_entropy': value}

        elif self.method == UncertaintyMethod.PROB_DIFFERENTIAL:
            value = self.entropy_calc.probability_differential(logits)
            should_retrieve = value > self.threshold
            details = {'prob_differential': value}

        elif self.method == UncertaintyMethod.CCE:
            if self.cce_computer is None:
                raise RuntimeError("CCE computer not initialized")
            cce_result = self.cce_computer.compute_cce(logits)
            value = cce_result['contrastive_entropy']
            should_retrieve = value > self.threshold
            details = cce_result

        elif self.method == UncertaintyMethod.PROBE:
            if self.probe_model is None:
                raise RuntimeError("Probe model not set")
            if hidden_states is None:
                raise ValueError("Hidden states required for probe method")

            # Use probe to classify
            value, should_retrieve = self._apply_probe(hidden_states)
            details = {'probe_score': value}

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return UncertaintyResult(
            position=self.state.position,
            value=value,
            method=self.method.value,
            should_retrieve=should_retrieve,
            token=token,
            generated_context=self.state.generated_text,
            details=details,
        )

    def _apply_probe(
        self,
        hidden_states: np.ndarray
    ) -> tuple:
        """
        Apply the trained probe to hidden states.

        Returns:
            (score, should_retrieve) tuple
        """
        # Reshape if needed
        if hidden_states.ndim == 1:
            features = hidden_states.reshape(1, -1)
        else:
            features = hidden_states.reshape(1, -1)

        # Scale features
        if self.probe_scaler is not None:
            features = self.probe_scaler.transform(features)

        # Get prediction
        prediction = self.probe_model.predict(features)[0]
        proba = self.probe_model.predict_proba(features)[0]

        # Score is probability of 'code' class
        code_prob = proba[1] if len(proba) > 1 else proba[0]

        # Should retrieve if classified as code (using threshold from Week 3)
        should_retrieve = code_prob > self.threshold

        return float(code_prob), should_retrieve

    def get_entropy_trace(self) -> List[Dict]:
        """
        Get the entropy trace for visualization.

        Returns:
            List of dicts with position, value, and metadata
        """
        return [
            {
                'position': r.position,
                'value': r.value,
                'token': r.token,
                'should_retrieve': r.should_retrieve,
            }
            for r in self.state.measurements
        ]

    def get_summary(self) -> Dict:
        """
        Get summary statistics of the monitoring session.

        Returns:
            Dictionary with monitoring statistics
        """
        values = [r.value for r in self.state.measurements]

        if not values:
            return {
                'total_tokens': self.state.position,
                'measurements_taken': 0,
                'measurement_rate': 0.0,
                'retrievals_triggered': self.state.total_retrievals,
            }

        return {
            'total_tokens': self.state.position,
            'measurements_taken': len(self.state.measurements),
            'measurement_rate': len(self.state.measurements) / max(1, self.state.position),
            'retrievals_triggered': self.state.total_retrievals,
            'mean_uncertainty': float(np.mean(values)),
            'max_uncertainty': float(np.max(values)),
            'min_uncertainty': float(np.min(values)),
            'std_uncertainty': float(np.std(values)),
            'above_threshold_count': sum(1 for v in values if v > self.threshold),
        }

    def visualize_trace(self, figsize: tuple = (12, 4)):
        """
        Visualize the entropy trace.

        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for visualization")

        trace = self.get_entropy_trace()
        if not trace:
            print("No measurements to visualize")
            return

        positions = [t['position'] for t in trace]
        values = [t['value'] for t in trace]
        retrievals = [t['position'] for t in trace if t['should_retrieve']]

        fig, ax = plt.subplots(figsize=figsize)

        # Plot entropy trace
        ax.plot(positions, values, 'b-', label=f'{self.method.value}', alpha=0.7)
        ax.scatter(positions, values, c='blue', s=20, alpha=0.5)

        # Mark threshold
        ax.axhline(y=self.threshold, color='r', linestyle='--',
                   label=f'Threshold ({self.threshold})')

        # Mark retrieval points
        if retrievals:
            retrieval_values = [values[positions.index(p)] for p in retrievals]
            ax.scatter(retrievals, retrieval_values, c='red', s=100,
                      marker='v', label='Retrieval triggered', zorder=5)

        ax.set_xlabel('Token Position')
        ax.set_ylabel('Uncertainty Value')
        ax.set_title(f'Uncertainty Trace ({self.method.value})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_monitor(
    method: str = "cce",
    threshold: float = 0.3,
    tokenizer = None,
    strategy: str = "semantic_boundary",
    **kwargs
) -> UncertaintyMonitor:
    """
    Factory function to create uncertainty monitors.

    Args:
        method: Uncertainty method
        threshold: Retrieval threshold
        tokenizer: HuggingFace tokenizer
        strategy: Measurement strategy
        **kwargs: Additional arguments

    Returns:
        Configured UncertaintyMonitor
    """
    return UncertaintyMonitor(
        method=method,
        threshold=threshold,
        tokenizer=tokenizer,
        measurement_strategy=strategy,
        **kwargs
    )


def create_probe_monitor(
    probe_model,
    probe_scaler,
    probe_layers: List[int] = None,
    threshold: float = 0.06,
    strategy: str = "semantic_boundary",
) -> UncertaintyMonitor:
    """
    Create a monitor using the Week 3 learned probe.

    Args:
        probe_model: Trained LogisticRegression or similar
        probe_scaler: StandardScaler for features
        probe_layers: Hidden state layers to use
        threshold: Classification threshold (default from Week 3: 0.06)
        strategy: Measurement strategy

    Returns:
        UncertaintyMonitor configured for probe-based detection
    """
    return UncertaintyMonitor(
        method="probe",
        threshold=threshold,
        measurement_strategy=strategy,
        probe_model=probe_model,
        probe_scaler=probe_scaler,
        probe_layers=probe_layers or [16, 24, 31],
    )
