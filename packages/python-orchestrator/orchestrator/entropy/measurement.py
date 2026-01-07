"""
Measurement Strategy Module for Uncertainty Monitoring

This module defines strategies for deciding WHEN to measure entropy during generation.
Different strategies offer tradeoffs between overhead and detection accuracy.

Strategies:
- EveryTokenStrategy: Measure at every token (highest overhead, most complete)
- LineStartStrategy: Measure at line boundaries (UnCert-CoT style)
- SemanticBoundaryStrategy: Measure at code semantic boundaries (our approach)
- NthTokenStrategy: Measure every N tokens (simple, configurable)

Week 4: Uncertainty Monitoring System
"""

from abc import ABC, abstractmethod
from typing import List, Set, Optional
from dataclasses import dataclass
import re


@dataclass
class MeasurementContext:
    """Context information for measurement decision."""
    token: str                    # Current token being generated
    position: int                 # Position in generation (0-indexed)
    generated_so_far: str         # Full text generated so far
    previous_tokens: List[str]    # List of previous tokens (for pattern matching)


class MeasurementStrategy(ABC):
    """
    Abstract base class for measurement strategies.

    Strategies determine at which positions during generation we should
    measure entropy/uncertainty. This is a key design decision that affects
    both overhead and detection accuracy.
    """

    @abstractmethod
    def should_measure(self, context: MeasurementContext) -> bool:
        """
        Determine if we should measure entropy at this position.

        Args:
            context: MeasurementContext with current generation state

        Returns:
            True if entropy should be measured, False otherwise
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name for logging/display."""
        pass

    def get_expected_overhead(self) -> str:
        """Return expected measurement overhead description."""
        return "Unknown"


class EveryTokenStrategy(MeasurementStrategy):
    """
    Measure entropy at every token position.

    Pros:
    - Complete coverage, catches all uncertainty spikes
    - No risk of missing critical positions

    Cons:
    - Highest computational overhead
    - May be unnecessary for many positions

    Use when: Research/debugging, or when compute is not a concern
    """

    def should_measure(self, context: MeasurementContext) -> bool:
        return True

    @property
    def name(self) -> str:
        return "every_token"

    def get_expected_overhead(self) -> str:
        return "100% of tokens measured"


class NthTokenStrategy(MeasurementStrategy):
    """
    Measure entropy every N tokens.

    Simple periodic sampling strategy.

    Args:
        n: Measure every N tokens (default: 5)
        include_first: Always measure first token (default: True)
    """

    def __init__(self, n: int = 5, include_first: bool = True):
        self.n = n
        self.include_first = include_first

    def should_measure(self, context: MeasurementContext) -> bool:
        if self.include_first and context.position == 0:
            return True
        return context.position % self.n == 0

    @property
    def name(self) -> str:
        return f"every_{self.n}_tokens"

    def get_expected_overhead(self) -> str:
        return f"~{100/self.n:.0f}% of tokens measured"


class LineStartStrategy(MeasurementStrategy):
    """
    Measure entropy at line boundaries (after newlines).

    Based on UnCert-CoT paper (arXiv:2503.15341) which found that
    line boundaries are natural points to check for uncertainty.

    Pros:
    - Much lower overhead than every-token
    - Natural boundaries in code
    - Works well for line-by-line generation

    Cons:
    - May miss uncertainty within long lines
    - Depends on code style (long lines = less coverage)

    Args:
        include_first: Always measure first position (default: True)
        newline_chars: Characters to consider as newlines
    """

    def __init__(
        self,
        include_first: bool = True,
        newline_chars: Set[str] = None
    ):
        self.include_first = include_first
        self.newline_chars = newline_chars or {'\n', '\r', '\r\n'}

    def should_measure(self, context: MeasurementContext) -> bool:
        # Always measure first position
        if self.include_first and context.position == 0:
            return True

        # Check if current token is a newline
        if context.token.strip() == '' and any(
            nl in context.token for nl in self.newline_chars
        ):
            return True

        # Check if previous token was a newline (measure at line start)
        if context.previous_tokens:
            prev = context.previous_tokens[-1]
            if any(nl in prev for nl in self.newline_chars):
                return True

        return False

    @property
    def name(self) -> str:
        return "line_start"

    def get_expected_overhead(self) -> str:
        return "~5-15% of tokens measured (depends on line length)"


class SemanticBoundaryStrategy(MeasurementStrategy):
    """
    Measure entropy at code semantic boundaries.

    This is our primary approach: measure at positions where the model
    is making semantically significant decisions about code.

    Semantic boundaries include:
    - Function calls: foo(
    - Imports: import, from
    - Assignments: variable =
    - Method access: object.
    - Type annotations: : Type
    - List/dict access: [
    - Return statements: return
    - Class definitions: class

    Pros:
    - Focuses measurement on high-value positions
    - Lower overhead than every-token
    - Targets positions where context matters most

    Cons:
    - May miss some uncertainty positions
    - Requires pattern matching

    Args:
        include_first: Always measure first position (default: True)
        custom_triggers: Additional trigger patterns to include
    """

    # Default semantic boundary triggers
    DEFAULT_TRIGGERS = {
        # Function/method calls
        '(',

        # Imports
        'import ',
        'from ',
        'require(',

        # Assignments
        ' = ',
        ' := ',

        # Method/attribute access
        '.',

        # Type annotations
        ': ',
        '-> ',

        # Data structure access
        '[',

        # Return/yield
        'return ',
        'yield ',

        # Async
        'await ',

        # Class/function definitions
        'class ',
        'def ',
        'function ',
        'const ',
        'let ',
        'var ',
    }

    # Keywords that trigger measurement when followed by space
    TRIGGER_KEYWORDS = {
        'import', 'from', 'return', 'yield', 'await',
        'class', 'def', 'function', 'const', 'let', 'var',
        'if', 'elif', 'else', 'for', 'while', 'try', 'except',
        'with', 'as', 'async', 'new', 'throw', 'raise',
    }

    def __init__(
        self,
        include_first: bool = True,
        custom_triggers: Set[str] = None,
        include_newlines: bool = True
    ):
        self.include_first = include_first
        self.triggers = self.DEFAULT_TRIGGERS.copy()
        if custom_triggers:
            self.triggers.update(custom_triggers)
        self.include_newlines = include_newlines

    def should_measure(self, context: MeasurementContext) -> bool:
        # Always measure first position
        if self.include_first and context.position == 0:
            return True

        generated = context.generated_so_far

        # Check if generated text ends with a trigger
        for trigger in self.triggers:
            if generated.endswith(trigger):
                return True

        # Check for keyword + space patterns
        # e.g., "import " at end of generated text
        for keyword in self.TRIGGER_KEYWORDS:
            if generated.endswith(f'{keyword} '):
                return True
            if generated.endswith(f'{keyword}\n'):
                return True

        # Optionally include newlines (line boundaries)
        if self.include_newlines:
            if generated.endswith('\n'):
                return True

        return False

    @property
    def name(self) -> str:
        return "semantic_boundary"

    def get_expected_overhead(self) -> str:
        return "~10-20% of tokens measured"


class CompositeStrategy(MeasurementStrategy):
    """
    Combine multiple strategies with OR logic.

    Measures if ANY of the component strategies would measure.

    Args:
        strategies: List of strategies to combine
    """

    def __init__(self, strategies: List[MeasurementStrategy]):
        if not strategies:
            raise ValueError("At least one strategy required")
        self.strategies = strategies

    def should_measure(self, context: MeasurementContext) -> bool:
        return any(s.should_measure(context) for s in self.strategies)

    @property
    def name(self) -> str:
        names = [s.name for s in self.strategies]
        return f"composite({'+'.join(names)})"

    def get_expected_overhead(self) -> str:
        return "Combined overhead of component strategies"


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_strategy(
    name: str,
    **kwargs
) -> MeasurementStrategy:
    """
    Factory function to create measurement strategies.

    Args:
        name: Strategy name ('every_token', 'line_start', 'semantic_boundary',
              'every_n', 'composite')
        **kwargs: Strategy-specific parameters

    Returns:
        MeasurementStrategy instance

    Example:
        >>> strategy = create_strategy('semantic_boundary', include_newlines=True)
        >>> strategy = create_strategy('every_n', n=3)
    """
    strategies = {
        'every_token': EveryTokenStrategy,
        'line_start': LineStartStrategy,
        'semantic_boundary': SemanticBoundaryStrategy,
        'every_n': lambda **kw: NthTokenStrategy(n=kw.get('n', 5)),
    }

    if name not in strategies:
        raise ValueError(
            f"Unknown strategy: {name}. "
            f"Available: {list(strategies.keys())}"
        )

    if name == 'every_n':
        return strategies[name](**kwargs)

    return strategies[name](**kwargs)


def get_recommended_strategy(use_case: str = 'production') -> MeasurementStrategy:
    """
    Get recommended strategy for a use case.

    Args:
        use_case: One of 'production', 'research', 'debugging'

    Returns:
        Recommended MeasurementStrategy for the use case
    """
    recommendations = {
        'production': SemanticBoundaryStrategy(include_newlines=False),
        'research': EveryTokenStrategy(),
        'debugging': EveryTokenStrategy(),
        'balanced': SemanticBoundaryStrategy(include_newlines=True),
    }

    return recommendations.get(use_case, recommendations['balanced'])
