"""
Simple Mock Model for Week 7 Integration Tests.

Simplified version that avoids complex syntax issues.
"""

from typing import Tuple
from dataclasses import dataclass
import numpy as np
import random


@dataclass
class ModelOutput:
    """Output from model forward pass."""

    logits: np.ndarray  # [vocab_size] array
    predicted_token: str  # The token that was predicted
    predicted_token_id: int  # Token ID of predicted token


class MockCodeLlama:
    """
    Simplified mock of CodeLlama-7B behavior.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        seed: int = 42,
    ):
        """Initialize mock model."""
        self.vocab_size = vocab_size
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def __call__(self, prompt: str, output_hidden_states: bool = False) -> ModelOutput:
        """Run forward pass."""
        # Simple logits generation
        logits = np.random.randn(self.vocab_size)

        # Sample token
        token_id = int(np.random.choice(self.vocab_size))
        token_str = f"token{token_id % 100}"

        # Detect if this should be high entropy (missing context)
        is_api_call = any(
            kw in prompt.lower()
            for kw in ["pandas", "react", "firebase", "fastapi", "numpy"]
        )

        if is_api_call:
            # Boost probabilities to spread out
            logits = logits * 0.5  # More uniform = higher entropy
        else:
            # Sharpen distribution
            logits[token_id] += 5.0  # Higher prob = lower entropy

        return ModelOutput(
            logits=logits, predicted_token=token_str, predicted_token_id=token_id
        )

    def forward(self, prompt: str, output_hidden_states: bool = False) -> ModelOutput:
        """Alternative interface."""
        return self(prompt, output_hidden_states)
