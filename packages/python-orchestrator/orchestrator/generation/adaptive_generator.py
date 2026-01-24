"""
Adaptive Generator Module.

This module provides the main generation loop that integrates:
1. Uncertainty monitoring (entropy/CCE)
2. Topic inference (what to retrieve)
3. Adaptive retrieval (when to retrieve)
4. Context management (token budget)

The AdaptiveGenerator orchestrates all these components to generate
responses with uncertainty-triggered context retrieval.

Phase 3, Week 7: Generation Loop & Context Management
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Union
import numpy as np
import time

from ..entropy.monitor import UncertaintyMonitor, UncertaintyResult, create_monitor
from ..entropy.measurement import MeasurementStrategy, SemanticBoundaryStrategy
from ..retrieval.topic_inference import TopicInferrer, TopicResult
from ..retrieval.adaptive import AdaptiveContextRetriever, RetrievalResult
from ..retrieval.context_manager import ContextManager, ContextPriority


@dataclass
class GenerationConfig:
    """Configuration for adaptive generation."""
    # Generation limits
    max_tokens: int = 512
    max_retrievals: int = 5  # Increased for more retrievals

    # Uncertainty monitoring
    uncertainty_method: str = "raw_entropy"
    uncertainty_threshold: float = 2.5  # Lowered further for better sensitivity
    measurement_strategy: str = "every_token"  # Default to every_token for reliability
    cooldown_tokens: int = 10  # Minimum tokens between retrievals

    # Retrieval settings
    min_topic_confidence: float = 0.3
    min_relevance_score: float = 0.1  # Lowered for more matches
    top_k_retrieval: int = 3

    # Context management
    context_budget: int = 4096
    reserve_tokens: int = 512

    # Probe settings (for Week 3 probe method)
    use_probe: bool = False
    probe_model: Any = None
    probe_scaler: Any = None
    probe_layers: List[int] = field(default_factory=lambda: [16, 24, 31])
    probe_threshold: float = 0.06

    def __post_init__(self):
        if self.use_probe and self.probe_model is None:
            raise ValueError("probe_model required when use_probe=True")


@dataclass
class RetrievalEvent:
    """Record of a retrieval event during generation."""
    position: int                    # Token position where triggered
    topic: TopicResult              # Inferred topic
    result: Optional[RetrievalResult]  # Retrieval result (None if failed)
    uncertainty_value: float        # Uncertainty value that triggered
    timestamp: float               # When it occurred


@dataclass
class GenerationResult:
    """Result of adaptive generation."""
    # Generated content
    response: str                    # Final generated response
    tokens: List[str]                # Individual tokens (decoded)

    # Traces for analysis
    entropy_trace: List[Dict]        # Entropy at each measurement point
    retrieval_events: List[RetrievalEvent]  # All retrieval events

    # Statistics
    total_tokens: int                # Tokens in response
    total_context_tokens: int        # Tokens used for context
    total_retrievals: int            # Number of retrievals performed
    generation_time: float           # Time to generate (seconds)

    # Context state
    final_context: str               # Final context used
    context_sources: List[str]       # Sources of context

    # Metadata
    config: GenerationConfig         # Config used

    # Optional fields with defaults (must come last)
    token_ids: List[int] = field(default_factory=list)  # Token IDs for analysis
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"GenerationResult(tokens={self.total_tokens}, "
            f"retrievals={self.total_retrievals}, "
            f"time={self.generation_time:.2f}s)"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_tokens': self.total_tokens,
            'total_context_tokens': self.total_context_tokens,
            'total_retrievals': self.total_retrievals,
            'generation_time': self.generation_time,
            'measurements': len(self.entropy_trace),
            'avg_entropy': np.mean([e['value'] for e in self.entropy_trace]) if self.entropy_trace else 0,
            'retrieval_positions': [e.position for e in self.retrieval_events],
        }


class AdaptiveGenerator:
    """
    Generate responses with uncertainty-triggered retrieval.

    The main generation loop:
    1. Start generating with initial context
    2. At each measurement point (determined by strategy):
       a. Compute uncertainty (CCE/entropy/probe)
       b. If uncertain, infer topic and retrieve context
       c. Add retrieved context to context manager
    3. Continue generation until done
    4. Return response with all traces

    Example:
        >>> generator = AdaptiveGenerator(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     retriever=reposynth_retriever,
        ...     config=GenerationConfig(max_tokens=256)
        ... )
        >>> result = generator.generate(
        ...     query="How does authentication work?",
        ...     initial_context=base_context
        ... )
        >>> print(result.response)
        >>> print(f"Retrievals: {result.total_retrievals}")
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        retriever=None,
        config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize adaptive generator.

        Args:
            model: Language model for generation
            tokenizer: Tokenizer for the model
            retriever: Base retriever for context (optional)
            config: Generation configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()

        # Initialize components
        self._init_monitor()
        self._init_retriever(retriever)
        self._init_context_manager()

    def _init_monitor(self):
        """Initialize uncertainty monitor."""
        if self.config.use_probe:
            from ..entropy.monitor import create_probe_monitor
            # Week 3 probe method requires model and tokenizer for lookahead
            self.monitor = create_probe_monitor(
                model=self.model,
                tokenizer=self.tokenizer,
                probe_model=self.config.probe_model,
                probe_scaler=self.config.probe_scaler,
                probe_layers=self.config.probe_layers,
                threshold=self.config.probe_threshold,
                strategy=self.config.measurement_strategy,
                max_retrievals=self.config.max_retrievals,
                cooldown_tokens=self.config.cooldown_tokens,
            )
        else:
            self.monitor = create_monitor(
                method=self.config.uncertainty_method,
                threshold=self.config.uncertainty_threshold,
                tokenizer=self.tokenizer,
                strategy=self.config.measurement_strategy,
                max_retrievals=self.config.max_retrievals,
                cooldown_tokens=self.config.cooldown_tokens,
            )

    def _init_retriever(self, base_retriever):
        """Initialize adaptive retriever."""
        self.topic_inferrer = TopicInferrer(
            tokenizer=self.tokenizer,
        )

        self.retriever = AdaptiveContextRetriever(
            base_retriever=base_retriever,
            topic_inferrer=self.topic_inferrer,
            max_retrievals=self.config.max_retrievals,
            min_confidence=self.config.min_topic_confidence,
            min_relevance=self.config.min_relevance_score,
            top_k=self.config.top_k_retrieval,
        )

    def _init_context_manager(self):
        """Initialize context manager."""
        self.context_manager = ContextManager(
            max_tokens=self.config.context_budget,
            reserve_tokens=self.config.reserve_tokens,
            tokenizer=self.tokenizer,
        )

    def reset(self):
        """Reset all components for new generation."""
        self.monitor.reset()
        self.retriever.reset()
        self.context_manager.reset()

    def generate(
        self,
        query: str,
        initial_context: str = "",
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate response with adaptive retrieval.

        Args:
            query: User query/prompt
            initial_context: Initial context to include
            system_prompt: Optional system prompt

        Returns:
            GenerationResult with response and traces
        """
        start_time = time.time()
        self.reset()

        # Initialize context
        if system_prompt:
            self.context_manager.add_segment(
                content=system_prompt,
                source="system",
                priority=ContextPriority.CRITICAL,
            )

        if initial_context:
            self.context_manager.add_segment(
                content=initial_context,
                source="initial",
                priority=ContextPriority.HIGH,
            )

        self.context_manager.add_segment(
            content=query,
            source="query",
            priority=ContextPriority.CRITICAL,
        )

        # Generation state - track token IDs for proper decoding
        generated_token_ids: List[int] = []
        entropy_trace: List[Dict] = []
        retrieval_events: List[RetrievalEvent] = []

        # Main generation loop
        for i in range(self.config.max_tokens):
            # Build current prompt - decode full sequence for proper spacing
            current_context = self.context_manager.get_formatted_context()
            if generated_token_ids and self.tokenizer:
                generated_so_far = self.tokenizer.decode(
                    generated_token_ids,
                    skip_special_tokens=True
                )
            else:
                generated_so_far = ""
            full_prompt = current_context + generated_so_far

            # Get model output
            logits, hidden_states = self._forward(full_prompt)

            if logits is None:
                break

            # Monitor uncertainty
            # For probe method, pass full_prompt for lookahead analysis
            # For other methods, just pass logits and token
            if self.config.use_probe:
                result = self.monitor.step(logits, generated_so_far, full_prompt=full_prompt)
            else:
                result = self.monitor.step(logits, generated_so_far)

            if result:
                entropy_trace.append({
                    'position': i,
                    'value': result.value,
                    'should_retrieve': result.should_retrieve,
                    'method': result.method,
                })

                # Check if we should retrieve
                if result.should_retrieve:
                    retrieval_event = self._handle_retrieval(
                        logits=logits,
                        context=generated_so_far,
                        position=i,
                        uncertainty_value=result.value,
                        original_query=query,  # Use original query for retrieval!
                    )
                    if retrieval_event:
                        retrieval_events.append(retrieval_event)

            # Sample next token - get ID instead of decoded string
            next_token_id = self._sample_token_id(logits)
            if next_token_id is None:
                break

            # Check for end tokens
            if self.tokenizer:
                token_str = self.tokenizer.decode([next_token_id])
                if token_str in ['</s>', '<|endoftext|>', '<|end|>', '']:
                    break
                # Also check by ID for common EOS tokens
                if next_token_id == self.tokenizer.eos_token_id:
                    break

            generated_token_ids.append(next_token_id)

        generation_time = time.time() - start_time

        # Decode final response properly
        if generated_token_ids and self.tokenizer:
            final_response = self.tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True
            )
            # Also get individual token strings for analysis
            token_strings = [self.tokenizer.decode([tid]) for tid in generated_token_ids]
        else:
            final_response = ""
            token_strings = []

        return GenerationResult(
            response=final_response,
            tokens=token_strings,
            token_ids=generated_token_ids,
            entropy_trace=entropy_trace,
            retrieval_events=retrieval_events,
            total_tokens=len(generated_token_ids),
            total_context_tokens=self.context_manager.current_tokens,
            total_retrievals=len(retrieval_events),
            generation_time=generation_time,
            final_context=self.context_manager.get_formatted_context(),
            context_sources=[s.source for s in self.context_manager.segments],
            config=self.config,
        )

    def _forward(self, prompt: str):
        """
        Get model output logits for next token prediction.

        Note: For probe method, hidden states are extracted in the monitor
        during lookahead analysis, not here. This saves memory by not
        storing hidden states for the main forward pass.

        Returns:
            (logits, None) tuple - hidden_states no longer returned
        """
        if self.model is None:
            return None, None

        try:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                # Don't request hidden_states here - saves memory
                # Probe method does its own forward passes with hidden_states
                outputs = self.model(**inputs)

            logits = outputs.logits[0, -1, :].cpu().numpy()

            # Clean up to save memory
            del outputs, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return logits, None

        except Exception as e:
            print(f"Forward pass error: {e}")
            return None, None

    def _sample_token_id(self, logits: np.ndarray) -> Optional[int]:
        """Sample next token ID from logits."""
        try:
            # Greedy decoding (can be extended to support sampling)
            from ..entropy.calculator import softmax
            probs = softmax(logits)
            token_id = int(np.argmax(probs))
            return token_id
        except Exception:
            return None

    def _sample_token(self, logits: np.ndarray) -> Optional[str]:
        """Sample next token from logits (legacy, returns decoded string)."""
        if self.tokenizer is None:
            return None
        token_id = self._sample_token_id(logits)
        if token_id is None:
            return None
        return self.tokenizer.decode([token_id])

    def _handle_retrieval(
        self,
        logits: np.ndarray,
        context: str,
        position: int,
        uncertainty_value: float,
        original_query: str = "",
    ) -> Optional[RetrievalEvent]:
        """Handle a retrieval event."""
        # Attempt retrieval using ORIGINAL QUERY (not inferred topic)
        # CCE determines WHEN to retrieve, but WHAT to retrieve should
        # be based on the original user query for best results
        retrieval_result = self.retriever.retrieve_on_uncertainty(
            logits=logits,
            context=context,
            position=position,
            uncertainty_value=uncertainty_value,
            should_retrieve=True,
            original_query=original_query,  # Pass original query!
        )

        # Create event record
        event = RetrievalEvent(
            position=position,
            topic=self.topic_inferrer.last_result,
            result=retrieval_result,
            uncertainty_value=uncertainty_value,
            timestamp=time.time(),
        )

        # If retrieval succeeded, add to context
        if retrieval_result:
            self.context_manager.add_segment(
                content=retrieval_result.content,
                source=retrieval_result.source,
                priority=ContextPriority.MEDIUM,
                relevance_score=retrieval_result.relevance_score,
                retrieval_id=retrieval_result.retrieval_id,
            )

        return event

    def generate_with_callback(
        self,
        query: str,
        initial_context: str = "",
        on_token: Optional[Callable[[str, int], None]] = None,
        on_retrieval: Optional[Callable[[RetrievalEvent], None]] = None,
    ) -> GenerationResult:
        """
        Generate with callbacks for streaming/monitoring.

        Args:
            query: User query
            initial_context: Initial context
            on_token: Callback for each token (token, position)
            on_retrieval: Callback for retrieval events

        Returns:
            GenerationResult
        """
        # This is a simplified version - full implementation would
        # integrate callbacks into the main generate() loop
        result = self.generate(query, initial_context)

        # Call callbacks with recorded events
        if on_token:
            for i, token in enumerate(result.tokens):
                on_token(token, i)

        if on_retrieval:
            for event in result.retrieval_events:
                on_retrieval(event)

        return result


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_adaptive_generator(
    model=None,
    tokenizer=None,
    retriever=None,
    **kwargs
) -> AdaptiveGenerator:
    """
    Factory function to create adaptive generators.

    Args:
        model: Language model
        tokenizer: Tokenizer
        retriever: Base retriever
        **kwargs: Config options

    Returns:
        Configured AdaptiveGenerator
    """
    config = GenerationConfig(**kwargs)
    return AdaptiveGenerator(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        config=config,
    )
