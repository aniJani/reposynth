"""
Context Manager Module for Adaptive Context Retrieval.

This module manages the context window during adaptive retrieval:
1. Tracks token budget (max context size)
2. Adds new context from retrievals
3. Evicts low-priority context when budget is exceeded
4. Maintains provenance (where context came from)

Phase 3, Week 7: Generation Loop & Context Management
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from enum import Enum
import numpy as np


class ContextPriority(Enum):
    """Priority levels for context segments."""
    CRITICAL = 5      # Never evict (user query, system prompt)
    HIGH = 4          # Rarely evict (relevant file contents)
    MEDIUM = 3        # Default for retrieved context
    LOW = 2           # Can be evicted (supplementary info)
    EPHEMERAL = 1     # First to evict (temporary context)


@dataclass
class ContextSegment:
    """A segment of context in the window."""
    content: str                           # The actual content
    source: str                            # Where it came from
    priority: ContextPriority              # Eviction priority
    token_count: int                       # Number of tokens
    timestamp: int                         # When it was added (for LRU)
    segment_id: int                        # Unique identifier

    # Optional metadata
    relevance_score: Optional[float] = None   # Retrieval relevance
    retrieval_id: Optional[int] = None        # Link to retrieval event
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ContextSegment(source='{self.source}', "
            f"priority={self.priority.name}, tokens={self.token_count})"
        )


class ContextManager:
    """
    Manages context window during adaptive retrieval.

    Features:
    - Token budget management (stay within max_tokens)
    - Priority-based eviction (remove low priority first)
    - LRU eviction within same priority (remove oldest first)
    - Provenance tracking (know where context came from)
    - Context formatting (combine segments into prompt)

    Example:
        >>> manager = ContextManager(max_tokens=4096)
        >>> manager.add_segment(
        ...     content="System prompt here",
        ...     source="system",
        ...     priority=ContextPriority.CRITICAL
        ... )
        >>> manager.add_segment(
        ...     content=retrieved_context,
        ...     source="src/auth/login.ts",
        ...     priority=ContextPriority.MEDIUM
        ... )
        >>> prompt = manager.get_formatted_context()
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        reserve_tokens: int = 512,
        tokenizer=None,
        format_template: Optional[str] = None,
    ):
        """
        Initialize context manager.

        Args:
            max_tokens: Maximum total tokens allowed
            reserve_tokens: Tokens to reserve for generation output
            tokenizer: Optional tokenizer for accurate counting
            format_template: Template for formatting context
        """
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.effective_max = max_tokens - reserve_tokens
        self.tokenizer = tokenizer
        self.format_template = format_template or "{content}"

        # State
        self.segments: List[ContextSegment] = []
        self.current_tokens = 0
        self._segment_id_counter = 0
        self._timestamp_counter = 0

        # Statistics
        self.stats = {
            'total_added': 0,
            'total_evicted': 0,
            'eviction_events': [],
        }

    def add_segment(
        self,
        content: str,
        source: str,
        priority: ContextPriority = ContextPriority.MEDIUM,
        relevance_score: Optional[float] = None,
        retrieval_id: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Add a context segment.

        Args:
            content: The content to add
            source: Source identifier
            priority: Eviction priority
            relevance_score: Optional relevance score
            retrieval_id: Optional link to retrieval event
            metadata: Optional metadata

        Returns:
            True if added successfully, False if couldn't fit even after eviction
        """
        token_count = self._count_tokens(content)

        # Check if it fits without eviction
        if self.current_tokens + token_count <= self.effective_max:
            return self._add_segment_internal(
                content, source, priority, token_count,
                relevance_score, retrieval_id, metadata
            )

        # Need to evict
        tokens_needed = (self.current_tokens + token_count) - self.effective_max
        evicted = self._evict_lowest_priority(tokens_needed, priority)

        if not evicted:
            # Couldn't make room even after eviction
            return False

        return self._add_segment_internal(
            content, source, priority, token_count,
            relevance_score, retrieval_id, metadata
        )

    def _add_segment_internal(
        self,
        content: str,
        source: str,
        priority: ContextPriority,
        token_count: int,
        relevance_score: Optional[float],
        retrieval_id: Optional[int],
        metadata: Optional[Dict],
    ) -> bool:
        """Internal method to add segment after space is confirmed."""
        self._segment_id_counter += 1
        self._timestamp_counter += 1

        segment = ContextSegment(
            content=content,
            source=source,
            priority=priority,
            token_count=token_count,
            timestamp=self._timestamp_counter,
            segment_id=self._segment_id_counter,
            relevance_score=relevance_score,
            retrieval_id=retrieval_id,
            metadata=metadata or {},
        )

        self.segments.append(segment)
        self.current_tokens += token_count
        self.stats['total_added'] += 1

        return True

    def _evict_lowest_priority(
        self,
        tokens_needed: int,
        new_priority: ContextPriority
    ) -> bool:
        """
        Evict lowest priority segments to make room.

        Args:
            tokens_needed: How many tokens to free
            new_priority: Priority of new content (won't evict same or higher)

        Returns:
            True if enough space was freed
        """
        # Sort segments by priority (ascending) then timestamp (ascending = oldest first)
        evictable = [
            s for s in self.segments
            if s.priority.value < new_priority.value
        ]

        if not evictable:
            return False

        evictable.sort(key=lambda s: (s.priority.value, s.timestamp))

        freed = 0
        to_remove = []

        for segment in evictable:
            if freed >= tokens_needed:
                break
            to_remove.append(segment)
            freed += segment.token_count

        if freed < tokens_needed:
            return False

        # Remove segments
        for segment in to_remove:
            self.segments.remove(segment)
            self.current_tokens -= segment.token_count
            self.stats['total_evicted'] += 1
            self.stats['eviction_events'].append({
                'segment_id': segment.segment_id,
                'source': segment.source,
                'tokens': segment.token_count,
                'priority': segment.priority.name,
            })

        return True

    def remove_segment(self, segment_id: int) -> bool:
        """
        Remove a specific segment by ID.

        Args:
            segment_id: ID of segment to remove

        Returns:
            True if removed, False if not found
        """
        for segment in self.segments:
            if segment.segment_id == segment_id:
                self.segments.remove(segment)
                self.current_tokens -= segment.token_count
                return True
        return False

    def get_formatted_context(
        self,
        separator: str = "\n\n",
        include_sources: bool = False,
    ) -> str:
        """
        Get all context segments formatted as a single string.

        Args:
            separator: Separator between segments
            include_sources: Whether to include source annotations

        Returns:
            Formatted context string
        """
        if not self.segments:
            return ""

        # Sort by priority (highest first) then timestamp (oldest first)
        sorted_segments = sorted(
            self.segments,
            key=lambda s: (-s.priority.value, s.timestamp)
        )

        parts = []
        for segment in sorted_segments:
            if include_sources:
                parts.append(f"[{segment.source}]\n{segment.content}")
            else:
                parts.append(segment.content)

        return separator.join(parts)

    def get_context_by_source(self, source: str) -> Optional[ContextSegment]:
        """Get segment by source name."""
        for segment in self.segments:
            if segment.source == source:
                return segment
        return None

    def get_context_by_priority(
        self,
        priority: ContextPriority
    ) -> List[ContextSegment]:
        """Get all segments with given priority."""
        return [s for s in self.segments if s.priority == priority]

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        # Fallback: estimate ~4 chars per token
        return len(text) // 4

    def get_statistics(self) -> Dict[str, Any]:
        """Get context manager statistics."""
        priority_breakdown = {}
        for priority in ContextPriority:
            segments = [s for s in self.segments if s.priority == priority]
            priority_breakdown[priority.name] = {
                'count': len(segments),
                'tokens': sum(s.token_count for s in segments),
            }

        return {
            'current_tokens': self.current_tokens,
            'max_tokens': self.effective_max,
            'utilization': self.current_tokens / self.effective_max if self.effective_max > 0 else 0,
            'segment_count': len(self.segments),
            'priority_breakdown': priority_breakdown,
            'total_added': self.stats['total_added'],
            'total_evicted': self.stats['total_evicted'],
        }

    def reset(self):
        """Reset context manager to initial state."""
        self.segments = []
        self.current_tokens = 0
        self._segment_id_counter = 0
        self._timestamp_counter = 0
        self.stats = {
            'total_added': 0,
            'total_evicted': 0,
            'eviction_events': [],
        }

    def get_token_budget_remaining(self) -> int:
        """Get remaining token budget."""
        return self.effective_max - self.current_tokens

    def can_fit(self, content: str) -> bool:
        """Check if content can fit (without eviction)."""
        tokens = self._count_tokens(content)
        return self.current_tokens + tokens <= self.effective_max

    def visualize_context(self) -> str:
        """Create a visual representation of context allocation."""
        lines = []
        lines.append(f"Context Window: {self.current_tokens}/{self.effective_max} tokens")
        lines.append(f"Utilization: {100 * self.current_tokens / self.effective_max:.1f}%")
        lines.append("-" * 50)

        for segment in self.segments:
            bar_width = int(40 * segment.token_count / self.effective_max)
            bar = "█" * bar_width
            lines.append(
                f"{segment.priority.name[:4]:4} | {bar:40} | "
                f"{segment.token_count:5} | {segment.source}"
            )

        return "\n".join(lines)


# ============================================================================
# SPECIALIZED CONTEXT MANAGERS
# ============================================================================

class SlidingWindowContextManager(ContextManager):
    """
    Context manager with sliding window strategy.

    Keeps most recent context, evicts oldest when full.
    """

    def _evict_lowest_priority(
        self,
        tokens_needed: int,
        new_priority: ContextPriority
    ) -> bool:
        """Evict oldest segments first (sliding window)."""
        # Sort by timestamp (oldest first), but respect CRITICAL priority
        evictable = [
            s for s in self.segments
            if s.priority != ContextPriority.CRITICAL
        ]

        if not evictable:
            return False

        evictable.sort(key=lambda s: s.timestamp)

        freed = 0
        to_remove = []

        for segment in evictable:
            if freed >= tokens_needed:
                break
            to_remove.append(segment)
            freed += segment.token_count

        if freed < tokens_needed:
            return False

        for segment in to_remove:
            self.segments.remove(segment)
            self.current_tokens -= segment.token_count
            self.stats['total_evicted'] += 1

        return True


class RelevanceBasedContextManager(ContextManager):
    """
    Context manager that evicts based on relevance scores.

    Lower relevance = evicted first.
    """

    def _evict_lowest_priority(
        self,
        tokens_needed: int,
        new_priority: ContextPriority
    ) -> bool:
        """Evict lowest relevance segments first."""
        # Sort by relevance (ascending), then priority, then timestamp
        evictable = [
            s for s in self.segments
            if s.priority.value < new_priority.value
        ]

        if not evictable:
            return False

        # Use relevance_score if available, else use priority
        def sort_key(s):
            relevance = s.relevance_score if s.relevance_score is not None else 0.5
            return (relevance, s.priority.value, -s.timestamp)

        evictable.sort(key=sort_key)

        freed = 0
        to_remove = []

        for segment in evictable:
            if freed >= tokens_needed:
                break
            to_remove.append(segment)
            freed += segment.token_count

        if freed < tokens_needed:
            return False

        for segment in to_remove:
            self.segments.remove(segment)
            self.current_tokens -= segment.token_count
            self.stats['total_evicted'] += 1

        return True


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_context_manager(
    max_tokens: int = 4096,
    strategy: str = "priority",
    tokenizer=None,
    **kwargs
) -> ContextManager:
    """
    Factory function to create context managers.

    Args:
        max_tokens: Maximum token budget
        strategy: Eviction strategy ('priority', 'sliding', 'relevance')
        tokenizer: Optional tokenizer
        **kwargs: Additional arguments

    Returns:
        Configured ContextManager
    """
    strategies = {
        'priority': ContextManager,
        'sliding': SlidingWindowContextManager,
        'relevance': RelevanceBasedContextManager,
    }

    manager_class = strategies.get(strategy, ContextManager)
    return manager_class(max_tokens=max_tokens, tokenizer=tokenizer, **kwargs)
