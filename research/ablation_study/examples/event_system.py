"""
Event System Module

Provides pub/sub messaging, event sourcing, and async event handling
for decoupled application architecture.
"""

import asyncio
import threading
import queue
import time
import uuid
from typing import Callable, Dict, List, Any, Optional, Set, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict
import json
import weakref


T = TypeVar('T')


class EventPriority(Enum):
    """Priority levels for event processing."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """
    Base event class representing something that happened in the system.

    Events are immutable records of facts that occurred at a specific time.
    """
    event_type: str
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'payload': self.payload,
            'timestamp': self.timestamp,
            'source': self.source,
            'correlation_id': self.correlation_id,
            'priority': self.priority.name,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        data['priority'] = EventPriority[data.get('priority', 'NORMAL')]
        return cls(**data)

    def with_correlation(self, correlation_id: str) -> 'Event':
        """Create a new event with the given correlation ID."""
        return Event(
            event_type=self.event_type,
            payload=self.payload,
            source=self.source,
            correlation_id=correlation_id,
            priority=self.priority,
            metadata=self.metadata
        )


# Type alias for event handlers
EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], 'asyncio.Future']


@dataclass
class Subscription:
    """Represents a subscription to events."""
    handler: EventHandler
    event_types: Set[str]
    filter_func: Optional[Callable[[Event], bool]] = None
    priority: EventPriority = EventPriority.NORMAL
    max_retries: int = 3
    active: bool = True

    def matches(self, event: Event) -> bool:
        """Check if event matches this subscription."""
        if not self.active:
            return False
        if '*' not in self.event_types and event.event_type not in self.event_types:
            return False
        if self.filter_func and not self.filter_func(event):
            return False
        return True


class EventBus:
    """
    Simple synchronous event bus for pub/sub messaging.

    Usage:
        bus = EventBus()

        # Subscribe to events
        @bus.subscribe('user.created')
        def on_user_created(event):
            print(f"User created: {event.payload}")

        # Publish events
        bus.publish(Event('user.created', {'user_id': 123}))
    """

    def __init__(self):
        self._subscriptions: List[Subscription] = []
        self._lock = threading.Lock()
        self._event_history: List[Event] = []
        self._history_limit = 1000

    def subscribe(
        self,
        *event_types: str,
        filter_func: Callable[[Event], bool] = None,
        priority: EventPriority = EventPriority.NORMAL
    ) -> Callable:
        """
        Decorator to subscribe a handler to event types.

        Args:
            event_types: Event types to subscribe to ('*' for all)
            filter_func: Optional filter function
            priority: Handler priority
        """
        def decorator(handler: EventHandler) -> EventHandler:
            subscription = Subscription(
                handler=handler,
                event_types=set(event_types) if event_types else {'*'},
                filter_func=filter_func,
                priority=priority
            )
            with self._lock:
                self._subscriptions.append(subscription)
            return handler
        return decorator

    def subscribe_handler(
        self,
        handler: EventHandler,
        *event_types: str,
        **kwargs
    ) -> Subscription:
        """
        Subscribe a handler function to event types.

        Returns:
            Subscription object that can be used to unsubscribe
        """
        subscription = Subscription(
            handler=handler,
            event_types=set(event_types) if event_types else {'*'},
            **kwargs
        )
        with self._lock:
            self._subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> bool:
        """Unsubscribe a subscription."""
        with self._lock:
            if subscription in self._subscriptions:
                subscription.active = False
                self._subscriptions.remove(subscription)
                return True
            return False

    def publish(self, event: Event) -> int:
        """
        Publish an event to all matching subscribers.

        Args:
            event: Event to publish

        Returns:
            Number of handlers that received the event
        """
        with self._lock:
            # Record in history
            self._event_history.append(event)
            if len(self._event_history) > self._history_limit:
                self._event_history = self._event_history[-self._history_limit:]

            # Get matching subscriptions sorted by priority
            matching = [s for s in self._subscriptions if s.matches(event)]
            matching.sort(key=lambda s: s.priority.value, reverse=True)

        # Notify handlers
        notified = 0
        for sub in matching:
            try:
                sub.handler(event)
                notified += 1
            except Exception as e:
                # Log error but continue with other handlers
                print(f"Error in event handler: {e}")

        return notified

    def emit(self, event_type: str, payload: Dict = None, **kwargs) -> Event:
        """
        Convenience method to create and publish an event.

        Args:
            event_type: Type of event
            payload: Event payload
            **kwargs: Additional event attributes
        """
        event = Event(event_type=event_type, payload=payload or {}, **kwargs)
        self.publish(event)
        return event

    def get_history(
        self,
        event_type: str = None,
        since: float = None,
        limit: int = 100
    ) -> List[Event]:
        """Get event history with optional filtering."""
        with self._lock:
            events = self._event_history.copy()

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if since:
            events = [e for e in events if e.timestamp >= since]

        return events[-limit:]


class AsyncEventBus:
    """
    Asynchronous event bus for non-blocking event processing.

    Usage:
        bus = AsyncEventBus()

        @bus.subscribe('order.placed')
        async def on_order_placed(event):
            await process_order(event.payload)

        await bus.publish(Event('order.placed', {'order_id': 456}))
    """

    def __init__(self, max_workers: int = 10):
        self._subscriptions: Dict[str, List[AsyncEventHandler]] = defaultdict(list)
        self._semaphore = asyncio.Semaphore(max_workers)

    def subscribe(self, *event_types: str):
        """Decorator to subscribe an async handler."""
        def decorator(handler: AsyncEventHandler) -> AsyncEventHandler:
            for event_type in event_types:
                self._subscriptions[event_type].append(handler)
            return handler
        return decorator

    async def publish(self, event: Event) -> List[Any]:
        """Publish an event and wait for all handlers to complete."""
        handlers = self._subscriptions.get(event.event_type, [])
        handlers.extend(self._subscriptions.get('*', []))

        async def run_handler(handler):
            async with self._semaphore:
                return await handler(event)

        results = await asyncio.gather(
            *[run_handler(h) for h in handlers],
            return_exceptions=True
        )

        return results

    async def publish_fire_and_forget(self, event: Event) -> None:
        """Publish an event without waiting for handlers."""
        handlers = self._subscriptions.get(event.event_type, [])
        handlers.extend(self._subscriptions.get('*', []))

        for handler in handlers:
            asyncio.create_task(handler(event))


class EventQueue:
    """
    Background event processing queue with retry support.

    Useful for handling events asynchronously without blocking the publisher.
    """

    def __init__(self, max_workers: int = 5, max_retries: int = 3):
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._workers: List[threading.Thread] = []
        self._running = False
        self._max_workers = max_workers
        self._max_retries = max_retries

    def register(self, event_type: str, handler: EventHandler) -> None:
        """Register a handler for an event type."""
        self._handlers[event_type].append(handler)

    def enqueue(self, event: Event) -> None:
        """Add an event to the processing queue."""
        # Priority queue orders by first element (lower = higher priority)
        priority = 10 - event.priority.value
        self._queue.put((priority, event, 0))  # (priority, event, retry_count)

    def start(self) -> None:
        """Start worker threads."""
        if self._running:
            return

        self._running = True
        for i in range(self._max_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self._workers.append(worker)

    def stop(self) -> None:
        """Stop worker threads."""
        self._running = False
        # Put sentinel values to unblock workers
        for _ in self._workers:
            self._queue.put((999, None, 0))

    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        while self._running:
            try:
                priority, event, retry_count = self._queue.get(timeout=1.0)

                if event is None:  # Sentinel value
                    break

                self._process_event(event, retry_count)

            except queue.Empty:
                continue

    def _process_event(self, event: Event, retry_count: int) -> None:
        """Process a single event."""
        handlers = self._handlers.get(event.event_type, [])
        handlers.extend(self._handlers.get('*', []))

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                if retry_count < self._max_retries:
                    # Re-queue with incremented retry count
                    priority = 10 - event.priority.value
                    self._queue.put((priority, event, retry_count + 1))
                else:
                    print(f"Event processing failed after {self._max_retries} retries: {e}")


class EventStore:
    """
    Event sourcing store for persisting and replaying events.

    Supports event streams, snapshots, and projections.
    """

    def __init__(self):
        self._streams: Dict[str, List[Event]] = defaultdict(list)
        self._snapshots: Dict[str, Tuple[int, Any]] = {}
        self._projections: Dict[str, Callable[[Any, Event], Any]] = {}

    def append(self, stream_id: str, event: Event) -> int:
        """
        Append an event to a stream.

        Returns:
            The new stream version number
        """
        self._streams[stream_id].append(event)
        return len(self._streams[stream_id])

    def get_stream(
        self,
        stream_id: str,
        from_version: int = 0,
        to_version: int = None
    ) -> List[Event]:
        """Get events from a stream."""
        events = self._streams.get(stream_id, [])
        if to_version:
            return events[from_version:to_version]
        return events[from_version:]

    def get_all_events(
        self,
        event_type: str = None,
        since: float = None
    ) -> List[Event]:
        """Get all events across all streams."""
        all_events = []
        for events in self._streams.values():
            all_events.extend(events)

        all_events.sort(key=lambda e: e.timestamp)

        if event_type:
            all_events = [e for e in all_events if e.event_type == event_type]
        if since:
            all_events = [e for e in all_events if e.timestamp >= since]

        return all_events

    def save_snapshot(self, stream_id: str, version: int, state: Any) -> None:
        """Save a snapshot of aggregate state."""
        self._snapshots[stream_id] = (version, state)

    def get_snapshot(self, stream_id: str) -> Optional[Tuple[int, Any]]:
        """Get the latest snapshot for a stream."""
        return self._snapshots.get(stream_id)

    def register_projection(
        self,
        name: str,
        reducer: Callable[[Any, Event], Any]
    ) -> None:
        """Register a projection (event reducer)."""
        self._projections[name] = reducer

    def project(
        self,
        projection_name: str,
        stream_id: str = None,
        initial_state: Any = None
    ) -> Any:
        """
        Run a projection over events.

        Args:
            projection_name: Name of registered projection
            stream_id: Optional stream to project (None = all streams)
            initial_state: Initial state for the projection
        """
        reducer = self._projections.get(projection_name)
        if not reducer:
            raise ValueError(f"Unknown projection: {projection_name}")

        if stream_id:
            events = self.get_stream(stream_id)
        else:
            events = self.get_all_events()

        state = initial_state
        for event in events:
            state = reducer(state, event)

        return state


# Domain Events for common scenarios

class DomainEvent(Event):
    """Base class for domain-specific events."""

    def __init__(self, aggregate_id: str, payload: Dict = None, **kwargs):
        super().__init__(
            event_type=self.__class__.__name__,
            payload={'aggregate_id': aggregate_id, **(payload or {})},
            **kwargs
        )
        self.aggregate_id = aggregate_id


class EntityCreated(DomainEvent):
    """Event fired when a new entity is created."""
    pass


class EntityUpdated(DomainEvent):
    """Event fired when an entity is updated."""
    pass


class EntityDeleted(DomainEvent):
    """Event fired when an entity is deleted."""
    pass
