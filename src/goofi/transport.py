"""Unified transport for goofi-pipe.

Replaces `connection.py` entirely. Provides a single `Publisher` / `Subscriber`
/ `Listener` / `Notifier` / `WaitSet` API backed by two implementations:

- **`Ipc*`** — iceoryx2 publish/subscribe + sibling event service. Used for
  every channel that crosses a process boundary. Data plane uses
  latest-wins (`enable_safe_overflow(True)`); control/status planes use
  reliable, in-order delivery (`enable_safe_overflow(False)`).
- **`Thread*`** — single-slot under a lock + `threading.Event`. Used for
  every channel between two nodes in the *same* process group. Producer
  serializes through the same codec path as iceoryx2 — guarantees the same
  atomic-instance and latest-wins semantics with zero shared-object risk.

Both implementations pass `bytes` through their `send` / `take_latest` /
`take_next` API; encoding/decoding is the caller's responsibility (see
`codec.py`).

The transport assumes a single iceoryx2 node per OS process (created lazily
via `_get_node()`). Forking parents and children each get their own.
"""
from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import ClassVar, Optional

import iceoryx2 as iox2

log = logging.getLogger(__name__)

# Initial slice allocation for an iceoryx2 publisher. iceoryx2 grows the
# slice on demand via `AllocationStrategy.PowerOfTwo`, so this is just the
# starting size — 64 KiB comfortably fits typical EEG / audio / small
# control frames in one allocation; video / screen-grab grows transparently
# on the first oversized push (each doubling is microseconds).
DEFAULT_MAX_PAYLOAD = 64 * 1024

# Reasonable history for the reliable ctrl/status planes. We want a few
# messages of buffer so a slow consumer isn't dropped, but not so much that
# stale params pile up after a long pause.
CTRL_HISTORY_SIZE = 64

# Process-wide instance id, embedded in every iceoryx2 service name so
# concurrent goofi-pipe instances on the same host don't collide. Set by
# `set_instance_id()`, defaults to "{pid}" if never called.
_instance_id: str = ""


def set_instance_id(iid: str) -> None:
    global _instance_id
    _instance_id = iid


def get_instance_id() -> str:
    return _instance_id or str(os.getpid())


def data_service_name(src_node: str, slot_name: str) -> str:
    """Canonical service name for a node's output-slot data plane."""
    return f"goofi.{get_instance_id()}.data.{src_node}.{slot_name}"


def event_service_name(channel_name: str) -> str:
    """Sibling event service name for an arbitrary channel."""
    return channel_name + ".evt"


def ctrl_service_name(node_name: str) -> str:
    """Canonical service name for the manager→node control plane."""
    return f"goofi.{get_instance_id()}.ctrl.{node_name}"


def status_service_name(node_name: str) -> str:
    """Canonical service name for the node→manager status plane."""
    return f"goofi.{get_instance_id()}.status.{node_name}"


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class Publisher(ABC):
    @abstractmethod
    def send(self, payload: bytes) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class Subscriber(ABC):
    @abstractmethod
    def take_latest(self) -> Optional[bytes]:
        """Drain all pending samples and return the newest, or `None` if no
        new sample is available. Used on the data plane (latest-wins)."""

    @abstractmethod
    def take_next(self) -> Optional[bytes]:
        """Return the next pending sample in FIFO order, or `None`. Used on
        ctrl/status (reliable, in-order)."""

    @abstractmethod
    def close(self) -> None: ...


class Notifier(ABC):
    @abstractmethod
    def notify(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class Listener(ABC):
    """Wakes when a paired `Notifier` fires. Can be attached to a `WaitSet`."""


# ---------------------------------------------------------------------------
# iceoryx2 node singleton (per process)
# ---------------------------------------------------------------------------


class _NodeSingleton:
    _node: ClassVar[object] = None
    _pid: ClassVar[int] = -1
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get(cls):
        # Detect fork: pid changed → previous node is invalid in this process.
        if cls._node is None or cls._pid != os.getpid():
            with cls._lock:
                if cls._node is None or cls._pid != os.getpid():
                    iox2.set_log_level(iox2.LogLevel.Error)
                    cls._node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
                    cls._pid = os.getpid()
        return cls._node

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._node = None
            cls._pid = -1


# ---------------------------------------------------------------------------
# iceoryx2 transport
# ---------------------------------------------------------------------------


class IpcPublisher(Publisher):
    """Wraps an iceoryx2 publisher with a power-of-two-growing slice buffer.

    `initial_max_slice_len` sets the starting allocation; iceoryx2 grows it
    automatically on demand via `AllocationStrategy.PowerOfTwo`. Frames
    that exceed the current slice cap trigger a reallocation rather than
    an error — important for video / screen-grab nodes whose frame size
    isn't known until the camera opens.
    """

    def __init__(
        self,
        name: str,
        *,
        max_payload: int = DEFAULT_MAX_PAYLOAD,
        latest_wins: bool = True,
        max_subscribers: int = 16,
    ) -> None:
        self._name = name
        builder = (
            _NodeSingleton.get()
            .service_builder(iox2.ServiceName.new(name))
            .publish_subscribe(iox2.Slice[ctypes.c_uint8])
            .enable_safe_overflow(latest_wins)
            .max_publishers(1)
            .max_subscribers(max_subscribers)
        )
        if not latest_wins:
            builder = builder.history_size(CTRL_HISTORY_SIZE).subscriber_max_buffer_size(CTRL_HISTORY_SIZE)
        svc = builder.open_or_create()
        self._pub = (
            svc.publisher_builder()
            .initial_max_slice_len(max_payload)
            .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo)
            .create()
        )

    def send(self, payload: bytes) -> None:
        n = len(payload)
        loan = self._pub.loan_slice_uninit(n)
        ctypes.memmove(loan.payload_ptr, payload, n)
        loan.assume_init().send()

    def close(self) -> None:
        self._pub = None


class IpcSubscriber(Subscriber):
    def __init__(
        self,
        name: str,
        *,
        latest_wins: bool = True,
        buffer_size: int = CTRL_HISTORY_SIZE,
    ) -> None:
        self._name = name
        builder = (
            _NodeSingleton.get()
            .service_builder(iox2.ServiceName.new(name))
            .publish_subscribe(iox2.Slice[ctypes.c_uint8])
            .enable_safe_overflow(latest_wins)
            .max_publishers(1)
            .max_subscribers(16)
        )
        if not latest_wins:
            builder = builder.history_size(buffer_size).subscriber_max_buffer_size(buffer_size)
        svc = builder.open_or_create()
        self._sub = svc.subscriber_builder().create()

    @staticmethod
    def _sample_to_bytes(sample) -> bytes:
        p = sample.payload()
        n = p.number_of_elements
        return bytes((ctypes.c_uint8 * n).from_address(p.data_ptr))

    def take_latest(self) -> Optional[bytes]:
        latest = None
        while True:
            s = self._sub.receive()
            if s is None:
                break
            latest = s
        return None if latest is None else self._sample_to_bytes(latest)

    def take_next(self) -> Optional[bytes]:
        s = self._sub.receive()
        return None if s is None else self._sample_to_bytes(s)

    def close(self) -> None:
        self._sub = None


class IpcNotifier(Notifier):
    def __init__(self, name: str) -> None:
        self._name = name
        svc = _NodeSingleton.get().service_builder(iox2.ServiceName.new(name)).event().open_or_create()
        self._notifier = svc.notifier_builder().create()

    def notify(self) -> None:
        self._notifier.notify()

    def close(self) -> None:
        self._notifier = None


class IpcListener(Listener):
    def __init__(self, name: str) -> None:
        self._name = name
        svc = _NodeSingleton.get().service_builder(iox2.ServiceName.new(name)).event().open_or_create()
        self._listener = svc.listener_builder().create()

    def _drain(self) -> int:
        """Consume any pending event triggers. Used after the WaitSet fires."""
        count = 0
        for _ in self._listener.try_wait_all():
            count += 1
        return count

    def close(self) -> None:
        self._listener = None


# ---------------------------------------------------------------------------
# Thread-local transport (intra-process-group, first-class)
# ---------------------------------------------------------------------------
#
# A `ThreadPublisher` and its single paired `ThreadSubscriber` are connected
# through a shared `_ThreadSlot`. Publishing serializes the bytes into the
# slot under a lock; the previous slot value is discarded — same latest-wins
# semantics as iceoryx2's `enable_safe_overflow(True)`.
#
# Atomic-instance guarantee: the slot stores `bytes` (immutable) — producer
# cannot mutate what the subscriber will see. For reliable ctrl/status
# channels we use a `deque` with bounded length instead.


class _ThreadSlot:
    """Single-slot, lock-protected store. Newest write wins."""

    __slots__ = ("_value", "_lock")

    def __init__(self) -> None:
        self._value: Optional[bytes] = None
        self._lock = threading.Lock()

    def put(self, value: bytes) -> None:
        with self._lock:
            self._value = value

    def take(self) -> Optional[bytes]:
        with self._lock:
            v = self._value
            self._value = None
            return v

    def peek(self) -> Optional[bytes]:
        with self._lock:
            return self._value


class _ThreadQueue:
    """Bounded FIFO. Oldest dropped on overflow."""

    __slots__ = ("_q", "_lock", "_maxlen")

    def __init__(self, maxlen: int) -> None:
        self._q: deque[bytes] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._maxlen = maxlen

    def put(self, value: bytes) -> None:
        with self._lock:
            self._q.append(value)

    def take_one(self) -> Optional[bytes]:
        with self._lock:
            return self._q.popleft() if self._q else None

    def take_latest(self) -> Optional[bytes]:
        with self._lock:
            if not self._q:
                return None
            last = self._q[-1]
            self._q.clear()
            return last


# Per-process registry mapping service name -> the local subscriber's
# storage + a shared "fired" event the publisher must set on send. Producers
# look up the registry to find their consumer.
_thread_registry: dict[str, "_ThreadChannel"] = {}
_thread_registry_lock = threading.Lock()


class _ThreadChannel:
    """Internal: bundles store + per-subscriber event used for wake-up."""

    __slots__ = ("store", "event", "latest_wins")

    def __init__(self, latest_wins: bool) -> None:
        self.latest_wins = latest_wins
        self.store = _ThreadSlot() if latest_wins else _ThreadQueue(CTRL_HISTORY_SIZE)
        # Each subscriber gets its own event; producers fire it on send.
        self.event = threading.Event()


def _thread_channel(name: str, *, latest_wins: bool) -> _ThreadChannel:
    """Look up or create the per-process registry entry for a channel.

    Both the publisher and the subscriber call this. Whichever calls first
    creates the channel; the other side reuses it. Safe under threading.
    """
    with _thread_registry_lock:
        ch = _thread_registry.get(name)
        if ch is None:
            ch = _ThreadChannel(latest_wins=latest_wins)
            _thread_registry[name] = ch
        elif ch.latest_wins != latest_wins:
            raise RuntimeError(
                f"Thread channel {name} requested with latest_wins={latest_wins} "
                f"but already exists with latest_wins={ch.latest_wins}"
            )
        return ch


def _drop_thread_channel(name: str) -> None:
    with _thread_registry_lock:
        _thread_registry.pop(name, None)


class ThreadPublisher(Publisher):
    def __init__(self, name: str, *, latest_wins: bool = True) -> None:
        self._name = name
        self._ch = _thread_channel(name, latest_wins=latest_wins)

    def send(self, payload: bytes) -> None:
        # `bytes` are immutable — atomic-instance guarantee. The producer
        # cannot reach into the consumer's view.
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError(f"ThreadPublisher.send expects bytes-like, got {type(payload).__name__}")
        b = bytes(payload) if not isinstance(payload, bytes) else payload
        self._ch.store.put(b)
        self._ch.event.set()

    def close(self) -> None:
        # The channel lives in the registry; the subscriber owns its
        # lifetime. Nothing to free here.
        pass


class ThreadSubscriber(Subscriber):
    def __init__(self, name: str, *, latest_wins: bool = True) -> None:
        self._name = name
        self._ch = _thread_channel(name, latest_wins=latest_wins)

    def take_latest(self) -> Optional[bytes]:
        if isinstance(self._ch.store, _ThreadSlot):
            v = self._ch.store.take()
        else:
            v = self._ch.store.take_latest()
        if v is None:
            # Spurious wake or already drained.
            self._ch.event.clear()
        return v

    def take_next(self) -> Optional[bytes]:
        if isinstance(self._ch.store, _ThreadSlot):
            return self._ch.store.take()
        return self._ch.store.take_one()

    def close(self) -> None:
        _drop_thread_channel(self._name)


class ThreadNotifier(Notifier):
    """The thread-local equivalent of `IpcNotifier`. Wakes a `ThreadListener`."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._ch = _thread_channel(name, latest_wins=True)
        # Listener gets its own dedicated event so that ThreadPublisher's
        # data-event and the explicit notify don't collide on semantics —
        # listeners exist for wake-up only.
        self._evt = self._ch.event

    def notify(self) -> None:
        self._evt.set()

    def close(self) -> None:
        pass


class ThreadListener(Listener):
    def __init__(self, name: str) -> None:
        self._name = name
        self._ch = _thread_channel(name, latest_wins=True)
        self._evt = self._ch.event

    def _consume(self) -> bool:
        if self._evt.is_set():
            self._evt.clear()
            return True
        return False

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Unified WaitSet
# ---------------------------------------------------------------------------


class WaitSet:
    """Wait for events from a mix of Ipc and Thread listeners.

    Three internal paths:
    - all-Ipc: native iceoryx2 WaitSet, sub-millisecond wake-up.
    - all-Thread: a shared `threading.Event` across the attached listeners.
    - mixed: short-poll loop alternating the two (rare; only when a node
      has both same-group and cross-group inputs).
    """

    def __init__(self) -> None:
        self._ipc_listeners: list[IpcListener] = []
        self._thread_listeners: list[ThreadListener] = []
        self._ipc_ws = None  # built lazily; invalidated when attach/detach
        self._ipc_guards: dict[int, tuple[object, IpcListener]] = {}
        # Shared event for thread listeners; each ThreadListener uses its own
        # per-channel event, but we OR-trigger by polling them in a loop.

    def attach(self, listener: Listener) -> None:
        if isinstance(listener, IpcListener):
            self._ipc_listeners.append(listener)
            self._ipc_ws = None  # rebuild on next wait
        elif isinstance(listener, ThreadListener):
            self._thread_listeners.append(listener)
        else:
            raise TypeError(f"WaitSet cannot attach {type(listener).__name__}")

    def detach(self, listener: Listener) -> None:
        if isinstance(listener, IpcListener) and listener in self._ipc_listeners:
            self._ipc_listeners.remove(listener)
            self._ipc_ws = None
        elif isinstance(listener, ThreadListener) and listener in self._thread_listeners:
            self._thread_listeners.remove(listener)

    def _build_ipc_ws(self) -> None:
        self._ipc_ws = iox2.WaitSetBuilder.new().create(iox2.ServiceType.Ipc)
        self._ipc_guards = {}
        for l in self._ipc_listeners:
            guard = self._ipc_ws.attach_notification(l._listener)
            self._ipc_guards[id(l)] = (guard, l)

    def _drain_thread(self) -> list[ThreadListener]:
        fired = [l for l in self._thread_listeners if l._consume()]
        return fired

    def _drain_ipc(self, timeout_s: float) -> list[IpcListener]:
        if not self._ipc_listeners:
            return []
        if self._ipc_ws is None:
            self._build_ipc_ws()
        ids, _ = self._ipc_ws.wait_and_process_with_timeout(iox2.Duration.from_secs_f64(max(timeout_s, 0.0)))
        fired: list[IpcListener] = []
        for aid in ids:
            for guard, l in self._ipc_guards.values():
                if aid.has_event_from(guard):
                    l._drain()
                    fired.append(l)
                    break
        return fired

    def wait(self, timeout_s: float) -> list[Listener]:
        """Block until any attached listener fires, or `timeout_s` elapses.

        Returns the list of listeners that fired (and were consumed). The
        caller is responsible for draining the corresponding subscribers.
        """
        has_ipc = bool(self._ipc_listeners)
        has_thread = bool(self._thread_listeners)

        if has_ipc and not has_thread:
            return list(self._drain_ipc(timeout_s))
        if has_thread and not has_ipc:
            return list(self._wait_thread(timeout_s))
        if has_ipc and has_thread:
            return list(self._wait_mixed(timeout_s))
        # Nothing attached — just sleep so the caller doesn't tight-loop.
        time.sleep(min(timeout_s, 0.01))
        return []

    def _wait_thread(self, timeout_s: float) -> list[ThreadListener]:
        # Each ThreadListener has its own threading.Event. Race-wait by
        # registering on any-set semantics through a polling loop with
        # short bursts. For typical pure-thread setups, the producer's
        # `set()` lands and the first iteration wakes us — no spin.
        deadline = time.monotonic() + timeout_s
        # First, optimistic fast path:
        fired = self._drain_thread()
        if fired:
            return fired
        # Block on any single event; if none fires, fall through to poll.
        # Use a small thread to OR them together. For ≤4 listeners this is
        # cheap; bigger fan-in still works.
        if len(self._thread_listeners) == 1:
            l = self._thread_listeners[0]
            if l._evt.wait(timeout_s):
                l._consume()
                return [l]
            return []
        # Multiple thread listeners — poll loop. Reasonable because typical
        # nodes have very few inputs.
        poll = 0.001
        while True:
            fired = self._drain_thread()
            if fired:
                return fired
            now = time.monotonic()
            if now >= deadline:
                return []
            time.sleep(min(poll, deadline - now))

    def _wait_mixed(self, timeout_s: float) -> list[Listener]:
        deadline = time.monotonic() + timeout_s
        poll = 0.002
        while True:
            fired_thread = self._drain_thread()
            now = time.monotonic()
            slice_to = min(poll, max(deadline - now, 0.0))
            fired_ipc = self._drain_ipc(slice_to) if slice_to > 0 else []
            if fired_thread or fired_ipc:
                # Also drain the other side opportunistically with zero timeout.
                if not fired_thread:
                    fired_thread = self._drain_thread()
                if not fired_ipc:
                    fired_ipc = self._drain_ipc(0.0)
                return list(fired_thread) + list(fired_ipc)
            if time.monotonic() >= deadline:
                return []


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------


def create_data_publisher(name: str, *, in_process: bool, max_payload: int = DEFAULT_MAX_PAYLOAD) -> tuple[Publisher, Notifier]:
    """Create the publisher + notifier pair for a data-plane channel."""
    if in_process:
        return ThreadPublisher(name, latest_wins=True), ThreadNotifier(event_service_name(name))
    return (
        IpcPublisher(name, max_payload=max_payload, latest_wins=True),
        IpcNotifier(event_service_name(name)),
    )


def create_data_subscriber(name: str, *, in_process: bool) -> tuple[Subscriber, Listener]:
    """Create the subscriber + listener pair for a data-plane channel."""
    if in_process:
        return ThreadSubscriber(name, latest_wins=True), ThreadListener(event_service_name(name))
    return (
        IpcSubscriber(name, latest_wins=True),
        IpcListener(event_service_name(name)),
    )


def create_ctrl_publisher(name: str, *, in_process: bool) -> tuple[Publisher, Notifier]:
    if in_process:
        return ThreadPublisher(name, latest_wins=False), ThreadNotifier(event_service_name(name))
    return (
        IpcPublisher(name, latest_wins=False),
        IpcNotifier(event_service_name(name)),
    )


def create_ctrl_subscriber(name: str, *, in_process: bool) -> tuple[Subscriber, Listener]:
    if in_process:
        return ThreadSubscriber(name, latest_wins=False), ThreadListener(event_service_name(name))
    return (
        IpcSubscriber(name, latest_wins=False),
        IpcListener(event_service_name(name)),
    )
