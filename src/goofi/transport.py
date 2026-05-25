"""Unified transport for goofi-pipe.

Two endpoint flavours behind one API:

- **`Ipc*`** — iceoryx2 publish/subscribe + sibling event service. Used for
  every channel that crosses a process boundary. Latest-wins on the data
  plane, reliable/buffered on the ctrl/status planes.
- **`Thread*`** — locked single-slot or bounded deque + `threading.Event`.
  Used inside a process group; producers serialize through the same codec
  path as iceoryx2 so atomic-instance + latest-wins semantics match.

All endpoints carry `bytes` payloads; encoding/decoding belongs to the
caller (see `codec.py`). The two factories `open_publisher` and
`open_subscriber` are the only constructors callers should reach for.
"""
from __future__ import annotations

import ctypes
import os
import threading
import time
from collections import deque
from typing import ClassVar, Deque, Optional, Tuple, Union

import iceoryx2 as iox2


# Starting slice allocation. iceoryx2 grows the slice on demand via
# AllocationStrategy.PowerOfTwo — this is just the initial size.
DEFAULT_MAX_PAYLOAD = 64 * 1024

# History depth for the reliable ctrl/status planes.
CTRL_HISTORY_SIZE = 64

# Iceoryx2 publishers cap subscriber count at service-build time.
DEFAULT_MAX_SUBSCRIBERS = 16

# Suffix for the sibling event service that wakes a paired listener.
_EVT = ".evt"


# ---------------------------------------------------------------------------
# Instance id (embedded in every service name to keep concurrent goofi-pipe
# instances on one host from colliding in /dev/shm).
# ---------------------------------------------------------------------------

_instance_id: str = ""


def set_instance_id(iid: str) -> None:
    global _instance_id
    _instance_id = iid


def get_instance_id() -> str:
    return _instance_id or str(os.getpid())


def data_service_name(src_node: str, slot_name: str) -> str:
    return f"goofi.{get_instance_id()}.data.{src_node}.{slot_name}"


def ctrl_service_name(node_name: str) -> str:
    return f"goofi.{get_instance_id()}.ctrl.{node_name}"


def status_service_name(node_name: str) -> str:
    return f"goofi.{get_instance_id()}.status.{node_name}"


# ---------------------------------------------------------------------------
# iceoryx2 node singleton (per OS process; rebuilt after fork)
# ---------------------------------------------------------------------------


class _NodeSingleton:
    _node: ClassVar[object] = None
    _pid: ClassVar[int] = -1
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get(cls):
        if cls._node is None or cls._pid != os.getpid():
            with cls._lock:
                if cls._node is None or cls._pid != os.getpid():
                    iox2.set_log_level(iox2.LogLevel.Error)
                    cls._node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
                    cls._pid = os.getpid()
        return cls._node


def _open_byte_service(name: str, *, latest_wins: bool):
    """Open or create a `publish_subscribe` byte-slice service.

    `latest_wins=True` configures overflow-safe (data plane); `False` gives
    bounded, in-order delivery (ctrl / status plane).
    """
    builder = (
        _NodeSingleton.get()
        .service_builder(iox2.ServiceName.new(name))
        .publish_subscribe(iox2.Slice[ctypes.c_uint8])
        .enable_safe_overflow(latest_wins)
        .max_publishers(1)
        .max_subscribers(DEFAULT_MAX_SUBSCRIBERS)
    )
    if not latest_wins:
        builder = builder.history_size(CTRL_HISTORY_SIZE).subscriber_max_buffer_size(CTRL_HISTORY_SIZE)
    return builder.open_or_create()


# ---------------------------------------------------------------------------
# iceoryx2 endpoints
# ---------------------------------------------------------------------------


class IpcPublisher:
    def __init__(self, name: str, *, latest_wins: bool = True, max_payload: int = DEFAULT_MAX_PAYLOAD) -> None:
        self._name = name
        svc = _open_byte_service(name, latest_wins=latest_wins)
        self._pub = (
            svc.publisher_builder()
            .initial_max_slice_len(max_payload)
            .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo)
            .create()
        )

    def send(self, payload: bytes) -> None:
        loan = self._pub.loan_slice_uninit(len(payload))
        ctypes.memmove(loan.payload_ptr, payload, len(payload))
        loan.assume_init().send()

    def close(self) -> None:
        self._pub = None


class IpcSubscriber:
    def __init__(self, name: str, *, latest_wins: bool = True) -> None:
        self._name = name
        svc = _open_byte_service(name, latest_wins=latest_wins)
        self._sub = svc.subscriber_builder().create()

    @staticmethod
    def _bytes(sample) -> bytes:
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
        return None if latest is None else self._bytes(latest)

    def take_next(self) -> Optional[bytes]:
        s = self._sub.receive()
        return None if s is None else self._bytes(s)

    def close(self) -> None:
        self._sub = None


class IpcNotifier:
    def __init__(self, name: str) -> None:
        svc = _NodeSingleton.get().service_builder(iox2.ServiceName.new(name)).event().open_or_create()
        self._notifier = svc.notifier_builder().create()

    def notify(self) -> None:
        self._notifier.notify()

    def close(self) -> None:
        self._notifier = None


class IpcListener:
    def __init__(self, name: str) -> None:
        svc = _NodeSingleton.get().service_builder(iox2.ServiceName.new(name)).event().open_or_create()
        self._listener = svc.listener_builder().create()

    def _drain(self) -> None:
        for _ in self._listener.try_wait_all():
            pass

    def close(self) -> None:
        self._listener = None


# ---------------------------------------------------------------------------
# Thread-local endpoints
# ---------------------------------------------------------------------------


class _ThreadStore:
    """Lock-protected backing store for a `Thread*` endpoint pair.

    Two modes: `latest_wins=True` keeps only the newest payload (data
    plane); `latest_wins=False` keeps a bounded FIFO (ctrl/status plane).
    """

    __slots__ = ("_lock", "_latest_wins", "_value", "_queue")

    def __init__(self, latest_wins: bool, maxlen: int = CTRL_HISTORY_SIZE) -> None:
        self._lock = threading.Lock()
        self._latest_wins = latest_wins
        self._value: Optional[bytes] = None
        self._queue: Optional[Deque[bytes]] = None if latest_wins else deque(maxlen=maxlen)

    def put(self, value: bytes) -> None:
        with self._lock:
            if self._latest_wins:
                self._value = value
            else:
                self._queue.append(value)

    def take_one(self) -> Optional[bytes]:
        with self._lock:
            if self._latest_wins:
                v, self._value = self._value, None
                return v
            return self._queue.popleft() if self._queue else None

    def take_latest(self) -> Optional[bytes]:
        with self._lock:
            if self._latest_wins:
                v, self._value = self._value, None
                return v
            if not self._queue:
                return None
            last = self._queue[-1]
            self._queue.clear()
            return last


class _ThreadChannel:
    """Pairs a `_ThreadStore` with a `threading.Event` for wake-up."""

    __slots__ = ("store", "event", "latest_wins")

    def __init__(self, latest_wins: bool) -> None:
        self.latest_wins = latest_wins
        self.store = _ThreadStore(latest_wins)
        self.event = threading.Event()


_thread_registry: dict[str, _ThreadChannel] = {}
_thread_registry_lock = threading.Lock()


def _thread_channel(name: str, *, latest_wins: bool) -> _ThreadChannel:
    """Look up or create the per-process registry entry for a channel."""
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


class ThreadPublisher:
    def __init__(self, name: str, *, latest_wins: bool = True) -> None:
        self._ch = _thread_channel(name, latest_wins=latest_wins)

    def send(self, payload: bytes) -> None:
        # `bytes` are immutable — no shared-object risk with the consumer.
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError(f"ThreadPublisher.send expects bytes-like, got {type(payload).__name__}")
        self._ch.store.put(payload if isinstance(payload, bytes) else bytes(payload))
        self._ch.event.set()

    def close(self) -> None:
        pass


class ThreadSubscriber:
    def __init__(self, name: str, *, latest_wins: bool = True) -> None:
        self._name = name
        self._ch = _thread_channel(name, latest_wins=latest_wins)

    def take_latest(self) -> Optional[bytes]:
        v = self._ch.store.take_latest()
        if v is None:
            self._ch.event.clear()
        return v

    def take_next(self) -> Optional[bytes]:
        return self._ch.store.take_one()

    def close(self) -> None:
        with _thread_registry_lock:
            _thread_registry.pop(self._name, None)


class ThreadNotifier:
    def __init__(self, name: str) -> None:
        self._evt = _thread_channel(name, latest_wins=True).event

    def notify(self) -> None:
        self._evt.set()

    def close(self) -> None:
        pass


class ThreadListener:
    def __init__(self, name: str) -> None:
        self._evt = _thread_channel(name, latest_wins=True).event

    def _consume(self) -> bool:
        if self._evt.is_set():
            self._evt.clear()
            return True
        return False

    def close(self) -> None:
        pass


# Type aliases — concrete classes are the runtime types; these document
# intent at call sites without an ABC layer.
Publisher = Union[IpcPublisher, ThreadPublisher]
Subscriber = Union[IpcSubscriber, ThreadSubscriber]
Notifier = Union[IpcNotifier, ThreadNotifier]
Listener = Union[IpcListener, ThreadListener]


# ---------------------------------------------------------------------------
# Unified WaitSet
# ---------------------------------------------------------------------------


class WaitSet:
    """Wait for events from a mix of Ipc and Thread listeners.

    Three internal paths:
    - all-Ipc: native iceoryx2 WaitSet, sub-millisecond wake-up.
    - all-Thread: blocks on the per-listener `threading.Event`.
    - mixed: short-poll loop alternating the two (rare; only when a node
      has both same-group and cross-group inputs).
    """

    def __init__(self) -> None:
        self._ipc_listeners: list[IpcListener] = []
        self._thread_listeners: list[ThreadListener] = []
        self._ipc_ws = None  # built lazily, invalidated on attach/detach
        self._ipc_guards: dict[int, tuple[object, IpcListener]] = {}

    def attach(self, listener: Listener) -> None:
        if isinstance(listener, IpcListener):
            self._ipc_listeners.append(listener)
            self._ipc_ws = None
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

    def wait(self, timeout_s: float) -> list[Listener]:
        """Block until any attached listener fires; return the ones that did."""
        has_ipc = bool(self._ipc_listeners)
        has_thread = bool(self._thread_listeners)
        if has_ipc and not has_thread:
            return list(self._drain_ipc(timeout_s))
        if has_thread and not has_ipc:
            return list(self._wait_thread(timeout_s))
        if has_ipc and has_thread:
            return list(self._wait_mixed(timeout_s))
        time.sleep(min(timeout_s, 0.01))
        return []

    def _build_ipc_ws(self) -> None:
        # `SignalHandlingMode.Disabled` is critical: iceoryx2's default
        # `HandleTerminationRequests` installs a C-level SIGINT handler that
        # overrides Python's, so KeyboardInterrupt never propagates and the
        # process becomes Ctrl+C-proof. Disabling here lets Python keep
        # its signal handling and exit cleanly on Ctrl+C.
        self._ipc_ws = (
            iox2.WaitSetBuilder.new()
            .signal_handling_mode(iox2.SignalHandlingMode.Disabled)
            .create(iox2.ServiceType.Ipc)
        )
        self._ipc_guards = {}
        for l in self._ipc_listeners:
            guard = self._ipc_ws.attach_notification(l._listener)
            self._ipc_guards[id(l)] = (guard, l)

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

    def _drain_thread(self) -> list[ThreadListener]:
        return [l for l in self._thread_listeners if l._consume()]

    def _wait_thread(self, timeout_s: float) -> list[ThreadListener]:
        fired = self._drain_thread()
        if fired:
            return fired
        if len(self._thread_listeners) == 1:
            l = self._thread_listeners[0]
            if l._evt.wait(timeout_s):
                l._consume()
                return [l]
            return []
        # >1 thread listeners: poll loop (rare; nodes typically have few inputs).
        deadline = time.monotonic() + timeout_s
        while True:
            fired = self._drain_thread()
            if fired:
                return fired
            now = time.monotonic()
            if now >= deadline:
                return []
            time.sleep(min(0.001, deadline - now))

    def _wait_mixed(self, timeout_s: float) -> list[Listener]:
        deadline = time.monotonic() + timeout_s
        poll = 0.002
        while True:
            fired_thread = self._drain_thread()
            now = time.monotonic()
            slice_to = min(poll, max(deadline - now, 0.0))
            fired_ipc = self._drain_ipc(slice_to) if slice_to > 0 else []
            if fired_thread or fired_ipc:
                if not fired_thread:
                    fired_thread = self._drain_thread()
                if not fired_ipc:
                    fired_ipc = self._drain_ipc(0.0)
                return list(fired_thread) + list(fired_ipc)
            if time.monotonic() >= deadline:
                return []


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------


def open_publisher(name: str, *, in_process: bool, latest_wins: bool = True) -> Tuple[Publisher, Notifier]:
    """Build the producer side of a channel — pub + paired notifier."""
    if in_process:
        return ThreadPublisher(name, latest_wins=latest_wins), ThreadNotifier(name + _EVT)
    return IpcPublisher(name, latest_wins=latest_wins), IpcNotifier(name + _EVT)


def open_subscriber(name: str, *, in_process: bool, latest_wins: bool = True) -> Tuple[Subscriber, Listener]:
    """Build the consumer side of a channel — sub + paired listener."""
    if in_process:
        return ThreadSubscriber(name, latest_wins=latest_wins), ThreadListener(name + _EVT)
    return IpcSubscriber(name, latest_wins=latest_wins), IpcListener(name + _EVT)
