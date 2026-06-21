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


def self_trigger_service_name(node_name: str) -> str:
    """Per-node in-process wake channel: any thread of the node can ping
    its own processing loop by publishing on this service."""
    return f"goofi.{get_instance_id()}.self.{node_name}"


# ---------------------------------------------------------------------------
# iceoryx2 node singleton (per OS process; rebuilt after fork)
# ---------------------------------------------------------------------------


def ensure_iox2_runtime_dirs() -> None:
    """Ensure iceoryx2's configured runtime directory layout exists.

    iceoryx2 keeps its node/service bookkeeping under ``<root>/nodes`` and
    ``<root>/services`` (``<root>`` is ``/tmp/iceoryx2`` on POSIX,
    ``C:\\Temp\\iceoryx2`` on Windows). Pre-creating them is an idempotent
    initialisation step: where iceoryx2 creates them lazily this is a harmless
    no-op; where it doesn't, it's required — so no platform branch is needed.

    The latter is the case on Windows (iceoryx2 0.9.1): node creation scans
    these directories through the POSIX ``opendir`` shim, whose
    ``FindFirstFileA`` errors on a missing directory instead of treating it as
    empty the way POSIX does — surfacing as ``NodeCreationFailure:
    InternalError`` on the first node of a fresh session. Creating the (empty)
    directories first lets the scan succeed; iceoryx2 still owns everything
    inside them.

    Paths come from iceoryx2's own config so they stay correct if the root is
    reconfigured.
    """
    try:
        g = iox2.config.global_config().global_cfg
        for path in (g.node_dir.to_string(), g.service_dir.to_string()):
            os.makedirs(path, exist_ok=True)
    except Exception:
        # Best-effort: a future iceoryx2 may reshape this config surface. Fall
        # back to letting iceoryx2 create/fail on its own rather than masking an
        # unrelated error here.
        pass


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


class _IpcLoan:
    """Writable, single-use loan of an iceoryx2 SHM slice.

    The producer fills `.buffer` (a memoryview backed by shared memory),
    then calls `.send()` to publish. Each call to `IpcPublisher.loan`
    returns memory from iceoryx2's pool — subsequent loans yield a
    different region, so previously-published samples remain valid for
    consumers (no producer mutation can leak across publications).
    """

    __slots__ = ("_loan", "_size", "_buffer", "_sent")

    def __init__(self, loan, size: int) -> None:
        self._loan = loan
        self._size = size
        self._buffer: Optional[memoryview] = None
        self._sent = False

    @property
    def buffer(self) -> memoryview:
        if self._sent or self._loan is None:
            raise RuntimeError("Loan already sent or released")
        if self._buffer is None:
            arr = (ctypes.c_uint8 * self._size).from_address(self._loan.payload_ptr)
            # `.cast("B")` normalises the format to plain unsigned-byte so
            # downstream `view[a:b] = src_bytes` slice assignments work
            # (ctypes' memoryview format defaults to "<B", which Python's
            # slice-assignment rejects with NotImplementedError).
            self._buffer = memoryview(arr).cast("B")
        return self._buffer

    def send(self) -> None:
        if self._sent:
            raise RuntimeError("Loan already sent")
        # Release the memoryview before iceoryx2 consumes the loan so the
        # underlying ctypes buffer is no longer referenced from Python.
        if self._buffer is not None:
            self._buffer.release()
            self._buffer = None
        self._loan.assume_init().send()
        self._loan = None
        self._sent = True


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

    def loan(self, size: int) -> _IpcLoan:
        """Allocate `size` bytes of iceoryx2 shared memory for the next publish."""
        return _IpcLoan(self._pub.loan_slice_uninit(size), size)

    def send(self, payload: bytes) -> None:
        """Compatibility path: copy `payload` into a fresh loan and commit."""
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
        # Capture once: the data-pump can race with `close()` and clear
        # `self._sub` between this and the `receive` call. Treat a closed
        # subscriber as having no pending samples.
        sub = self._sub
        if sub is None:
            return None
        latest = None
        while True:
            s = sub.receive()
            if s is None:
                break
            latest = s
        return None if latest is None else self._bytes(latest)

    def take_next(self) -> Optional[bytes]:
        sub = self._sub
        if sub is None:
            return None
        s = sub.receive()
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


class _ThreadLoan:
    """Writable, single-use loan for thread transport.

    Wraps a fresh `bytearray` of the requested size. On `.send()`, the
    bytearray is stored directly into the channel (zero extra copy);
    the producer's reference is cleared so it cannot mutate the buffer
    after publication. The consumer receives the bytearray as-is.
    """

    __slots__ = ("_channel", "_ba", "_buffer", "_sent")

    def __init__(self, channel: "_ThreadChannel", size: int) -> None:
        self._channel = channel
        self._ba = bytearray(size)
        self._buffer: Optional[memoryview] = None
        self._sent = False

    @property
    def buffer(self) -> memoryview:
        if self._sent or self._ba is None:
            raise RuntimeError("Loan already sent or released")
        if self._buffer is None:
            self._buffer = memoryview(self._ba)
        return self._buffer

    def send(self) -> None:
        if self._sent:
            raise RuntimeError("Loan already sent")
        if self._buffer is not None:
            self._buffer.release()
            self._buffer = None
        ba, self._ba = self._ba, None
        # `bytes(bytearray)` for the store ensures the consumer sees an
        # immutable view that survives even hypothetical re-acquisition
        # of the bytearray by the producer. The conversion is one memcpy
        # but at this point the array is small (rare large payloads use
        # the iceoryx2 path).
        self._channel.store.put(bytes(ba))
        self._channel.event.set()
        self._sent = True


class ThreadPublisher:
    def __init__(self, name: str, *, latest_wins: bool = True) -> None:
        self._ch = _thread_channel(name, latest_wins=latest_wins)

    def loan(self, size: int) -> _ThreadLoan:
        return _ThreadLoan(self._ch, size)

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
        # Guards all reads/writes of the listener lists + lazy ws/guards. The
        # processing (or data-pump) thread waits while another thread wires
        # links / (un)subscribes expression slots; without this, the waiter's
        # snapshot races the mutation and can drop a wakeup (reports H1/B3).
        # Released before every blocking wait so (un)subscribe never stalls.
        self._lock = threading.Lock()

    def attach(self, listener: Listener) -> None:
        with self._lock:
            if isinstance(listener, IpcListener):
                self._ipc_listeners.append(listener)
                self._ipc_ws = None
            elif isinstance(listener, ThreadListener):
                self._thread_listeners.append(listener)
            else:
                raise TypeError(f"WaitSet cannot attach {type(listener).__name__}")

    def detach(self, listener: Listener) -> None:
        with self._lock:
            if isinstance(listener, IpcListener) and listener in self._ipc_listeners:
                self._ipc_listeners.remove(listener)
                self._ipc_ws = None
            elif isinstance(listener, ThreadListener) and listener in self._thread_listeners:
                self._thread_listeners.remove(listener)

    def wait(self, timeout_s: float) -> list[Listener]:
        """Block until any attached listener fires; return the ones that did."""
        with self._lock:
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
        # Snapshot the built ws + guards under the lock, then run the blocking
        # iox2 wait unlocked so a concurrent attach/detach never stalls.
        with self._lock:
            if not self._ipc_listeners:
                return []
            if self._ipc_ws is None:
                self._build_ipc_ws()
            ws = self._ipc_ws
            guards = list(self._ipc_guards.values())
        ids, _ = ws.wait_and_process_with_timeout(iox2.Duration.from_secs_f64(max(timeout_s, 0.0)))
        fired: list[IpcListener] = []
        for aid in ids:
            for guard, l in guards:
                if aid.has_event_from(guard):
                    l._drain()
                    fired.append(l)
                    break
        return fired

    def _drain_thread(self) -> list[ThreadListener]:
        with self._lock:
            return [l for l in self._thread_listeners if l._consume()]

    def _wait_thread(self, timeout_s: float) -> list[ThreadListener]:
        fired = self._drain_thread()
        if fired:
            return fired
        with self._lock:
            single = self._thread_listeners[0] if len(self._thread_listeners) == 1 else None
        if single is not None:
            if single._evt.wait(timeout_s):
                single._consume()
                return [single]
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
