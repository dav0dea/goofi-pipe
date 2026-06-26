"""Node helpers: slot dataclasses, NodeRef, process-group registry.

`InputSlot` and `OutputSlot` are pure metadata containers; transport
endpoints live on the `Node` itself, keyed by slot name. Slots are pickled
across process spawn and must not hold iceoryx2 references — those are
reconstructed in the child process from canonical service names.

`NodeRef` is the manager's proxy for a node. It owns a `ctrl_pub` to push
control messages to the node and a `status_sub` + `Listener` pair to receive
status updates. State (`serialized_state`) is pushed by the node on every
dirty cycle; the manager's `save()` reads it directly without round-trips.
"""
from __future__ import annotations

import functools
import importlib
import inspect
import os
import pkgutil
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing import Pipe, Process
from multiprocessing.connection import _ConnectionBase
from threading import Event, Lock, RLock, Thread
from typing import Any, Callable, Dict, List, Optional, Type

from goofi import nodes as goofi_nodes
from goofi.codec import decode_message, encode_message
from goofi.data import Data, DataType
from goofi.message import Message, MessageType
from goofi.params import NodeParams, normalize_expression_binding
from goofi.transport import (
    Listener,
    Publisher,
    Subscriber,
    WaitSet,
    ctrl_service_name,
    data_service_name,
    open_publisher,
    open_subscriber,
    status_service_name,
    view_service_name,
)


class NodeLoadingError(Exception):
    """Custom exception for node loading errors."""


@functools.lru_cache(maxsize=1)
def list_nodes(verbose: bool = False) -> List[Type]:
    """
    Gather a list of all available nodes in the goofi.nodes module.
    """
    first_module = True

    def _list_nodes_recursive(nodes=None, parent_module=goofi_nodes):
        from goofi.node import Node

        nonlocal first_module

        if nodes is None:
            nodes = []

        for info in pkgutil.walk_packages(parent_module.__path__):
            if "goofi" + os.sep + "nodes" not in info.module_finder.path:
                continue
            if info.name.startswith("_"):
                continue

            start = time.time()
            try:
                module = importlib.import_module(f"{parent_module.__name__}.{info.name}")
            except ModuleNotFoundError as e:
                print()
                raise NodeLoadingError(
                    f"Missing dependency for node '{info.name}' -> {e}; make sure to install the required dependencies."
                ) from e
            except Exception as e:
                print()
                raise NodeLoadingError(f"Failed to load node '{info.name}' -> {e}") from e

            module_annot = "" if time.time() - start < 0.2 else "!!!"

            if verbose:
                parts = module.__name__.split(".")
                if len(parts) == 3:
                    if first_module:
                        first_module = False
                    else:
                        print()
                    print(f"- {parts[2]}:", end="")

            if info.ispkg:
                _list_nodes_recursive(nodes, module)
                continue

            members = inspect.getmembers(module, inspect.isclass)
            new_nodes = [cls for _, cls in members if issubclass(cls, Node) and cls is not Node]

            if len(new_nodes) != 1:
                raise ValueError(f"Expected exactly one node in module {module.__name__}, got {len(new_nodes)}")

            if verbose:
                print(f"  {module_annot}{new_nodes[0].__name__}{module_annot}", end="")

            nodes.extend(new_nodes)
        return nodes

    if verbose:
        print("Discovering goofi-pipe nodes...")
    res = _list_nodes_recursive()
    if verbose:
        print("\n")
    return res


# ---------------------------------------------------------------------------
# Slot dataclasses
# ---------------------------------------------------------------------------


@dataclass
class InputSlot:
    """Metadata + runtime state for one of a node's input slots.

    Carries the latest decoded `Data` value the node has seen on this slot.
    Transport endpoints (`subscriber`, `listener`) are populated by the node
    when a `SUBSCRIBE_INPUT` control message wires this slot to an upstream
    publisher, and torn down on `UNSUBSCRIBE_INPUT`.
    """

    dtype: DataType
    trigger_process: bool = True
    data: Optional[Data] = None

    # Runtime endpoints — populated by the node when SUBSCRIBE_INPUT wires
    # this slot, cleared on UNSUBSCRIBE_INPUT. They're always None at the
    # time a slot is constructed (via `cls._configure()`) and at the time
    # slots are pickled across process spawn, so no special pickle handling
    # is needed.
    subscriber: Optional[Subscriber] = field(default=None, repr=False, compare=False)
    listener: Optional[Listener] = field(default=None, repr=False, compare=False)
    service_name: Optional[str] = field(default=None, repr=False, compare=False)
    in_process: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self):
        if isinstance(self.dtype, str):
            self.dtype = DataType[self.dtype]

    def clear(self):
        self.data = None


@dataclass
class OutputSlot:
    """Metadata + runtime state for one of a node's output slots.

    `subscriber_count` is incremented by `REGISTER_SUBSCRIBER` control
    messages and decremented by `UNREGISTER_SUBSCRIBER`. The node skips
    publishing on a slot with zero registered subscribers to avoid wasting
    encode cycles on data nobody is reading.

    Each output slot may carry multiple publishers when the slot fans out
    to a mix of in-process consumers (thread transport) and out-of-process
    consumers (iceoryx2 transport) — the node creates whichever flavour is
    needed at the time the first subscriber of that flavour registers.
    """

    dtype: DataType
    subscriber_count: int = 0

    # Browser-viewer count (Option C node-side reduction). Written ONLY by the
    # messaging thread on REGISTER_VIEWER/UNREGISTER_VIEWER (under viewer_lock);
    # read ONLY by the processing loop's gate. A plain int keeps OutputSlot
    # picklable for MP spawn; it is 0 at pickle time because slots are pickled
    # via `cls._configure()` before any viewer wiring, so the lazy viewer_lock
    # (below) is likewise never present at pickle time.
    viewer_count: int = field(default=0, repr=False, compare=False)

    # Runtime, set by Node — `publishers` and `notifiers` are parallel
    # lists. Always empty at pickle time (slots are pickled via
    # `cls._configure()` before any wiring), so no pickle hooks needed.
    publishers: List[Publisher] = field(default_factory=list, repr=False, compare=False)
    notifiers: List[object] = field(default_factory=list, repr=False, compare=False)
    has_ipc: bool = field(default=False, repr=False, compare=False)
    has_thread: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self):
        if isinstance(self.dtype, str):
            self.dtype = DataType[self.dtype]

    # Lazy lock — NOT a dataclass field, so it never participates in
    # pickle/equality. Created on first use (always from the single messaging
    # thread on the first REGISTER_VIEWER for this slot, before any concurrent
    # writer can exist); later reads see the already-set `__dict__` entry.
    @property
    def viewer_lock(self) -> "Lock":
        lk = self.__dict__.get("_viewer_lock")
        if lk is None:
            lk = self.__dict__["_viewer_lock"] = Lock()
        return lk


# ---------------------------------------------------------------------------
# NodeRef — the manager-side proxy for a node
# ---------------------------------------------------------------------------


@dataclass
class NodeRef:
    """The manager's proxy for one node.

    Owns the manager-side endpoints of the ctrl and status channels:
    - `ctrl_pub` pushes control messages to the node (params, link wiring,
      terminate).
    - `status_sub` receives status updates from the node (state pushes on
      dirty, processing errors, shutdown requests).

    `serialized_state` is the most recent state snapshot pushed by the node
    via `STATE_UPDATE`. It updates automatically; `Manager.save()` reads it
    without round-trips.

    `node_id` is the canonical name used to derive iceoryx2 service names —
    must be unique per goofi-pipe instance.
    """

    node_id: str
    in_process: bool
    input_slots: Dict[str, DataType]
    output_slots: Dict[str, DataType]
    params: NodeParams
    node_class: Type

    category: str = None
    process: Optional[Process] = None
    callbacks: Dict[MessageType, Callable] = field(default_factory=dict)
    serialized_state: Optional[Dict[str, Any]] = None
    gui_kwargs: Dict[str, Any] = field(default_factory=dict)
    create_initialized: bool = True

    # Runtime fields populated in __post_init__ — not part of the dataclass
    # equality / repr surface.
    ctrl_pub: Optional[Publisher] = field(default=None, repr=False, compare=False)
    status_sub: Optional[Subscriber] = field(default=None, repr=False, compare=False)
    status_listener: Optional[Listener] = field(default=None, repr=False, compare=False)
    _alive: bool = field(default=False, repr=False, compare=False)
    _messaging_thread: Optional[Thread] = field(default=None, repr=False, compare=False)
    _waitset: Optional[WaitSet] = field(default=None, repr=False, compare=False)
    last_error: Optional[str] = field(default=None, repr=False, compare=False)
    # Latest execution telemetry the node pushed (NODE_STATS). Node-authoritative
    # cache like serialized_state / last_error; surfaced to the browser.
    node_stats: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)
    # The UNIVERSAL node identity: a uuid minted once by the manager
    # (Manager.add_node), persisted in the .gfi, carried across respawns/reloads,
    # and the key EVERYTHING references the node by (container, links, membership,
    # data plane, frontend). Identity bookkeeping, so excluded from dataclass equality.
    uid: Optional[str] = field(default=None, repr=False, compare=False)
    # Mutable DISPLAY name (e.g. "oscillator0", or a qualified "subpatch0::filter").
    # For display + `nd()` resolution ONLY — never a reference key, so it can be
    # renamed freely without touching anything that points at the node.
    name: Optional[str] = field(default=None, repr=False, compare=False)
    membership: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)
    # How many times the manager's liveness supervisor has respawned this node
    # after its process crashed. Carried across respawns; surfaced to the UI so a
    # crash-loop is observable even though restarts are unlimited.
    restart_count: int = field(default=0, repr=False, compare=False)

    def __post_init__(self):
        self.__doc__ = self.node_class.docstring()
        self.category = self.node_class.category()

        # Events / threads / handler bookkeeping. Init here (not as dataclass
        # fields) so dataclass equality doesn't recurse into thread state.
        self._first_state_event = Event()
        self._data_handlers: Dict[str, tuple] = {}
        self._data_handlers_lock = RLock()
        self._data_waitset = WaitSet()
        self._data_waitset_dirty = Event()
        self._data_pump_thread: Optional[Thread] = None

        self.ctrl_pub, self._ctrl_notifier = open_publisher(
            ctrl_service_name(self.node_id), in_process=self.in_process, latest_wins=False
        )
        self.status_sub, self.status_listener = open_subscriber(
            status_service_name(self.node_id), in_process=self.in_process, latest_wins=False
        )

        if self.create_initialized:
            self.initialize()

    @property
    def serialization_pending(self) -> bool:
        """True until the node has pushed its first STATE_UPDATE."""
        return not self._first_state_event.is_set()

    def initialize(self) -> None:
        if self._alive:
            return
        self._alive = True
        self._waitset = WaitSet()
        self._waitset.attach(self.status_listener)
        self._messaging_thread = Thread(
            target=self._messaging_loop,
            name=f"NodeRef-{self.node_id}-msg",
            daemon=True,
        )
        self._messaging_thread.start()

    def set_message_handler(self, msg_type: MessageType, callback: Optional[Callable]) -> None:
        """Override default handling for a specific message type."""
        if callback is None:
            self.callbacks.pop(msg_type, None)
        else:
            self.callbacks[msg_type] = callback

    def _send(self, msg: Message) -> None:
        if self.ctrl_pub is None:
            return
        self.ctrl_pub.send(encode_message(msg))
        self._ctrl_notifier.notify()

    def update_param(self, group: str, param_name: str, param_value: Any) -> None:
        if group not in self.params:
            raise ValueError(f"Parameter group '{group}' doesn't exist.")
        if param_name not in self.params[group]:
            raise ValueError(f"Parameter '{param_name}' doesn't exist in group '{group}'.")
        # Reflect locally so the manager has up-to-date params even before
        # the node pushes a STATE_UPDATE back.
        self.params[group][param_name].value = param_value
        self._send(
            Message(
                MessageType.PARAMETER_UPDATE,
                {"group": group, "param_name": param_name, "param_value": param_value},
            )
        )

    def set_expression(
        self,
        group: str,
        param_name: str,
        expression: Optional[str],
        enabled: bool = False,
        triggers_process: bool = False,
        autoeval: bool = False,
    ) -> None:
        """Bind a Python expression to a parameter (or stash one inactive).

        ``expression`` is the source — kept on the Param even when
        ``enabled`` is False so the user can toggle the fx button off and
        back on without losing what they wrote.
        ``triggers_process`` makes a re-eval that changes the param's
        value wake `process()`. ``autoeval`` makes the engine refresh
        before every process tick.
        """
        if group not in self.params:
            raise ValueError(f"Parameter group '{group}' doesn't exist.")
        if param_name not in self.params[group]:
            raise ValueError(f"Parameter '{param_name}' doesn't exist in group '{group}'.")
        # Apply the SAME canonical gating rule the node uses, so the forwarded RPC
        # carries exactly the values the node will normalize to.
        p = self.params[group][param_name]
        source, active, triggers_process, autoeval = normalize_expression_binding(
            expression, enabled, triggers_process, autoeval
        )
        # Reflect locally so the manager's snapshot matches what the node
        # will publish on its next state update.
        p.expression = source
        p.expression_enabled = active
        p.expression_triggers_process = triggers_process
        p.expression_autoeval = autoeval
        self._send(
            Message(
                MessageType.SET_EXPRESSION,
                {
                    "group": group,
                    "param_name": param_name,
                    "expression": source,
                    "expression_enabled": active,
                    "expression_triggers_process": triggers_process,
                    "expression_autoeval": autoeval,
                },
            )
        )

    def subscribe_input(self, slot_name_in: str, service_name: str, in_process: bool) -> None:
        self._send(
            Message(
                MessageType.SUBSCRIBE_INPUT,
                {
                    "slot_name_in": slot_name_in,
                    "service_name": service_name,
                    "in_process": in_process,
                },
            )
        )

    def unsubscribe_input(self, slot_name_in: str) -> None:
        self._send(Message(MessageType.UNSUBSCRIBE_INPUT, {"slot_name_in": slot_name_in}))

    def register_subscriber(self, slot_name_out: str) -> None:
        self._send(Message(MessageType.REGISTER_SUBSCRIBER, {"slot_name_out": slot_name_out}))

    def unregister_subscriber(self, slot_name_out: str) -> None:
        self._send(Message(MessageType.UNREGISTER_SUBSCRIBER, {"slot_name_out": slot_name_out}))

    def register_viewer(self, slot_name_out: str) -> None:
        """Tell the node a browser viewer attached to this output slot (Option C):
        the node starts producing + offering reduced frames on the slot's `.view`
        service even with no node consumer."""
        self._send(Message(MessageType.REGISTER_VIEWER, {"slot_name_out": slot_name_out}))

    def unregister_viewer(self, slot_name_out: str) -> None:
        self._send(Message(MessageType.UNREGISTER_VIEWER, {"slot_name_out": slot_name_out}))

    def set_viewspec(self, slot_name_out: str, spec: Dict[str, Any]) -> None:
        """Push the folded per-axis ViewSpec the node reduces this slot's viewer
        frames to. `spec` = `{axes:[{axis,max,method}], version}` (a plain dict)."""
        self._send(Message(MessageType.SET_VIEWSPEC, {"slot_name_out": slot_name_out, "spec": spec}))

    def send_directory(self, directory: Dict[str, str]) -> None:
        """Push the manager's current display-name -> node_id map so this
        node's expression engines can resolve `nd('name')` references to the
        producing node's stable transport id."""
        self._send(Message(MessageType.NODE_DIRECTORY, {"directory": dict(directory)}))

    def terminate(self) -> None:
        self._alive = False
        # Wake the data-pump out of its idle wait so it can exit.
        self._data_waitset_dirty.set()
        try:
            self._send(Message(MessageType.TERMINATE, {}))
        except Exception:
            pass
        # NB: We deliberately do *not* call `ctrl_pub.close()` here.
        # Iceoryx2's publisher drop is racy with in-flight sample
        # delivery — dropping it too soon after `send(...)` can prevent
        # the subscriber from ever observing the TERMINATE sample, even
        # though the paired event-service notify fired. Letting the
        # publisher live to GC time (when this NodeRef is collected)
        # gives the subscriber time to drain reliably. `status_sub` is
        # closed for the same reason in spirit; we trust GC.

    def wait_for_state(self, timeout: float = 3.0) -> bool:
        """Block until a STATE_UPDATE has been received at least once."""
        return self._first_state_event.wait(timeout=timeout)

    def data_service_for(self, slot_name_out: str) -> str:
        return data_service_name(self.node_id, slot_name_out)

    def open_output_subscriber(self, slot_name_out: str) -> tuple[Subscriber, Listener]:
        # REGISTER_SUBSCRIBER always provisions the node's IPC publisher
        # (`_ensure_output_endpoints(..., want_ipc=True)` in goofi.node), so
        # we always subscribe via IPC — whether the node lives in this
        # process or another. The thread transport is only used for explicit
        # SUBSCRIBE_INPUT wiring between same-group peers.
        return open_subscriber(self.data_service_for(slot_name_out), in_process=False, latest_wins=True)

    def open_view_subscriber(self, slot_name_out: str) -> tuple[Subscriber, Listener]:
        """Subscribe to the node's REDUCED viewer stream (`<dataservice>.view`),
        provisioned by REGISTER_VIEWER. Carries small node-reduced GOOF frames the
        manager relays to browsers verbatim (Option C)."""
        return open_subscriber(view_service_name(self.node_id, slot_name_out), in_process=False, latest_wins=True)

    def set_data_handler(
        self, slot_name_out: str, callback: Optional[Callable], *, raw: bool = False, view: bool = False
    ) -> None:
        """Register / unregister a callback invoked on every fresh frame.

        With `raw=False` (default) the callback is
        `callback(noderef, slot_name_out, data)` with a decoded `Data`. With
        `raw=True` it is `callback(noderef, slot_name_out, buf)` and receives
        the producer's GOOF wire `bytes` straight from the subscriber — the
        pump skips `decode_data`. The bridge uses `raw=True` so it can forward
        frames to the browser verbatim, with no manager-side re-encode (A1).

        The callback runs on the single per-ref data-pump thread that fans out
        across all registered output slots. Pass `callback=None` to unregister.

        Side-effect: also sends `REGISTER_SUBSCRIBER` / `UNREGISTER_SUBSCRIBER`
        (or `REGISTER_VIEWER` / `UNREGISTER_VIEWER` when `view=True`) so the
        producing node knows to start / stop publishing on the slot.

        With `view=True` (Option C) the handler subscribes to the node's REDUCED
        `.view` stream instead of the full output, registers as a *viewer* (so the
        node reduces per its folded ViewSpec — see `set_viewspec`), and forwards
        the node-reduced bytes verbatim (`raw` is forced True).
        """
        with self._data_handlers_lock:
            prev = self._data_handlers.pop(slot_name_out, None)
            if prev is not None:
                self._data_waitset.detach(prev[1])
                # Release BOTH endpoints — the subscriber and its listener.
                # Closing only the subscriber leaks a listener per cycle (B4).
                for endpoint in (prev[0], prev[1]):
                    try:
                        endpoint.close()
                    except Exception:
                        pass
                # Unregister on the same plane we registered on (stored on prev).
                if prev[4]:
                    self.unregister_viewer(slot_name_out)
                else:
                    self.unregister_subscriber(slot_name_out)

            if callback is None:
                return

            if view:
                raw = True  # reduced frames are pre-encoded; forward verbatim
                self.register_viewer(slot_name_out)
                sub, listener = self.open_view_subscriber(slot_name_out)
            else:
                self.register_subscriber(slot_name_out)
                sub, listener = self.open_output_subscriber(slot_name_out)
            self._data_handlers[slot_name_out] = (sub, listener, callback, raw, view)
            self._data_waitset.attach(listener)
            self._data_waitset_dirty.set()

            if self._data_pump_thread is None or not self._data_pump_thread.is_alive():
                self._data_pump_thread = Thread(
                    target=self._data_pump,
                    name=f"NodeRef-{self.node_id}-data-pump",
                    daemon=True,
                )
                self._data_pump_thread.start()

    def _data_pump(self) -> None:
        """Single thread serving every registered data handler on this NodeRef."""
        from goofi.codec import decode_data

        while self._alive:
            # Snapshot the listener→slot map so the wait can run unlocked.
            with self._data_handlers_lock:
                handlers = dict(self._data_handlers)
            if not handlers:
                # No subscribers — sleep on the dirty flag until something
                # registers (or terminate fires).
                self._data_waitset_dirty.wait(timeout=0.5)
                self._data_waitset_dirty.clear()
                continue
            listener_to_slot = {id(l): (n, s, cb, raw) for n, (s, l, cb, raw, _view) in handlers.items()}
            fired = self._data_waitset.wait(0.25)
            for listener in fired:
                entry = listener_to_slot.get(id(listener))
                if entry is None:
                    continue
                slot_name, sub, cb, raw = entry
                buf = sub.take_latest()
                if buf is None:
                    continue
                if raw:
                    # Forward the wire bytes verbatim — no decode (A1).
                    payload = buf
                else:
                    try:
                        payload = decode_data(buf)
                    except Exception:
                        import sys

                        print(f"decode failed for {self.node_id}.{slot_name}: {sys.exc_info()[1]}", file=sys.stderr)
                        continue
                try:
                    cb(self, slot_name, payload)
                except Exception:
                    import sys

                    print(f"data handler for {self.node_id}.{slot_name} failed: {sys.exc_info()[1]}", file=sys.stderr)

    def _messaging_loop(self) -> None:
        """Drain the status channel and dispatch messages."""
        while self._alive:
            fired = self._waitset.wait(0.5) if self._waitset else []
            if not fired:
                continue
            while True:
                # Guard against a concurrent `terminate()` clearing endpoints.
                if not self._alive or self.status_sub is None:
                    return
                buf = self.status_sub.take_next()
                if buf is None:
                    break
                try:
                    msg = decode_message(buf)
                except Exception:
                    traceback.print_exc()
                    continue

                self._dispatch_status(msg)

    def _dispatch_status(self, msg) -> None:
        """The SINGLE apply site for node-pushed status. Update the NodeRef cache
        from the node's authoritative push FIRST, then run any registered
        side-effect handler (e.g. the bridge broadcast). Centralizing the apply
        here means a handler and a synchronous reader always observe the same
        node-pushed values — the bridge handler used to "re-do the work" because
        the old structure skipped the default branch when a handler was set; that
        second apply site is gone. (The manager may still optimistically pre-apply
        a change it forwards via the shared normalize rule; the node echo handled
        here is the ultimate authority and overwrites it.)"""
        if msg.type == MessageType.STATE_UPDATE:
            self.serialized_state = msg.content
            self._first_state_event.set()
            # Mirror reported params back into the local NodeParams so GUI reads
            # stay coherent across reconnects.
            try:
                self.params.update(msg.content.get("params", {}))
            except Exception:
                pass
        elif msg.type == MessageType.PROCESSING_ERROR:
            self.last_error = msg.content.get("error")
        elif msg.type == MessageType.NODE_STATS:
            self.node_stats = msg.content.get("stats")

        cb = self.callbacks.get(msg.type)
        if cb is not None:
            try:
                cb(self, msg)
            except Exception:
                traceback.print_exc()


# ---------------------------------------------------------------------------
# Process groups
# ---------------------------------------------------------------------------


class NodeProcess:
    """Hosts multiple LOCAL-mode nodes in one subprocess.

    The host process runs a tiny dispatcher that receives (node_class,
    node_id, params) tuples over a setup pipe and instantiates each node
    locally. After that, all data plane traffic uses iceoryx2 (cross-group)
    or thread transport (intra-group); the setup pipe is used only for
    bootstrapping new nodes into the group.
    """

    def __init__(self, name: str, instance_id: str, capture_logs: bool = False):
        self.name = name
        self.instance_id = instance_id
        self.conn, recv = Pipe()
        self.proc = Process(
            target=self._run,
            name=name,
            args=(recv, instance_id, capture_logs),
            daemon=True,
        )
        self.proc.start()

    @staticmethod
    def _run(conn: _ConnectionBase, instance_id: str, capture_logs: bool = False):
        # In the child, propagate the instance id so service names match.
        from goofi.transport import set_instance_id

        set_instance_id(instance_id)
        nodes = []
        while True:
            try:
                payload = conn.recv()
            except (EOFError, OSError):
                break
            if payload is None:
                break
            cls, node_id, initial_params = payload
            # Drop references to nodes that have terminated. Each node closes
            # its own endpoints on shutdown (see Node._teardown_endpoints), so
            # this only releases the inert Python object — but without it the
            # host would pin every node it ever spawned for the group's life.
            nodes = [n for n in nodes if n.alive]
            try:
                node = cls._spawn_local(
                    node_id=node_id, initial_params=initial_params, capture_logs=capture_logs
                )
                nodes.append(node)
                conn.send(("ok", node_id))
            except Exception:
                conn.send(("err", traceback.format_exc()))

    def spawn(self, cls, node_id: str, initial_params: Optional[Dict[str, Dict[str, Any]]]):
        self.conn.send((cls, node_id, initial_params))
        status, info = self.conn.recv()
        if status != "ok":
            raise RuntimeError(f"Failed to spawn {node_id} in process group {self.name}: {info}")

    def kill(self):
        try:
            self.conn.send(None)
        except Exception:
            pass
        self.proc.kill()


_process_groups: Dict[str, NodeProcess] = {}


class NodeProcessRegistry:
    """Module-level registry of `NodeProcess` instances, keyed by group name.

    Implemented as a thin facade over `_process_groups` (a module-level
    dict) so the public API stays the same as the old singleton class —
    callers do `NodeProcessRegistry().get(name, iid)`. The `headless`
    attribute is a process-wide flag preserved for compatibility.
    """

    headless: bool = False

    def get(self, name: str, instance_id: str) -> NodeProcess:
        ng = _process_groups.get(name)
        if ng is None:
            # The host process captures node stdout/stderr unless headless.
            ng = NodeProcess(name, instance_id, capture_logs=not type(self).headless)
            _process_groups[name] = ng
        return ng

    def all(self) -> Dict[str, NodeProcess]:
        return dict(_process_groups)

    def terminate(self) -> None:
        for p in _process_groups.values():
            p.kill()
        _process_groups.clear()
