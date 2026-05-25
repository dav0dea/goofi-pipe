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
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Type

from goofi import nodes as goofi_nodes
from goofi.codec import decode_message, encode_message
from goofi.data import Data, DataType
from goofi.message import Message, MessageType
from goofi.params import NodeParams
from goofi.transport import (
    Listener,
    Publisher,
    Subscriber,
    WaitSet,
    create_ctrl_publisher,
    create_ctrl_subscriber,
    create_data_subscriber,
    ctrl_service_name,
    data_service_name,
    status_service_name,
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

    # Runtime, set by Node — not picklable across process boundaries.
    subscriber: Optional[Subscriber] = field(default=None, repr=False, compare=False)
    listener: Optional[Listener] = field(default=None, repr=False, compare=False)
    service_name: Optional[str] = field(default=None, repr=False, compare=False)
    in_process: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self):
        if isinstance(self.dtype, str):
            self.dtype = DataType[self.dtype]

    def clear(self):
        self.data = None

    def __getstate__(self):
        """Drop runtime endpoints on pickle (used when spawning child process)."""
        return {
            "dtype": self.dtype,
            "trigger_process": self.trigger_process,
            "data": self.data,
            "subscriber": None,
            "listener": None,
            "service_name": None,
            "in_process": False,
        }

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)


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

    # Runtime, set by Node. `publishers` and `notifiers` are parallel lists.
    publishers: List[Publisher] = field(default_factory=list, repr=False, compare=False)
    notifiers: List[object] = field(default_factory=list, repr=False, compare=False)
    # Track which transports already have endpoints, so REGISTER_SUBSCRIBER
    # is idempotent across subscriber counts.
    has_ipc: bool = field(default=False, repr=False, compare=False)
    has_thread: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self):
        if isinstance(self.dtype, str):
            self.dtype = DataType[self.dtype]

    def __getstate__(self):
        return {
            "dtype": self.dtype,
            "subscriber_count": 0,
            "publishers": [],
            "notifiers": [],
            "has_ipc": False,
            "has_thread": False,
        }

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)


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
    serialization_pending: bool = False
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

    def __post_init__(self):
        self.__doc__ = self.node_class.docstring()
        self.category = self.node_class.category()

        # Bring up our half of the ctrl/status channels.
        self.ctrl_pub, ctrl_notifier = create_ctrl_publisher(
            ctrl_service_name(self.node_id), in_process=self.in_process
        )
        # Hold a reference so the notifier isn't garbage-collected.
        self._ctrl_notifier = ctrl_notifier
        self.status_sub, self.status_listener = create_ctrl_subscriber(
            status_service_name(self.node_id), in_process=self.in_process
        )

        if self.create_initialized:
            self.initialize()

    def initialize(self) -> None:
        if self._alive:
            return
        self._alive = True
        self._waitset = WaitSet()
        self._waitset.attach(self.status_listener)
        self.serialization_pending = True  # cleared when first STATE_UPDATE arrives
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

    def terminate(self) -> None:
        self._alive = False
        try:
            self._send(Message(MessageType.TERMINATE, {}))
        except Exception:
            pass
        # Tear down our endpoints.
        if self.ctrl_pub:
            self.ctrl_pub.close()
        if self.status_sub:
            self.status_sub.close()

    def wait_for_state(self, timeout: float = 3.0) -> bool:
        """Block until a STATE_UPDATE has been received at least once. Returns True if state arrived."""
        deadline = time.time() + timeout
        while self.serialization_pending and time.time() < deadline:
            time.sleep(0.01)
        return not self.serialization_pending

    def data_service_for(self, slot_name_out: str) -> str:
        """Canonical iceoryx2 service name for one of the node's output slots."""
        return data_service_name(self.node_id, slot_name_out)

    def open_output_subscriber(self, slot_name_out: str) -> tuple[Subscriber, Listener]:
        """Open a subscriber on one of the node's output slots (e.g. for a GUI viewer)."""
        return create_data_subscriber(self.data_service_for(slot_name_out), in_process=self.in_process)

    def set_data_handler(self, slot_name_out: str, callback: Optional[Callable]) -> None:
        """Register a callback invoked whenever fresh data lands on `slot_name_out`.

        Spawns a tiny daemon thread per registered slot that owns the
        subscriber + listener pair, decodes incoming frames, and calls
        `callback(self, slot_name_out, data)` with the decoded `Data`. Pass
        `callback=None` to unregister.

        Behind the scenes this also sends a `REGISTER_SUBSCRIBER` to the
        node so the producer publishes for the GUI viewer.
        """
        if not hasattr(self, "_data_handlers"):
            self._data_handlers: Dict[str, "Thread"] = {}

        # unregister any prior handler on this slot
        prev = self._data_handlers.pop(slot_name_out, None)
        if prev is not None:
            prev_evt = getattr(prev, "_stop_evt", None)
            if prev_evt is not None:
                prev_evt.set()

        if callback is None:
            self.unregister_subscriber(slot_name_out)
            return

        self.register_subscriber(slot_name_out)
        sub, listener = self.open_output_subscriber(slot_name_out)
        from goofi.codec import decode_data
        from threading import Event as _Event
        stop_evt = _Event()

        def pump():
            ws = WaitSet()
            ws.attach(listener)
            while not stop_evt.is_set() and self._alive:
                fired = ws.wait(0.25)
                if not fired:
                    continue
                buf = sub.take_latest()
                if buf is None:
                    continue
                try:
                    data = decode_data(buf)
                except Exception:
                    traceback.print_exc()
                    continue
                try:
                    callback(self, slot_name_out, data)
                except Exception:
                    # Data callbacks are GUI code — failures here shouldn't
                    # flood the console at the pump's frame rate. Print
                    # once with a short summary; the user can re-run with
                    # PYTHONFAULTHANDLER=1 for deeper diagnosis if needed.
                    import sys
                    print(f"data handler for {self.node_id}.{slot_name_out} failed: {sys.exc_info()[1]}",
                          file=sys.stderr)

        t = Thread(target=pump, name=f"NodeRef-{self.node_id}-data-{slot_name_out}", daemon=True)
        t._stop_evt = stop_evt
        self._data_handlers[slot_name_out] = t
        t.start()

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

                if msg.type in self.callbacks:
                    try:
                        self.callbacks[msg.type](self, msg)
                    except Exception:
                        traceback.print_exc()
                    continue

                if msg.type == MessageType.STATE_UPDATE:
                    self.serialized_state = msg.content
                    self.serialization_pending = False
                    # Mirror reported params back into the local NodeParams
                    # so GUI reads stay coherent across reconnects.
                    try:
                        self.params.update(msg.content.get("params", {}))
                    except Exception:
                        pass
                elif msg.type == MessageType.PROCESSING_ERROR:
                    self.last_error = msg.content.get("error")
                elif msg.type == MessageType.SHUTDOWN:
                    # Forwarded to the manager via a registered callback.
                    pass


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

    def __init__(self, name: str, instance_id: str):
        self.name = name
        self.instance_id = instance_id
        self.conn, recv = Pipe()
        self.proc = Process(
            target=self._run,
            name=name,
            args=(recv, instance_id),
            daemon=True,
        )
        self.proc.start()

    @staticmethod
    def _run(conn: _ConnectionBase, instance_id: str):
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
            try:
                node = cls._spawn_local(node_id=node_id, initial_params=initial_params)
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


class NodeProcessRegistry:
    """Singleton registry for `NodeProcess` instances, keyed by group name."""

    _instance = None
    _processes: Dict[str, NodeProcess] = {}
    headless: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get(self, name: str, instance_id: str) -> NodeProcess:
        if name not in self._processes:
            self._processes[name] = NodeProcess(name, instance_id)
        return self._processes[name]

    def all(self) -> Dict[str, NodeProcess]:
        return dict(self._processes)

    def terminate(self):
        for p in self._processes.values():
            p.kill()
        self._processes.clear()
