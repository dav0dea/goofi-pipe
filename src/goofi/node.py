"""Base `Node` class.

Each node runs two background threads:

- **messaging loop** drains the ctrl channel (manager → node) and applies
  control messages: parameter updates, subscriber registration, input slot
  wiring, terminate. It also pushes a `STATE_UPDATE` on the status channel
  whenever the node has been marked dirty since the last push.
- **processing loop** waits on a `WaitSet` over all input slot listeners,
  drains each fired subscriber into the corresponding `slot.data` (latest
  wins), calls `process()`, and publishes the result on each output slot's
  publishers (skipping slots with no registered subscribers).

iceoryx2 endpoints are constructed lazily by the node from canonical
service names — `connection` objects are never sent across process
boundaries.
"""
from __future__ import annotations

import gc
import importlib.resources as pkg_resources
import os
import sys
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from multiprocessing import Pipe, Process
from pathlib import Path
from threading import Event, Lock, RLock, Thread, current_thread
from typing import Any, Dict, Optional, Tuple, Union

from goofi import assets
from goofi.audio.continuity import INDEX_META_KEY
from goofi.codec import decode_data, decode_message, encode_data_into, encode_message, prepare_encode
from goofi.data import Data, DataType, to_data
from goofi.expression import ExpressionEngine
from goofi import node_log, node_viewer
from goofi.message import Message, MessageType
from goofi.node_reduce import viewspec_from_dict
from goofi.node_stats import ExecStats
from goofi.node_helpers import InputSlot, NodeRef, OutputSlot, clean_docstring, normalize_config
from goofi.params import (
    InvalidParamError,
    NodeParams,
    StringParam,
    coerce_to_param_type,
    normalize_expression_binding,
)
from goofi.transport import (
    CTRL_HISTORY_SIZE,
    WaitSet,
    ctrl_service_name,
    data_service_name,
    open_publisher,
    open_subscriber,
    self_trigger_service_name,
    set_instance_id,
    status_service_name,
    view_service_name,
)


class MultiprocessingForbiddenError(Exception):
    pass


# Cadence for NODE_STATS pushes (seconds) — low + fixed, decoupled from the data
# rate so a kHz node still reports telemetry only ~once per second.
_STATS_PUSH_INTERVAL = 1.0

# Realtime processing-thread priority requested for `common.priority == "realtime"`
# nodes. Best-effort: SCHED_FIFO needs rtprio permission, else it degrades silently.
_SCHED_FIFO_PRIORITY = 10


def _safe_close(endpoint) -> None:
    """Close a transport endpoint, swallowing the None case and any error.

    Used by node teardown, where best-effort release is the right policy: a
    failed close must never stop the rest of the endpoints from being freed.
    """
    if endpoint is None:
        return
    try:
        endpoint.close()
    except Exception:
        pass


# During a live delivery-mode reallocation the manager tears down and recreates a
# source slot's shared iceoryx2 service at a new (deeper) buffer cap. The old
# shallow service lingers until its LAST holder closes — including subscribers in
# OTHER node processes — so open_or_create at the new cap transiently fails with
# `DoesNotSupportRequestedMinBufferSize`. Retrying converges: whoever opens after
# the old service is released recreates it at the new cap and the rest join.
# Bounded (~0.5 s) so a genuine misconfiguration still surfaces as an error.
_MIN_BUFFER_RETRY_ATTEMPTS = 50
_MIN_BUFFER_RETRY_DELAY = 0.01


def _open_data_endpoint(open_fn):
    """Open an iceoryx2 data-plane endpoint, retrying ONLY the transient
    min-buffer-size conflict that arises mid-reallocation (see above). Any other
    error propagates immediately; the thread transport never hits this path."""
    last = None
    for _ in range(_MIN_BUFFER_RETRY_ATTEMPTS):
        try:
            return open_fn()
        except Exception as exc:
            if "MinBufferSize" not in str(exc):
                raise
            last = exc
            time.sleep(_MIN_BUFFER_RETRY_DELAY)
    raise last


class NodeEnv(Enum):
    MULTIPROCESSING = 1  # the node owns its own process
    LOCAL = 2  # the node is running in the manager's process or a shared group process
    STANDALONE = 3  # running in the main process, but without a manager


class Node(ABC):
    """
    Base class for all nodes.

    Concrete subclasses implement `process(**inputs)`, optionally `setup()`,
    `terminate()`, and `config_input_slots() / config_output_slots() /
    config_params()`. The base class handles transport, threading, parameter
    propagation, error reporting, and the dirty-flag state push.
    """

    NO_MULTIPROCESSING = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        node_id: Optional[str],
        input_slots: Dict[str, InputSlot],
        output_slots: Dict[str, OutputSlot],
        params: NodeParams,
        environment: NodeEnv,
        *,
        in_process_with_manager: bool = False,
        capture_logs: bool = False,
    ) -> None:
        self._alive = True
        # One-time process-global GC/scheduling policy latch (Phase 4). Flipped by
        # `_apply_gc_policy`, invoked once from the processing loop after warm-up.
        self._gc_policy_applied = False
        self._setup_done = Event()
        self._dirty = False
        # `[group, name]` pairs whose ⟳ refresh completed since the last state push
        # (consumed once by `_push_state`). Lets the UI clear its per-param spinner
        # exactly when the fresh options land. Touched only on the messaging thread
        # (REFRESH_PARAM handler appends; _push_state drains) — no lock needed.
        self._refreshed_params: list[list[str]] = []
        # Last error string broadcast via PROCESSING_ERROR (None = healthy).
        # Errors are emitted on *every* occurrence so the frontend can count
        # repeats (console-style); only the cleared (None) state is deduped, so a
        # healthy node clearing every successful tick doesn't spam the channel.
        # The very first report is always sent (even None), so the manager learns
        # the node's initial health — `_error_announced` gates that.
        self._last_reported_error: Optional[str] = None
        self._error_announced = False
        # `_report_error` is called from the setup, processing, and ctrl threads;
        # this serializes the dedup-check + send + latch so the wire order can't
        # diverge from `_last_reported_error` (which would strand a stuck error).
        self._error_lock = Lock()
        # Serializes all ExpressionEngine create/eval/teardown so the same
        # engine can't run concurrently from the messaging thread (SET_EXPRESSION,
        # NODE_DIRECTORY) and the processing thread (autoeval, fired-listener
        # match), which would corrupt its subscription bookkeeping (report H2).
        # Reentrant: _set_expression calls _apply_expression under the same lock.
        # Lock order is always expression_lock -> WaitSet lock (never reversed),
        # so it can't deadlock with the data-plane wait.
        self._expression_lock = RLock()

        # Rolling execution telemetry (update rate, mean process() duration) over
        # the last N ticks. Written + read only on the processing thread, so no lock.
        self._exec_stats = ExecStats()
        self._last_stats_push = 0.0

        # Per-output monotonic emit counter for meta["index"] (§6). Written +
        # read only on the processing thread, so no lock (like _exec_stats).
        self._index_counters: Dict[str, int] = {}

        self.node_id = node_id
        self._input_slots = input_slots
        self._output_slots = output_slots
        self._params = params
        self._environment = environment
        self._in_process_with_manager = in_process_with_manager
        # stdout/stderr capture → per-node SSE log endpoint (non-headless only).
        # `_log_endpoint` is advertised to the manager via STATE_UPDATE so the
        # browser can subscribe peer-to-peer; None when capture is off.
        self._capture_logs = capture_logs
        self._log_endpoint: Optional[str] = None
        # Expression engines, keyed by (group, name). Populated lazily when
        # a SET_EXPRESSION ctrl message arrives (or when a saved patch
        # spawns the node with non-None expressions in its param dict).
        self._expressions: Dict[Tuple[str, str], ExpressionEngine] = {}
        # display name -> node_id, pushed by the manager (NODE_DIRECTORY). Lets
        # `nd('name')` expression references resolve to the producer's stable
        # id. Empty until the first push; references resolve to the literal
        # name meanwhile and self-heal once the directory arrives.
        self._node_directory: Dict[str, str] = {}

        # Per-slot iceoryx2 publisher for the reduced viewer stream (Option C).
        # Created lazily on the first REGISTER_VIEWER; the reducer thread sends
        # reduced frames on it (the manager relays them to browsers verbatim).
        self._view_pubs: Dict[str, tuple] = {}

        self._validate_attrs()
        self._waitset: WaitSet = WaitSet()

        if environment == NodeEnv.STANDALONE:
            # No transport / no threads — used by offline analysis code that
            # invokes `node(...)` directly via `__call__`.
            return

        self.ctrl_sub, self.ctrl_listener = open_subscriber(
            ctrl_service_name(node_id), in_process=in_process_with_manager, safe_overflow=False, buffer_cap=CTRL_HISTORY_SIZE
        )
        self.status_pub, self._status_notifier = open_publisher(
            status_service_name(node_id), in_process=in_process_with_manager, safe_overflow=False, buffer_cap=CTRL_HISTORY_SIZE
        )
        # Install stdout/stderr capture and start this process's SSE log server.
        # Idempotent per process; `register_node` returns the URL the frontend
        # connects to directly (the manager only relays the address).
        if capture_logs:
            self._log_endpoint = node_log.register_node(node_id)
        # Self-trigger pair: in-process pub/sub the messaging thread (or
        # an expression engine running in the processing thread) uses to
        # wake the processing loop without going through input slots.
        # Latest-wins so a burst of notifies coalesces into a single tick.
        self._self_trigger_pub, self._self_trigger_notifier = open_publisher(
            self_trigger_service_name(node_id), in_process=True
        )
        self._self_trigger_sub, self._self_trigger_listener = open_subscriber(
            self_trigger_service_name(node_id), in_process=True
        )
        self._waitset.attach(self._self_trigger_listener)
        # NB: ctrl_listener is *not* attached to `self._waitset` — that
        # WaitSet belongs to the processing loop and only watches input
        # data slots. The messaging loop owns its own WaitSet locally;
        # sharing a single listener between two WaitSets caused the
        # listener's events to be drained by whichever WaitSet woke
        # first, silently swallowing control messages on the other side.

        self.processing_thread = Thread(
            target=self._processing_loop,
            name=f"{self.__class__.__name__}-proc_loop",
            daemon=True,
        )
        self.processing_thread.start()

        if environment == NodeEnv.MULTIPROCESSING:
            self._messaging_loop()
        elif environment == NodeEnv.LOCAL:
            self.messaging_thread = Thread(
                target=self._messaging_loop,
                name=f"{self.__class__.__name__}-msg_loop",
                daemon=True,
            )
            self.messaging_thread.start()
        else:
            raise ValueError(f"Invalid environment: {environment}")

    def _validate_attrs(self) -> None:
        if self._environment != NodeEnv.STANDALONE:
            if not isinstance(self.node_id, str) or not self.node_id:
                raise ValueError("Expected non-empty node_id string.")
        for name, slot in self._input_slots.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"Expected input slot name '{name}' to be a non-empty string.")
            if not isinstance(slot, InputSlot):
                raise TypeError(f"Expected InputSlot for '{name}', got {type(slot)}")
        for name, slot in self._output_slots.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"Expected output slot name '{name}' to be a non-empty string.")
            if not isinstance(slot, OutputSlot):
                raise TypeError(f"Expected OutputSlot for '{name}', got {type(slot)}")
        if not isinstance(self._params, NodeParams):
            raise TypeError(f"Expected NodeParams, got {type(self._params)}")

    # ------------------------------------------------------------------
    # Setup / lifecycle
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        if self._capture_logs:
            node_log.bind_thread_node(self.node_id)
        try:
            self.setup()
            self._instantiate_saved_expressions()
            self._setup_done.set()
            self._report_error(None)
        except Exception:
            # No retry: with imports at module top, a setup failure is a real
            # bug or missing resource, not a transient race. Surface it; the
            # node idles (ctrl still works) and the manual restart affordance
            # re-runs the full bootstrap.
            self._report_error(traceback.format_exc())
        finally:
            # Announce setup_complete (or its absence) promptly via the state
            # plane — this is what flips the UI's 'setting up…' spinner.
            self._mark_dirty()

    def _instantiate_saved_expressions(self) -> None:
        """Build engines for any param that arrived from a saved patch
        with a non-None ``expression`` set on it."""
        for group in self.params.keys():
            group_data = self.params[group]
            for name in group_data._fields:
                param = group_data[name]
                expr = getattr(param, "expression", None)
                if expr:
                    try:
                        self._set_expression(
                            group,
                            name,
                            expr,
                            enabled=bool(getattr(param, "expression_enabled", False)),
                            triggers_process=bool(
                                getattr(param, "expression_triggers_process", False)
                            ),
                            autoeval=bool(getattr(param, "expression_autoeval", False)),
                        )
                    except Exception:
                        self._report_error(traceback.format_exc())

    def _report_error(self, error: Optional[str]) -> None:
        if self._environment == NodeEnv.STANDALONE:
            return
        # Hold the lock across the dedup check, the send, and the latch so
        # concurrent callers (setup / processing / ctrl threads) can't interleave
        # such that the wire order diverges from `_last_reported_error` — which
        # would suppress every later clear and strand a recovered node's error.
        with self._error_lock:
            # Dedupe only the cleared (None) state: a healthy node calls this every
            # successful tick, and re-sending None would flood the channel. Real
            # errors are always emitted so the frontend can count repeats. The
            # first report always goes out, announcing the node's initial health.
            if error is None and self._last_reported_error is None and self._error_announced:
                return
            try:
                self.status_pub.send(
                    encode_message(Message(MessageType.PROCESSING_ERROR, {"error": error}))
                )
                self._status_notifier.notify()
                # Only latch once the change actually went out, so a failed send
                # is retried on the next report.
                self._last_reported_error = error
                self._error_announced = True
            except Exception:
                pass

    def _clear_error_if_healthy(self) -> None:
        """Clear the reported error after a clean processing tick — UNLESS an
        expression engine is still failing. An autoeval expression that errors
        every tick (e.g. ``1/0``) would otherwise flicker error↔clear within a
        single tick (the expression reports, then a successful ``process()``
        clears), making the error effectively invisible in the inspector/chip.
        While an expression is actively failing the node isn't healthy, so the
        error stays reported."""
        for engine in self._expressions.values():
            if engine.last_error:
                return
        self._report_error(None)

    def _mark_dirty(self) -> None:
        self._dirty = True

    def _wake_processing(self) -> None:
        """Notify the processing loop's WaitSet from any thread.

        Safe no-op when the node isn't fully wired (STANDALONE env or
        between __init__ and the first transport setup completing).
        """
        notifier = getattr(self, "_self_trigger_notifier", None)
        if notifier is None:
            return
        try:
            notifier.notify()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Expression-bound params
    # ------------------------------------------------------------------

    def _resolve_node_ref(self, name: str) -> str:
        """Resolve a display name from `nd('name')` to the producing node's
        stable transport id via the manager-pushed directory. Falls back to
        the literal name when the directory hasn't arrived (or the referenced
        node doesn't exist yet); the expression engine re-resolves each eval,
        so it self-heals once the directory catches up."""
        return self._node_directory.get(name, name)

    def _set_expression(
        self,
        group: str,
        name: str,
        expression: Optional[str],
        enabled: bool = False,
        triggers_process: bool = False,
        autoeval: bool = False,
    ) -> None:
        """Bind / unbind an expression on a param.

        ``expression`` is the source — kept on the Param even when
        ``enabled`` is False so a user can toggle fx off and back on
        without losing what they wrote. ``enabled`` gates whether the
        ExpressionEngine is actually constructed and evaluating.
        """
        if group not in self.params or name not in self.params[group]:
            self._report_error(f"Unknown parameter {group}.{name}")
            return
        param = self.params[group][name]
        key = (group, name)

        # The node is the AUTHORITY for this binding: apply the canonical gating
        # rule (shared with the manager-side proxy) and write it onto the Param.
        source, active, triggers_process, autoeval = normalize_expression_binding(
            expression, enabled, triggers_process, autoeval
        )
        param.expression = source
        param.expression_enabled = active
        param.expression_triggers_process = triggers_process
        param.expression_autoeval = autoeval

        with self._expression_lock:
            engine = self._expressions.get(key)
            if active:
                if engine is None:
                    engine = ExpressionEngine(
                        location=f"{self.node_id}.{group}.{name}",
                        on_listener_added=self._waitset.attach,
                        on_listener_removed=self._waitset.detach,
                        resolve_node_id=self._resolve_node_ref,
                    )
                    self._expressions[key] = engine
                engine.set_source(source)
                if engine.last_error:
                    self._report_error(engine.last_error)
                else:
                    self._report_error(None)
                # Warm eval so the param has a value immediately. Subscriptions
                # for any referenced slots are opened during this first call.
                self._apply_expression(key, engine)
            else:
                # Tear down any running engine; the source stays on the Param.
                if engine is not None:
                    engine.close()
                    self._expressions.pop(key, None)
        self._mark_dirty()

    def _apply_expression(self, key: Tuple[str, str], engine: ExpressionEngine) -> None:
        """Run the engine once and push its result into the param.

        The new value is always written into the param. Whether that
        change wakes process() is gated by the per-param
        `expression_triggers_process` flag.
        """
        group, name = key
        with self._expression_lock:
            result = engine.evaluate()
            if engine.last_error:
                self._report_error(engine.last_error)
            if result is None:
                return
            param = self.params[group][name]
            coerced = coerce_to_param_type(param, result)
            if coerced is None:
                return
            prev = param._value
            param._value = coerced
            self._mark_dirty()
            if prev != coerced and getattr(param, "expression_triggers_process", False):
                self._wake_processing()

    def _match_fired_expression(self, listener) -> bool:
        """Run the expression engine owning `listener`, if any. Returns True
        when a matching engine was found and applied.

        Snapshots `_expressions` because the messaging thread can SET/clear an
        expression (mutating the dict) while this processing-thread loop walks
        it — iterating the live dict would raise RuntimeError (report H4). The
        engine lock serializes the match+apply against a concurrent set/clear
        (report H2).
        """
        with self._expression_lock:
            for key, engine in list(self._expressions.items()):
                if engine.owns_listener(listener):
                    self._apply_expression(key, engine)
                    return True
        return False

    def _push_state(self) -> None:
        """Publish a STATE_UPDATE if we're dirty."""
        if self._environment == NodeEnv.STANDALONE:
            self._dirty = False
            self._refreshed_params.clear()
            return
        if not self._dirty:
            return
        # StringParam option lists, as THIS process sees them — the authority
        # for runtime-enumerated options (device lists). The manager merges
        # these into its ref (NodeParams.update_options).
        param_options = {}
        for group_name in self.params.keys():
            group = self.params[group_name]
            opts = {
                name: group[name].options
                for name in group._fields
                if isinstance(group[name], StringParam) and group[name].options
            }
            if opts:
                param_options[group_name] = opts
        state = {
            "_type": type(self).__name__,
            "category": self.category(),
            "params": self.params.serialize(),
            "param_options": param_options,
            # `[group, name]` pairs whose ⟳ refresh just completed — the UI clears the
            # per-param spinner on these. Empty on ordinary pushes.
            "refreshed_params": list(self._refreshed_params),
            "output_subscribers": {n: s.subscriber_count for n, s in self.output_slots.items()},
            # Lifecycle: False while setup() is still running on its thread —
            # drives the manager-side 'setup' stage between the first state
            # push and setup completion.
            "setup_complete": self._setup_done.is_set(),
            # Peer-to-peer SSE log endpoint for this node (None when capture off).
            "log_endpoint": self._log_endpoint,
        }
        try:
            # Carry the node's current error on the idempotent state plane so a lost
            # PROCESSING_ERROR (the FIRST status message is dropped — iceoryx2 keeps
            # no history) still reaches the ref: a setup() failure otherwise hangs on
            # an eternal 'setting up…' spinner, and a healthy respawn's error-clear
            # never lifts a stale chip. Read the latched value AND send under
            # _error_lock — the same lock _report_error holds across its send — so a
            # concurrent PROCESSING_ERROR can't order itself between the read and the
            # send and make this push carry a stale error (a spurious clear/re-raise).
            with self._error_lock:
                state["last_error"] = self._last_reported_error
                self.status_pub.send(encode_message(Message(MessageType.STATE_UPDATE, state)))
                self._status_notifier.notify()
            self._dirty = False
            # Drain only after a successful send so a failed push re-reports the
            # completion on the next attempt (else the spinner would hang).
            self._refreshed_params.clear()
        except Exception:
            traceback.print_exc()

    def _stats_extra(self) -> dict:
        """Node-specific telemetry merged into the NODE_STATS payload. The base
        node has none; latency-sensitive sinks (AudioOut) override this to surface
        e.g. output latency and jitter-ring under/overrun counters in the
        inspector, without a bespoke status channel."""
        return {}

    def _build_stats_payload(self) -> dict:
        """The NODE_STATS payload: rolling exec stats plus any node-specific
        extras. Built + read on the processing thread, so no locking."""
        return {**self._exec_stats.snapshot(), **self._stats_extra()}

    def _maybe_push_stats(self, now: float) -> None:
        """Emit NODE_STATS at a low fixed cadence (decoupled from the data rate and
        the dirty/clean STATE_UPDATE). Cheap: one ExecStats snapshot + ctrl publish.
        `now` is a perf-counter reading (same clock as the throttle baseline)."""
        if self._environment == NodeEnv.STANDALONE:
            return
        if now - self._last_stats_push < _STATS_PUSH_INTERVAL:
            return
        self._last_stats_push = now
        try:
            self.status_pub.send(
                encode_message(Message(MessageType.NODE_STATS, {"stats": self._build_stats_payload()}))
            )
            self._status_notifier.notify()
        except Exception:
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Messaging loop (ctrl plane)
    # ------------------------------------------------------------------

    def _messaging_loop(self) -> None:
        if self._capture_logs:
            node_log.bind_thread_node(self.node_id)
        # The subclass must not override `_setup`.
        if self._setup.__self__.__class__._setup is not Node._setup:
            error = (
                f"The {self.__class__.__name__} node implements the _setup() method, which is reserved for "
                "internal use. Use setup() instead."
            )
            self._report_error(error)
            raise RuntimeError(error)

        self._setup_thread = Thread(
            target=self._setup, name=f"{self.__class__.__name__}-setup", daemon=True
        )
        self._setup_thread.start()

        ctrl_ws = WaitSet()
        ctrl_ws.attach(self.ctrl_listener)

        while self.alive:
            # Block on ctrl channel; short timeout so we re-check `alive`
            # and can push dirty state changes promptly.
            fired = ctrl_ws.wait(0.25)
            # Drain any pending ctrl messages, regardless of whether a
            # wake fired (other threads may have queued).
            while True:
                buf = self.ctrl_sub.take_next()
                if buf is None:
                    break
                try:
                    msg = decode_message(buf)
                except Exception:
                    self._report_error(traceback.format_exc())
                    continue
                try:
                    self._handle_ctrl(msg)
                except Exception:
                    self._report_error(traceback.format_exc())

            # If `_handle_ctrl` processed a TERMINATE message, bail out
            # before pushing more state to keep shutdown latency low.
            if not self.alive:
                break

            # Push any pending state changes.
            self._push_state()

            if not self.alive:
                break

        try:
            self.terminate()
        except Exception:
            self._report_error(traceback.format_exc())

        # Flush any pending (un-newlined) log line and drop this node's buffer.
        if self._capture_logs:
            node_log.unregister_node(self.node_id)

        # Release the node's transport endpoints. The process exits right after
        # this in a one-node-per-process spawn, but in a shared group host it
        # lives on — so an un-closed publisher would pin its per-service slot
        # forever, permanently blocking that service name. This runs last, after
        # the user terminate() hook and log flush, when nothing else publishes.
        self._teardown_endpoints()

    def _teardown_endpoints(self) -> None:
        """Join the worker threads, then close every transport endpoint the
        node owns so it leaves no iceoryx2 / thread resources behind."""
        # Nudge the processing loop out of its wait and wait for both worker
        # threads to stop touching the endpoints before we drop them.
        self._wake_processing()
        for th in (getattr(self, "processing_thread", None), getattr(self, "_setup_thread", None)):
            if th is not None and th is not current_thread():
                th.join(timeout=1.0)

        for engine in list(self._expressions.values()):
            try:
                engine.close()
            except Exception:
                pass
        self._expressions.clear()

        for slot in self._input_slots.values():
            _safe_close(slot.subscriber)
            _safe_close(slot.listener)
            slot.subscriber = None
            slot.listener = None

        for slot in self._output_slots.values():
            for endpoint in (*slot.publishers, *slot.notifiers):
                _safe_close(endpoint)
            slot.publishers.clear()
            slot.notifiers.clear()
            slot.has_ipc = False
            slot.has_thread = False

        # Reduced-viewer ('.view') endpoints + reducer state (Option C). These live
        # in self._view_pubs, separate from the node↔node output endpoints above, so
        # the loops there miss them. evict() FIRST — it removes the sink and blocks
        # until the SHARED reducer thread is no longer mid-send on this slot — so the
        # close below can't race the reducer loaning/sending on a dropped publisher;
        # then close releases the SHM segments. (Order matters: close-then-evict
        # would leave a use-after-close window.)
        for slot_name, (pub, notif) in self._view_pubs.items():
            node_viewer.evict(self.node_id, slot_name)
            _safe_close(pub)
            _safe_close(notif)
        self._view_pubs.clear()

        for attr in (
            "_self_trigger_pub",
            "_self_trigger_sub",
            "_self_trigger_notifier",
            "_self_trigger_listener",
            "ctrl_sub",
            "ctrl_listener",
            "status_pub",
            "_status_notifier",
        ):
            _safe_close(getattr(self, attr, None))

    def _handle_ctrl(self, msg: Message) -> None:
        t = msg.type
        if t == MessageType.TERMINATE:
            self._alive = False
        elif t == MessageType.PARAMETER_UPDATE:
            group = msg.content["group"]
            name = msg.content["param_name"]
            value = msg.content["param_value"]
            if group not in self.params or name not in self.params[group]:
                self._report_error(f"Unknown parameter {group}.{name}")
                return
            coerced = coerce_to_param_type(self.params[group][name], value)
            if coerced is None:
                self._report_error(
                    f"Cannot coerce {value!r} to the type of {group}.{name}; ignoring update."
                )
                return
            value = coerced
            self.params[group][name].value = value
            self._mark_dirty()
            cb = getattr(self, f"{group}_{name}_changed", None)
            if callable(cb):
                try:
                    cb(value)
                except Exception:
                    self._report_error(traceback.format_exc())
        elif t == MessageType.REFRESH_PARAM:
            group = msg.content["group"]
            name = msg.content["param_name"]
            try:
                if group not in self.params or name not in self.params[group]:
                    self._report_error(f"Unknown parameter {group}.{name}")
                else:
                    param = self.params[group][name]
                    fn_name = getattr(param, "refresh", None)
                    fn = getattr(self, fn_name, None) if fn_name else None
                    if callable(fn):
                        try:
                            opts = list(fn())
                            # Keep the current selection valid: if it fell out of the
                            # fresh list, prepend it so the dropdown doesn't re-point.
                            cur = param.value
                            if cur and cur not in opts:
                                opts = [cur, *opts]
                            param.options = opts
                        except Exception:
                            self._report_error(traceback.format_exc())
            finally:
                # Signal completion for ANY outcome (fresh options, refresh error, or an
                # unknown param) so the UI clears its ⟳ spinner on the next push rather
                # than stalling until the safety timeout. The refresh ran synchronously
                # on this messaging thread, so that push also carries the fresh options.
                self._refreshed_params.append([group, name])
                self._mark_dirty()
        elif t == MessageType.REGISTER_SUBSCRIBER:
            slot_name = msg.content["slot_name_out"]
            slot = self.output_slots[slot_name]
            slot.subscriber_count += 1
            self._ensure_output_endpoints(
                slot_name,
                want_ipc=True,
                buffer_cap=msg.content.get("buffer_cap"),
            )
            self._mark_dirty()
        elif t == MessageType.UNREGISTER_SUBSCRIBER:
            slot_name = msg.content["slot_name_out"]
            slot = self.output_slots[slot_name]
            slot.subscriber_count = max(0, slot.subscriber_count - 1)
            self._mark_dirty()
        elif t == MessageType.REGISTER_VIEWER:
            slot_name = msg.content["slot_name_out"]
            slot = self.output_slots[slot_name]
            # Wire the reduced-stream publisher + reducer sink BEFORE bumping the
            # count, so the first OR-gated offer() always has a sink to deliver to.
            node_viewer.register_viewer(self.node_id, slot_name, self._view_sink(slot_name))
            with slot.viewer_lock:
                slot.viewer_count += 1
            self._wake_processing()  # tick an idle free-running node so it produces
            self._mark_dirty()
        elif t == MessageType.UNREGISTER_VIEWER:
            slot_name = msg.content["slot_name_out"]
            slot = self.output_slots[slot_name]
            with slot.viewer_lock:
                slot.viewer_count = max(0, slot.viewer_count - 1)
                gone = slot.viewer_count == 0
            if gone:
                node_viewer.evict(self.node_id, slot_name)  # release pinned Data
            self._mark_dirty()
        elif t == MessageType.SET_VIEWSPEC:
            slot_name = msg.content["slot_name_out"]
            node_viewer.set_viewspec(
                self.node_id, slot_name, viewspec_from_dict(msg.content["spec"])
            )
        elif t == MessageType.SUBSCRIBE_INPUT:
            self._subscribe_input(
                slot_name_in=msg.content["slot_name_in"],
                service_name=msg.content["service_name"],
                in_process=msg.content["in_process"],
                queue=bool(msg.content.get("queue", False)),
                buffer_cap=msg.content.get("buffer_cap"),
            )
            self._mark_dirty()
        elif t == MessageType.UNSUBSCRIBE_INPUT:
            self._unsubscribe_input(msg.content["slot_name_in"])
            self._mark_dirty()
        elif t == MessageType.SET_EXPRESSION:
            self._set_expression(
                group=msg.content["group"],
                name=msg.content["param_name"],
                expression=msg.content.get("expression"),
                enabled=bool(msg.content.get("expression_enabled", False)),
                triggers_process=bool(msg.content.get("expression_triggers_process", False)),
                autoeval=bool(msg.content.get("expression_autoeval", False)),
            )
        elif t == MessageType.NODE_DIRECTORY:
            new_directory = dict(msg.content["directory"])
            if new_directory != self._node_directory:
                self._node_directory = new_directory
                # Re-resolve cross-node references against the new directory.
                # An expression warm-evaluated before a producer's id was known
                # (e.g. during a patch load, where nodes resolve in setup order)
                # would otherwise stay wired to a dangling service forever. Re-
                # applying re-keys the subscription onto the producer's real id.
                with self._expression_lock:
                    for expr_key, engine in list(self._expressions.items()):
                        if engine.references_other_nodes():
                            self._apply_expression(expr_key, engine)
        else:
            self._report_error(f"Unhandled control message type: {t}")

    # ------------------------------------------------------------------
    # Input slot wiring (called from messaging loop)
    # ------------------------------------------------------------------

    def _subscribe_input(
        self, *, slot_name_in: str, service_name: str, in_process: bool, queue: bool = False, buffer_cap=None
    ) -> None:
        slot = self.input_slots.get(slot_name_in)
        if slot is None:
            self._report_error(f"Unknown input slot: {slot_name_in}")
            return
        # Tear down any prior subscription on this slot.
        self._unsubscribe_input(slot_name_in)

        # queue → hold the full ordered burst (buffer_size == service cap);
        # latest → keep only the newest (take_latest drains a 2-deep port).
        sub, listener = _open_data_endpoint(
            lambda: open_subscriber(
                service_name,
                in_process=in_process,
                buffer_cap=buffer_cap,
                buffer_size=buffer_cap if queue else 2,
            )
        )
        slot.subscriber = sub
        slot.listener = listener
        slot.queue = queue
        # Attach EVERY input's listener so non-triggering slots are drained too;
        # whether a fire actually triggers process() is gated separately on
        # slot.trigger_process in the processing loop (report I2).
        self._waitset.attach(listener)

    def _unsubscribe_input(self, slot_name_in: str) -> None:
        slot = self.input_slots.get(slot_name_in)
        if slot is None or slot.subscriber is None:
            return
        if slot.listener is not None:
            self._waitset.detach(slot.listener)
        _safe_close(slot.subscriber)
        slot.subscriber = None
        slot.listener = None
        slot.clear()

    # ------------------------------------------------------------------
    # Output endpoints (lazy)
    # ------------------------------------------------------------------

    def _close_ipc_publisher(self, slot) -> None:
        """Tear down a slot's ipc publisher/notifier so its service can be
        recreated at a new buffer depth. v1 only: skipped by the caller if a
        thread publisher coexists (co-located fan-out is deferred, §12)."""
        for endpoint in (*slot.publishers, *slot.notifiers):
            _safe_close(endpoint)
        slot.publishers.clear()
        slot.notifiers.clear()
        slot.has_ipc = False
        slot.ipc_buffer_cap = None

    def _ensure_output_endpoints(
        self,
        slot_name: str,
        *,
        want_ipc: bool = False,
        want_thread: bool = False,
        buffer_cap=None,
    ) -> None:
        """Idempotent — create the requested publisher flavours for a slot.

        `REGISTER_SUBSCRIBER` always requests the ipc flavour; the manager sizes
        `buffer_cap` from the wired endpoints. Thread publishers are added when a
        same-group consumer subscribes. Endpoints are cheap (iceoryx2 services are
        ref-counted via `open_or_create`).
        """
        slot = self.output_slots[slot_name]
        service = data_service_name(self.node_id, slot_name)
        if want_ipc:
            if slot.has_ipc and buffer_cap != slot.ipc_buffer_cap and not slot.has_thread:
                # Live delivery-mode override changed the channel depth — drop the
                # ipc publisher so open_or_create rebuilds the service at the new
                # cap. Safe in v1: the consumer is unsubscribed before reallocation,
                # so this process holds the only remaining service reference.
                self._close_ipc_publisher(slot)
            if not slot.has_ipc:
                pub, notif = _open_data_endpoint(
                    lambda: open_publisher(service, in_process=False, buffer_cap=buffer_cap)
                )
                slot.publishers.append(pub)
                slot.notifiers.append(notif)
                slot.has_ipc = True
                slot.ipc_buffer_cap = buffer_cap
        if want_thread and not slot.has_thread:
            pub, notif = open_publisher(service, in_process=True)
            slot.publishers.append(pub)
            slot.notifiers.append(notif)
            slot.has_thread = True

    def _ensure_view_endpoint(self, slot_name: str) -> tuple:
        """Idempotent — create (once) the iceoryx2 publisher for a slot's reduced
        viewer stream on the `<dataservice>.view` service. Separate from the
        node↔node service so viewers receive reduced frames and node consumers the
        full Data. Created on the messaging thread (REGISTER_VIEWER); the reducer
        thread publishes on it."""
        existing = self._view_pubs.get(slot_name)
        if existing is not None:
            return existing
        service = view_service_name(self.node_id, slot_name)
        pub, notif = open_publisher(service, in_process=False)
        self._view_pubs[slot_name] = (pub, notif)
        return pub, notif

    def _view_sink(self, slot_name: str):
        """Return the reducer-thread sink that publishes one encoded reduced frame
        on the slot's `.view` service. Closes over the publisher created here on the
        messaging thread; the returned closure runs on the reducer thread."""
        pub, notif = self._ensure_view_endpoint(slot_name)

        def _sink(buf: bytes) -> None:
            loan = pub.loan(len(buf))
            loan.buffer[:] = buf
            loan.send()
            notif.notify()

        return _sink

    # ------------------------------------------------------------------
    # Processing loop (data plane)
    # ------------------------------------------------------------------

    def _drain_pending_inputs(self) -> None:
        """Non-blocking take of the latest frame on every subscribed input.

        Used by free-running (autotrigger) nodes so their non-triggering inputs
        still deliver current data to process() even without a wait()-driven
        fire (report I2)."""
        for slot in self.input_slots.values():
            # Capture once — _unsubscribe_input (messaging thread) can null
            # slot.subscriber between the check and the take.
            sub = slot.subscriber
            if sub is None:
                continue
            buf = sub.take_latest()
            if buf is not None:
                try:
                    slot.data = decode_data(buf)
                except Exception:
                    self._report_error(traceback.format_exc())

    def _drain_queue(self, sub) -> "list[Data]":
        """Drain ALL pending frames on a queue-mode input (take_next until empty),
        decoding each in arrival order."""
        frames: "list[Data]" = []
        while True:
            buf = sub.take_next()
            if buf is None:
                break
            try:
                frames.append(decode_data(buf))
            except Exception:
                self._report_error(traceback.format_exc())
        return frames

    def _dispatch_burst(self, queued_frames: "dict[str, list[Data]]") -> None:
        """Run the process/publish body once per frame of the deepest fired queue
        (>=1). Between calls, each fired queue slot's data advances frame-by-frame
        (clamped to its last frame if shorter); latest/non-fired inputs are held."""
        burst = max((len(f) for f in queued_frames.values()), default=1)
        for i in range(burst):
            for slot_name, frames in queued_frames.items():
                self.input_slots[slot_name].data = frames[i] if i < len(frames) else frames[-1]
            self._dispatch_once()
            if not self.alive:
                break

    def _dispatch_once(self) -> None:
        """Snapshot inputs, run process(), and publish per output slot.

        The single tick body, shared by the latest-wins path (called once per
        wake) and the queue path (called once per drained frame)."""
        input_data = {name: slot.data for name, slot in self.input_slots.items()}

        _t0 = time.perf_counter()
        try:
            output_data = self.process(**input_data)
        except Exception:
            self._report_error(traceback.format_exc())
            return
        _t1 = time.perf_counter()
        self._exec_stats.record(_t1 - _t0, _t1)
        self._maybe_push_stats(_t1)

        # One-time GC/scheduling policy, applied only after a real tick has
        # succeeded (warm-up done) so it never delays ready/_setup_done.
        if not self._gc_policy_applied:
            self._apply_gc_policy()

        if not self.alive:
            return
        if output_data is None:
            # A clean tick that produced nothing still clears a stale error.
            self._clear_error_if_healthy()
            return
        if not isinstance(output_data, dict):
            self._report_error(f"process() must return a dict, got {type(output_data).__name__}.")
            return

        # Track whether anything went wrong this tick. A fully clean tick clears
        # a previously-reported error (the node has recovered).
        tick_error = False
        extra_fields = set(output_data.keys()) - set(self.output_slots.keys())
        if extra_fields:
            self._report_error(
                f"Extra output fields: {sorted(extra_fields)}. process() must only return declared slots."
            )
            tick_error = True

        for slot_name, slot in self.output_slots.items():
            # OR-gate: produce if a node consumer OR a browser viewer wants it.
            if slot.subscriber_count == 0 and slot.viewer_count == 0:
                continue
            value = output_data.get(slot_name)
            if value is None:
                continue
            try:
                data = self._build_output_data(slot_name, slot, value)
            except Exception:
                self._report_error(traceback.format_exc())
                tick_error = True
                continue
            # Node↔node iceoryx2 fan-out — ONLY when real node consumers exist
            # (Option C split-encode). Zero-copy publish: pack the meta once via
            # `prepare_encode`, then loan a slice per publisher and write the Data
            # directly into the loan.
            if slot.subscriber_count > 0:
                try:
                    size, meta_bytes = prepare_encode(data)
                except Exception:
                    self._report_error(traceback.format_exc())
                    tick_error = True
                    size = None
                if size is not None:
                    for pub, notif in zip(slot.publishers, slot.notifiers):
                        try:
                            loan = pub.loan(size)
                            encode_data_into(data, loan.buffer, meta_bytes=meta_bytes)
                            loan.send()
                            notif.notify()
                        except Exception:
                            self._report_error(traceback.format_exc())
                            tick_error = True
            # Browser reduced fan-out — ONLY when a viewer is attached. O(1) offer
            # to the reducer thread (Change A), independent of the node↔node path.
            if slot.viewer_count > 0:
                node_viewer.offer(self.node_id, slot_name, data)

        if not tick_error:
            self._clear_error_if_healthy()

    def _processing_loop(self) -> None:
        if self._capture_logs:
            node_log.bind_thread_node(self.node_id)
        # Wait for setup() to finish before any processing tick. Polling the
        # Event with a short timeout lets us bail out promptly if the node
        # is terminated mid-setup.
        while not self._setup_done.wait(timeout=0.25):
            if not self.alive:
                return

        last_update = 0.0

        while self.alive:
            autotrigger = self.params.common.autotrigger.value
            triggered = False
            # Ordered frames drained from fired QUEUE inputs this wake (name -> [Data]).
            queued_frames: "dict[str, list[Data]]" = {}

            if autotrigger and self._has_no_triggering_inputs():
                # Free-running mode: process every tick. Pull any fresh input
                # data first so non-triggering inputs reach process() (I2).
                self._drain_pending_inputs()
                triggered = True
            else:
                fired = self._waitset.wait(0.1)
                if not self.alive:
                    break
                # Drain every fired data listener into its slot. A fire on
                # the self-trigger listener counts as a trigger but maps to
                # no slot — it's just a wake signal. A fire on an
                # expression-engine's subscriber listener runs that
                # engine's evaluation; whether process() re-triggers from
                # that depends on common.process_on_param_update.
                drained_any = False
                self_triggered = False
                for listener in fired:
                    if listener is self._self_trigger_listener:
                        try:
                            self._self_trigger_sub.take_latest()
                        except Exception:
                            pass
                        self_triggered = True
                        continue
                    if self._match_fired_expression(listener):
                        continue
                    # Fall back to: input slot listeners.
                    for slot_name, slot in self.input_slots.items():
                        # Capture the subscriber once — the messaging thread can
                        # null slot.subscriber via _unsubscribe_input between the
                        # check and the take, which would AttributeError.
                        sub = slot.subscriber
                        if slot.listener is listener and sub is not None:
                            if slot.queue:
                                # Lossless: drain every frame; dispatch runs
                                # process() once per frame (holding other inputs).
                                frames = self._drain_queue(sub)
                                if frames:
                                    queued_frames[slot_name] = frames
                                    slot.data = frames[-1]
                                    if slot.trigger_process:
                                        drained_any = True
                            else:
                                buf = sub.take_latest()
                                if buf is not None:
                                    try:
                                        slot.data = decode_data(buf)
                                        # Only a triggering slot's fresh data wakes
                                        # process(); non-triggering inputs are still
                                        # drained above so their data is current (I2).
                                        if slot.trigger_process:
                                            drained_any = True
                                    except Exception:
                                        self._report_error(traceback.format_exc())
                            break
                if drained_any or self_triggered:
                    triggered = True

            if not triggered:
                continue

            # Rate limiting. max_frequency <= 0 means unbounded (see method).
            sleep_time = self._rate_limit_sleep(last_update, time.time())
            if sleep_time is not None:
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_update = time.time()

            # Per-param expression autoeval: re-evaluate engines whose
            # param opted into "refresh before every process". This is what
            # makes expressions without slot references (`time.time()`,
            # `random.uniform(...)`) update per tick.
            if self._expressions:
                with self._expression_lock:
                    for key, engine in list(self._expressions.items()):
                        param = self.params[key[0]][key[1]]
                        if getattr(param, "expression_autoeval", False):
                            self._apply_expression(key, engine)

            # One dispatch per drained queue frame (latest-wins → burst of 1).
            self._dispatch_burst(queued_frames)

    def _has_no_triggering_inputs(self) -> bool:
        return not any(s.trigger_process and s.subscriber is not None for s in self.input_slots.values())

    def _rate_limit_sleep(self, last_update: float, now: float) -> Optional[float]:
        """Seconds to sleep this tick to honor common.max_frequency, or None if
        unbounded (max_frequency <= 0).

        max_frequency is one universal knob, no per-node special-casing: <= 0
        means "run as fast as the tick loop is driven". A free-running producer
        (autotrigger) sets a finite cap so its loop can't busy-spin; an
        input-triggered consumer leaves it unbounded (the default) so it runs at
        its producer's rate — it blocks on the WaitSet between frames, so
        unbounded never busy-spins there. Returns the seconds left in the tick
        period; <= 0 means we're already behind and won't sleep, but the caller
        still advances last_update.
        """
        max_freq = self.params.common.max_frequency.value
        if max_freq <= 0:
            return None
        period = 1.0 / max_freq if self.params.common.frequency_mode.value == "updates-per-second" else max_freq
        return period - (now - last_update)

    @staticmethod
    def _frame_count(value) -> int:
        """Number of frames a value spans, for index-timeline matching. Uses the
        total element count (numpy `.size`), so a length-preserving transform's
        output matches its input and a length-changing one (resampler, PSD, a
        generator's scalar control input) does not — regardless of dimensionality."""
        n = getattr(value, "size", None)  # numpy array
        if n is not None:
            return int(n)
        try:
            return len(value)
        except TypeError:
            return 1

    def _propagated_index(self, output_frames: int) -> Optional[int]:
        """Source-origin index to propagate this tick, or None.

        Propagate the index of the SINGLE input whose frame count equals this
        output's — i.e. the input that is genuinely the same timeline (a length-
        preserving pass-through like Math forwards its source's numbering, so an
        upstream drop stays visible at the sink). Any other case starts a fresh
        per-output counter: a generator (no matching input), a control input of a
        different length (an oscillator's scalar frequency), a length-changing
        transform (resampler/PSD), or an ambiguous >1 matching inputs. Frame count,
        not a per-slot flag, is the discriminator — index is always present on a
        connected input, so presence alone never distinguishes timeline from control.
        """
        found: Optional[int] = None
        count = 0
        for slot in self.input_slots.values():
            data = slot.data
            if data is None or INDEX_META_KEY not in data.meta:
                continue
            if self._frame_count(data.data) != output_frames:
                continue
            count += 1
            found = data.meta[INDEX_META_KEY]
        return found if count == 1 else None

    def _next_index(self, slot_name: str, output_frames: int) -> int:
        """meta["index"] for one emit on `slot_name` (see `_propagated_index`).

        Advances the per-output counter only in the fresh case; propagation
        just mirrors the matching input's value.
        """
        propagated = self._propagated_index(output_frames)
        if propagated is not None:
            return propagated
        idx = self._index_counters.get(slot_name, 0)
        self._index_counters[slot_name] = idx + 1
        return idx

    def _build_output_data(self, slot_name: str, slot: OutputSlot, value: Tuple) -> Data:
        """Build one output slot's outgoing Data and stamp its continuity
        index (propagate-or-fresh, §6) BEFORE the node fan-out encode and the
        viewer offer, so every node consumer and the browser viewer see one
        value. The scalar plus the constructor's shallow `dict(self.meta)` copy
        (data.py:79) keep it un-aliased across fan-out.
        """
        data = Data(slot.dtype, value[0], value[1])
        data.meta[INDEX_META_KEY] = self._next_index(slot_name, self._frame_count(value[0]))
        return data

    # ------------------------------------------------------------------
    # Standalone direct call
    # ------------------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._environment != NodeEnv.STANDALONE:
            raise RuntimeError("Only nodes running in standalone mode can be called directly.")
        if not self._setup_done.is_set():
            self.setup()
            self._setup_done.set()
        args = list(args)
        for i in range(len(args)):
            if not isinstance(args[i], Data):
                args[i] = to_data(args[i])
        for key, value in kwargs.items():
            if not isinstance(value, Data):
                kwargs[key] = to_data(value)
        return self.process(*args, **kwargs)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @classmethod
    def _configure(cls) -> Tuple[Dict[str, InputSlot], Dict[str, OutputSlot], NodeParams]:
        return normalize_config(cls.config_input_slots(), cls.config_output_slots(), cls.config_params())

    def common_autotrigger_changed(self, value):
        # No-op: autotrigger is consulted on each tick in the processing loop.
        pass

    def _apply_gc_policy(self) -> None:
        """One-time process-global GC/scheduling policy, applied after warm-up.

        Freezing the import/setup heap out of every future gen2 scan removes the
        ~60 ms collection tail (measured); the switch-interval bump cuts GIL
        hand-off jitter. On a node the user marked `realtime`, GC is disabled
        outright and the processing thread asks the kernel for SCHED_FIFO —
        best-effort, silently degrading when the process lacks rtprio permission
        (or on a platform without it). Guarded so it runs exactly once and never
        blocks `ready`/`_setup_done`.
        """
        if self._gc_policy_applied:
            return
        self._gc_policy_applied = True
        # Only safe when the node OWNS its process. In the co-located LOCAL
        # environments — the whole graph inside the manager+bridge process under
        # --no-multiprocessing, or a shared common.process_group host — freezing/
        # disabling GC or grabbing SCHED_FIFO would corrupt the bridge and every
        # sibling node. Latch first (so the loop stops retrying), then bail.
        if self._environment != NodeEnv.MULTIPROCESSING:
            return
        gc.freeze()
        sys.setswitchinterval(0.001)
        if self.params.common.priority.value == "realtime":
            gc.disable()
            try:
                os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(_SCHED_FIFO_PRIORITY))
            except (OSError, AttributeError, ValueError):
                # PermissionError (no rtprio) or unsupported platform: stay
                # collected-but-frozen. Rare clicks under load are accepted.
                pass

    def clear_error(self) -> None:
        self._report_error(None)

    def request_shutdown(self) -> None:
        if self._environment == NodeEnv.STANDALONE:
            raise RuntimeError(
                "Shutdown requested but the node is not running in standalone mode. "
                "A manager is required to perform the shutdown."
            )
        try:
            self.status_pub.send(encode_message(Message(MessageType.SHUTDOWN, {})))
            self._status_notifier.notify()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        node_id: Optional[str] = None,
        initial_params: Optional[Dict[str, Dict[str, Any]]] = None,
        capture_logs: bool = False,
    ) -> NodeRef:
        """Spawn the node in its own process and return a `NodeRef`.

        Thin wrapper over `spawn_node` for callers that already hold the class
        (tests, tooling). Group routing is the manager's responsibility, not
        this method's — callers that want to host a node inside a shared group
        process use `NodeProcessRegistry.spawn()` directly.
        """
        if cls.NO_MULTIPROCESSING:
            raise MultiprocessingForbiddenError("Multiprocessing is forbidden for this node.")

        from goofi.registry import NodeSpec

        return spawn_node(
            NodeSpec.from_class(cls),
            node_id=node_id,
            initial_params=initial_params,
            capture_logs=capture_logs,
        )

    @classmethod
    def create_local(
        cls,
        node_id: Optional[str] = None,
        initial_params: Optional[Dict[str, Dict[str, Any]]] = None,
        capture_logs: bool = False,
    ) -> Tuple[NodeRef, "Node"]:
        """Instantiate the node in the current process. Returns (ref, node)."""
        if node_id is None:
            node_id = f"{cls.__name__}-{uuid.uuid4().hex[:8]}"
        in_slots, out_slots, params = cls._configure()
        if initial_params is not None:
            try:
                params.update(initial_params)
            except InvalidParamError:
                print(
                    "\n====================== ERROR ======================\n"
                    f"Setting parameters failed for {cls.__name__}."
                    "\n===================================================\n"
                )

        ref = cls._build_ref(node_id, params, in_process=True, process=None)
        node = cls(
            node_id,
            in_slots,
            out_slots,
            params,
            NodeEnv.LOCAL,
            in_process_with_manager=True,
            capture_logs=capture_logs,
        )
        return ref, node

    @classmethod
    def _spawn_local(
        cls,
        node_id: str,
        initial_params: Optional[Dict[str, Dict[str, Any]]],
        capture_logs: bool = False,
    ) -> "Node":
        """Used by `NodeProcess` to create a node inside a shared group process."""
        in_slots, out_slots, params = cls._configure()
        if initial_params is not None:
            try:
                params.update(initial_params)
            except InvalidParamError:
                pass
        # In a group host process, the manager is in a *different* process,
        # so ctrl/status still go over iceoryx2 — but data plane between
        # group peers uses thread transport (decided at SUBSCRIBE_INPUT
        # time by the manager).
        return cls(
            node_id,
            in_slots,
            out_slots,
            params,
            NodeEnv.LOCAL,
            in_process_with_manager=False,
            capture_logs=capture_logs,
        )

    @classmethod
    def _build_ref(
        cls,
        node_id: str,
        params: NodeParams,
        in_process: bool,
        process: Optional[Process],
    ) -> NodeRef:
        from goofi.registry import NodeSpec

        return ref_from_spec(NodeSpec.from_class(cls), node_id, params, in_process, process)

    @classmethod
    def create_standalone(cls) -> "Node":
        in_slots, out_slots, params = cls._configure()
        return cls(None, in_slots, out_slots, params, NodeEnv.STANDALONE)

    @classmethod
    def category(cls) -> str:
        return cls.__module__.split(".")[-2]

    @classmethod
    def docstring(cls) -> str:
        return clean_docstring(cls.__doc__)

    @property
    def assets_path(self) -> Path:
        return Path(str(pkg_resources.files(assets)))

    @property
    def alive(self) -> bool:
        return self._alive

    @property
    def params(self) -> NodeParams:
        return self._params

    @property
    def input_slots(self) -> Dict[str, InputSlot]:
        return self._input_slots

    @property
    def output_slots(self) -> Dict[str, OutputSlot]:
        return self._output_slots

    # ------------------------------------------------------------------
    # Subclass override hooks
    # ------------------------------------------------------------------

    @staticmethod
    def config_input_slots() -> Dict[str, Union[InputSlot, DataType]]:
        return {}

    @staticmethod
    def config_output_slots() -> Dict[str, Union[OutputSlot, DataType]]:
        return {}

    @staticmethod
    def config_params() -> Dict[str, Dict[str, Any]]:
        return {}

    def setup(self) -> None:
        pass

    @abstractmethod
    def process(self, **kwargs) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        pass

    def terminate(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Spec-based spawning — the manager-side path (no class object, no import)
# ---------------------------------------------------------------------------


def ref_from_spec(
    spec,
    node_id: str,
    params: NodeParams,
    in_process: bool,
    process: Optional[Process],
) -> NodeRef:
    """Build the manager-side proxy purely from registry metadata."""
    in_slots, out_slots, _ = spec.configure()
    return NodeRef(
        node_id=node_id,
        in_process=in_process,
        input_slots={name: slot.dtype for name, slot in in_slots.items()},
        output_slots={name: slot.dtype for name, slot in out_slots.items()},
        params=params,
        spec=spec,
        process=process,
    )


def spawn_node(
    spec,
    node_id: Optional[str] = None,
    initial_params: Optional[Dict[str, Dict[str, Any]]] = None,
    capture_logs: bool = False,
) -> NodeRef:
    """Spawn a node in its own process from a NodeSpec. The implementation
    module is imported INSIDE the child — the calling process stays
    import-free. A bootstrap failure (import/construct) is reported over the
    ref's one-shot `boot_conn` pipe; the supervisor turns that into a
    permanent error instead of a restart loop."""
    if node_id is None:
        node_id = f"{spec.type}-{uuid.uuid4().hex[:8]}"

    in_slots, out_slots, params = spec.configure()
    if initial_params is not None:
        try:
            params.update(initial_params)
        except InvalidParamError:
            print(
                f"\n====================== ERROR ======================\n"
                f"Setting parameters failed for {spec.type}. This is likely due to updates"
                " in the node's code. Make sure to reconfigure the node."
                "\n===================================================\n"
            )

    from goofi.transport import get_instance_id

    instance_id = get_instance_id()

    recv_conn, send_conn = Pipe(duplex=False)
    # Open the manager-side ref — and with it the STATUS SUBSCRIBER — BEFORE
    # starting the child. iceoryx2 keeps no history: a subscriber opened after
    # the child's first STATE_UPDATE loses it, and the pre-ready ctrl queue
    # would then withhold every message, deadlocking the node in 'creating'
    # (nothing ever dirties it into a re-push).
    ref = ref_from_spec(spec, node_id, params, in_process=False, process=None)
    ref.boot_conn = recv_conn

    # Retry the os.fork()/proc.start() a few times — a transient fork failure
    # under load is worth another attempt before surfacing a bootstrap error.
    MAX_FORK_RETRIES = 3
    tries = 0
    while True:
        try:
            proc = Process(
                target=_run_node_process,
                # Feed the child the caller's RAW overrides, not params.serialize():
                # spec.configure() ran the AST hooks, whose dynamic fallback (device
                # lists, self-class refs) is a placeholder that would clobber the
                # child's real _configure() default. The child applies real config +
                # these overrides; the manager-side ref then converges to the child's
                # authoritative values via the STATE_UPDATE echo. Mirrors the group /
                # local paths, which already pass initial_params through.
                args=(spec.module, spec.cls_name, node_id, initial_params, instance_id, capture_logs, send_conn),
                daemon=True,
            )
            proc.start()
            break
        except Exception as e:
            tries += 1
            if tries >= MAX_FORK_RETRIES:
                try:
                    ref.terminate()
                except Exception:
                    pass
                raise e
            time.sleep(0.1)
    # The child holds the write end; dropping ours makes its EOF observable.
    send_conn.close()

    ref.process = proc
    return ref


# ---------------------------------------------------------------------------
# Process entrypoint for `spawn_node()` / `Node.create()`
# ---------------------------------------------------------------------------


def _run_node_process(
    module: str,
    cls_name: str,
    node_id: str,
    initial_params: Optional[Dict[str, Dict[str, Any]]],
    instance_id: str,
    capture_logs: bool,
    err_conn,
) -> None:
    """Child main: import the implementation HERE, then construct the node.
    A failure before the node's first state push is reported over `err_conn`
    (the manager can't hear us on iceoryx2 yet — endpoints don't exist)."""
    set_instance_id(instance_id)
    if capture_logs:
        # This process hosts exactly one node — attribute every thread (incl.
        # ones the node spawns itself) to it by default.
        node_log.set_process_default_node(node_id)
    try:
        import importlib

        cls = getattr(importlib.import_module(module), cls_name)
        in_slots, out_slots, params = cls._configure()
        if initial_params is not None:
            try:
                params.update(initial_params)
            except InvalidParamError:
                pass
        cls(
            node_id,
            in_slots,
            out_slots,
            params,
            NodeEnv.MULTIPROCESSING,
            in_process_with_manager=False,
            capture_logs=capture_logs,
        )
    except Exception:
        # Truncate head+tail so an oversized traceback (an exception embedding a
        # large payload) can't exceed the ~64 KiB pipe capacity and block the
        # dying child forever in write() — which would wedge the node in
        # 'creating' with no error surfaced. Keep both ends: the frame chain
        # (head) and the raising frame + exception type/message (tail).
        tb = traceback.format_exc()
        if len(tb) > 16384:
            tb = tb[:8192] + "\n... [traceback truncated] ...\n" + tb[-8192:]
        try:
            err_conn.send(tb)
        finally:
            err_conn.close()
        raise SystemExit(1)
