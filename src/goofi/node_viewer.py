"""Node-side viewer reducer subsystem (Option C node-side reduction).

One ``goofi-data-reducer`` daemon thread per host process. The node's processing
loop calls :func:`offer` (O(1): a private snapshot taken on the node thread plus a
latest-wins enqueue). The reducer thread reduces each slot's latest snapshot to
its folded :class:`~goofi.node_reduce.ViewSpec`, encodes the small frame, and
hands the bytes to that slot's ``sink``. The node wires a sink that publishes on
the slot's ``.view`` iceoryx2 service; the manager subscribes to that service and
relays the reduced frames to browsers verbatim (no manager-side decode/re-encode).

Transport-agnostic by design: the sink is injected, so this module is fully
unit-testable without iceoryx2.

**Safety (load-bearing).** :func:`offer` snapshots on the NODE thread — a
contiguous copy of the array plus a copied meta — so the reducer never aliases
live node state. This is correct even for in-place mutators (e.g. ``LatentRotator``
``+=``/``/=`` on a persistent buffer it returns) and meta-reusers (e.g. ``pca``
returning ``self.meta``): the copy executes on the node thread, serialized with
that node's own mutations across ticks, so there is never a torn read and the
reducer can never corrupt or stall the node↔node path (which encodes the original
Data synchronously on the node thread, independently of this module).
"""
from __future__ import annotations

import threading
import traceback
from collections import deque
from typing import Callable, Deque, Dict, Optional, Set, Tuple

import numpy as np

from goofi.codec import encode_data
from goofi.data import Data
from goofi.node_reduce import ViewSpec, reduce_for_view

_SKey = Tuple[str, str]  # (node_id, slot_name)
Sink = Callable[[bytes], None]

# All shared state is guarded by `_cond` (a Condition over a single Lock). The
# reducer thread waits on `_cond`; producers (node + manager-driven messaging
# thread) notify it.
_cond = threading.Condition(threading.Lock())
_pending: Dict[_SKey, Data] = {}      # latest offered snapshot per slot (latest-wins)
_dirty: Deque[_SKey] = deque()        # slots needing reduction (round-robin fairness)
_dirty_set: Set[_SKey] = set()        # membership guard so a slot enqueues at most once
_specs: Dict[_SKey, ViewSpec] = {}    # folded ViewSpec per slot (last-received-wins)
_sinks: Dict[_SKey, Sink] = {}        # encoded-frame sink per slot
_in_flight: Optional[_SKey] = None    # slot whose sink the reducer is executing right now
_thread: Optional[threading.Thread] = None
_running = False


def _snapshot_for_offer(data: Data) -> Data:
    """Run on the NODE thread. Return a Data that aliases nothing the node retains:
    a contiguous COPY of the array + a copied meta (shallow + channels sub-dict).
    A TABLE body is a ``dict[str, Data]`` (no ``ndim``), so recurse into it — else
    the snapshot would share the node's live child arrays and tear under mutation."""
    arr = data.data
    meta = dict(data.meta)
    ch = meta.get("channels")
    if isinstance(ch, dict):
        meta["channels"] = {k: list(v) if isinstance(v, (list, tuple)) else v
                            for k, v in ch.items()}
    if hasattr(arr, "ndim"):           # ndarray -> contiguous private copy
        body = np.ascontiguousarray(arr).copy()
    elif isinstance(arr, dict):        # TABLE -> recurse so no child Data aliases live state
        body = {k: _snapshot_for_offer(v) if isinstance(v, Data) else v
                for k, v in arr.items()}
    else:                              # STRING / scalar -> immutable, safe to share
        body = arr
    return Data(data.dtype, body, meta)


def _ensure_thread_locked() -> None:
    """Start the single reducer thread if needed. Caller holds `_cond`."""
    global _thread, _running
    if _thread is None or not _thread.is_alive():
        _running = True
        _thread = threading.Thread(target=_reducer_loop, name="goofi-data-reducer", daemon=True)
        _thread.start()


def register_viewer(node_id: str, slot_name: str, sink: Sink) -> None:
    """Begin serving reduced frames for a slot. Called on the messaging thread on
    the first browser (REGISTER_VIEWER). `sink` receives each encoded reduced frame."""
    with _cond:
        _sinks[(node_id, slot_name)] = sink
        _ensure_thread_locked()


def set_viewspec(node_id: str, slot_name: str, spec: ViewSpec) -> None:
    """Set the folded ViewSpec for a slot (last-received-wins). Messaging thread."""
    with _cond:
        _specs[(node_id, slot_name)] = spec


def offer(node_id: str, slot_name: str, data: Data) -> None:
    """NODE-thread O(1) handoff: private snapshot + latest-wins enqueue + notify.
    Drops silently if no viewer is registered (raced teardown)."""
    skey = (node_id, slot_name)
    snap = _snapshot_for_offer(data)
    with _cond:
        if skey not in _sinks:
            return
        _pending[skey] = snap  # latest-wins
        if skey not in _dirty_set:
            _dirty.append(skey)
            _dirty_set.add(skey)
        _cond.notify()


def evict(node_id: str, slot_name: str) -> None:
    """Drop all state for a slot and BLOCK until the reducer is no longer executing
    its sink. Called when the last browser leaves (UNREGISTER_VIEWER 1->0) and at
    node teardown right before the slot's '.view' publisher is closed — quiescing
    here is what makes that close safe: removing the sink stops any NEW sink() call,
    and the wait drains an in-flight one, so the reducer can never loan/send on a
    publisher the caller is about to drop. The wait is bounded (the view sink is one
    loan+send+notify); _cond.wait releases the lock so the reducer's finally can run."""
    skey = (node_id, slot_name)
    with _cond:
        _sinks.pop(skey, None)  # no NEW sink() will start for this slot
        _specs.pop(skey, None)
        _pending.pop(skey, None)
        _dirty_set.discard(skey)  # discard + remove are absence-safe; no need to pre-check
        try:
            _dirty.remove(skey)
        except ValueError:
            pass
        while _in_flight == skey:  # drain a sink() already in progress for this slot
            _cond.wait()


def _reducer_loop() -> None:
    global _in_flight
    while True:
        with _cond:
            while _running and not _dirty:
                _cond.wait()
            if not _running:
                return
            skey = _dirty.popleft()
            _dirty_set.discard(skey)
            snap = _pending.pop(skey, None)
            spec = _specs.get(skey)
            sink = _sinks.get(skey)
            # Publish the in-flight slot UNDER the lock so evict() can quiesce: it
            # waits until we clear this before the caller closes the slot's pub.
            _in_flight = skey if (snap is not None and sink is not None) else None
        if snap is None or sink is None:
            continue
        try:
            reduced = reduce_for_view(snap, spec)
            buf = encode_data(reduced)
            sink(buf)
        except Exception:
            # Reduce/encode/sink errors are swallowed here so a bad frame for one
            # slot never kills the reducer or touches node state / node↔node.
            traceback.print_exc()
        finally:
            with _cond:
                _in_flight = None
                _cond.notify_all()  # wake any evict() quiescing on this slot


def _reset_for_tests() -> None:
    """Stop + join the reducer thread and clear all state. Test-fixture hook."""
    global _thread, _running, _in_flight
    with _cond:
        _running = False
        _in_flight = None
        _pending.clear()
        _dirty.clear()
        _dirty_set.clear()
        _specs.clear()
        _sinks.clear()
        _cond.notify_all()
    t = _thread
    if t is not None:
        t.join(timeout=2.0)
    _thread = None
