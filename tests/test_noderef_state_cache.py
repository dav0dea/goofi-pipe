"""NodeRef is the SINGLE manager-side cache of node-pushed truth.

A node pushes STATE_UPDATE / PROCESSING_ERROR on the status plane; the NodeRef
messaging loop applies it to the cache at ONE place (`_dispatch_status`) and then
runs any registered side-effect handler (the bridge broadcast). Previously the
apply was duplicated: the loop did it on the default path, and a registered
handler "re-did the work" — two apply sites that had to stay in sync. These tests
pin the unified contract via an isolated fake (no transport needed).
"""
import threading
from types import SimpleNamespace

from goofi.message import Message, MessageType
from goofi.node_helpers import NodeRef


def _fake_ref(callbacks=None, updates=None):
    return SimpleNamespace(
        serialized_state=None,
        last_error=None,
        node_stats=None,
        stage="creating",
        _first_state_event=threading.Event(),
        _ctrl_queue_lock=threading.RLock(),
        _pending_ctrl=[],
        params=SimpleNamespace(update=lambda d: (updates if updates is not None else []).append(d)),
        callbacks=callbacks or {},
    )


def test_dispatch_status_applies_state_then_fires_callback():
    fired = []
    updates = []
    fake = _fake_ref(callbacks={MessageType.STATE_UPDATE: lambda ref, m: fired.append(m)}, updates=updates)

    content = {"_type": "X", "category": "test", "params": {"grp": {"a": 1}}, "output_subscribers": {}}
    NodeRef._dispatch_status(fake, Message(MessageType.STATE_UPDATE, content))

    # cache updated at the single apply site...
    assert fake.serialized_state == content
    assert fake._first_state_event.is_set()
    assert updates == [{"grp": {"a": 1}}]
    # ...AND the side-effect handler (broadcast) still ran exactly once
    assert len(fired) == 1


def test_dispatch_status_processing_error_sets_last_error_then_fires():
    fired = []
    fake = _fake_ref(callbacks={MessageType.PROCESSING_ERROR: lambda ref, m: fired.append(m)})

    NodeRef._dispatch_status(fake, Message(MessageType.PROCESSING_ERROR, {"error": "boom"}))

    assert fake.last_error == "boom"
    assert len(fired) == 1


def test_dispatch_status_applies_even_without_a_callback():
    # Headless / not-yet-wired: the cache must still update with no handler set.
    updates = []
    fake = _fake_ref(updates=updates)
    content = {"_type": "X", "category": "test", "params": {"g": {}}, "output_subscribers": {}}
    NodeRef._dispatch_status(fake, Message(MessageType.STATE_UPDATE, content))
    assert fake.serialized_state == content
    assert updates == [{"g": {}}]


def test_dispatch_status_node_stats_caches_then_fires():
    # NODE_STATS is node-owned telemetry: it lands in the cache at the single
    # apply site (so a synchronous reader / snapshot sees it) and then the bridge
    # broadcast handler runs.
    fired = []
    fake = _fake_ref(callbacks={MessageType.NODE_STATS: lambda ref, m: fired.append(m)})
    stats = {"updates_per_second": 12.4, "mean_process_ms": 3.1, "total_ticks": 42}

    NodeRef._dispatch_status(fake, Message(MessageType.NODE_STATS, {"stats": stats}))

    assert fake.node_stats == stats
    assert len(fired) == 1
    # stats are independent of state/error caches
    assert fake.serialized_state is None
    assert fake.last_error is None


def test_dispatch_status_shutdown_only_dispatches_callback():
    fired = []
    fake = _fake_ref(callbacks={MessageType.SHUTDOWN: lambda ref, m: fired.append(m)})
    NodeRef._dispatch_status(fake, Message(MessageType.SHUTDOWN, {}))
    assert len(fired) == 1
    assert fake.serialized_state is None  # SHUTDOWN touches no cache
