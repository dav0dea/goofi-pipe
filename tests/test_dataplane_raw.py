"""NodeRef view-plane data handler contract (Option C).

`set_data_handler(slot, cb)` subscribes to the node's REDUCED `.view` stream,
registers the manager as a *viewer* (REGISTER_VIEWER), and hands the pump's
node-reduced GOOF wire bytes straight to `cb(noderef, slot, buf)` — forwarded to
the browser verbatim, with no manager-side decode/re-encode (A1). `cb=None`
unregisters (UNREGISTER_VIEWER + endpoint release).
"""
import time

import pytest

from .test_manager import _bare_manager


def _wait_until(pred, timeout: float) -> None:
    deadline = time.time() + timeout
    while not pred() and time.time() < deadline:
        time.sleep(0.02)


def test_unregister_closes_subscriber_and_listener():
    """report B4: unregistering a data handler must release BOTH the iceoryx2
    subscriber and its listener — previously only the subscriber was closed,
    leaking a listener per subscribe/unsubscribe cycle."""
    mgr = _bare_manager()
    try:
        osc = mgr.add_node("Oscillator", "inputs")
        ref = mgr.nodes[osc]
        ref.wait_for_state(timeout=2.0)

        ref.set_data_handler("out", lambda *_a: None)
        sub, listener = ref._data_handlers["out"][0], ref._data_handlers["out"][1]
        ref.set_data_handler("out", None)

        # close() nulls the inner handle on both IPC endpoints.
        assert sub._sub is None, "subscriber not closed on unregister"
        assert listener._listener is None, "listener leaked on unregister"
    finally:
        mgr.terminate(notify_gui=False)


def test_view_subscriber_fault_leaves_no_stranded_viewer_registration():
    """A `.view` subscriber-build fault must raise BEFORE REGISTER_VIEWER is sent,
    so the node isn't left with a bumped viewer_count reducing into a subscriber the
    manager never opened. The bridge connect-path rollback (set_data_handler(None))
    can't undo a register it never observed — it no-ops on the empty handler map —
    so the ordering has to make the register unreachable when the subscribe fails."""
    mgr = _bare_manager()
    try:
        osc = mgr.add_node("Oscillator", "inputs")
        ref = mgr.nodes[osc]
        ref.wait_for_state(timeout=2.0)

        events: list = []
        ref.register_viewer = lambda s: events.append(("reg", s))
        ref.unregister_viewer = lambda s: events.append(("unreg", s))

        def _boom(_slot):
            raise RuntimeError("simulated .view subscriber build failure")

        ref.open_view_subscriber = _boom

        with pytest.raises(RuntimeError):
            ref.set_data_handler("out", lambda *_a: None)

        regs = [e for e in events if e[0] == "reg"]
        unregs = [e for e in events if e[0] == "unreg"]
        assert len(regs) == len(unregs), f"stranded viewer registration: {events}"
        assert "out" not in ref._data_handlers
    finally:
        mgr.terminate(notify_gui=False)
