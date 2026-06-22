"""_service_budget_ok checks the node's REAL slot names, not a synthetic guess (backlog #14)."""
from .test_manager import _bare_manager


def test_service_budget_uses_real_slot_lengths():
    mgr = _bare_manager()
    try:
        # a normal short slot fits the 255-byte service-name budget
        assert mgr._service_budget_ok("node0", slots=["out"]) is True
        # a pathologically long REAL slot is rejected (a 48-char guess would mask it)
        assert mgr._service_budget_ok("node0", slots=["x" * 220]) is False
        # no slots given → conservative default still works
        assert mgr._service_budget_ok("node0") is True
    finally:
        mgr.terminate(notify_gui=False)
