"""restart_node must not orphan the freshly-spawned process if the node is deleted
mid-restart.

restart_node holds _supervisor_lock, but remove_node takes no lock, so a user
deleting a crashed node during the (real, slow) spawn window makes
NodeContainer.replace(uid, new_ref) raise KeyError — after the new process has
already started. The new ref is only a local until replace() stores it, so an
un-handled KeyError leaks a running daemon process for the manager's lifetime.
"""
import pytest

from .test_manager import _bare_manager


class _FakeNew:
    """Stands in for the freshly-spawned NodeRef; records terminate()."""

    def __init__(self):
        self.terminated = False

    def set_message_handler(self, *a, **k):
        pass

    def terminate(self):
        self.terminated = True


def test_restart_node_terminates_the_new_process_if_the_uid_was_removed():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        fake = _FakeNew()
        mgr._spawn_node = lambda *a, **k: fake  # don't really spawn
        # Simulate remove_node deleting the uid during the spawn window: replace() raises.
        mgr.nodes.replace = lambda *a, **k: (_ for _ in ()).throw(KeyError("gone"))

        with pytest.raises(KeyError):
            mgr.restart_node(a)

        assert fake.terminated, "restart_node orphaned the freshly-spawned process on a delete race"
    finally:
        mgr.terminate(notify_gui=False)
