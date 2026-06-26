"""Node-process liveness supervision + auto-restart (manager-owned).

The manager spawns each default node in its own OS process. Before this, a crashed
process left the NodeRef looking healthy forever (no event, no recovery). The
supervisor polls process liveness and respawns a dead node with a FRESH transport
id, re-applying its params and re-wiring its links (both directions, since the new
id changes the node's service names).
"""
from types import SimpleNamespace

from goofi.manager import Manager
from goofi.message import MessageType

from .test_manager import _bare_manager


# ---- crash detection (pure; no real process needed) --------------------------

def test_node_is_dead_only_for_an_exited_own_process():
    alive = SimpleNamespace(process=SimpleNamespace(is_alive=lambda: True))
    dead = SimpleNamespace(process=SimpleNamespace(is_alive=lambda: False))
    local = SimpleNamespace(process=None)  # LOCAL / shared-group node: not supervised
    assert Manager._node_is_dead(alive) is False
    assert Manager._node_is_dead(dead) is True
    assert Manager._node_is_dead(local) is False


# ---- restart_node (real spawned nodes) ---------------------------------------

def test_restart_node_respawns_fresh_id_preserving_identity_and_links():
    mgr = _bare_manager(use_multiprocessing=True)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Buffer", "signal")
        mgr.add_link(a, b, "out", "val")

        old_ref = mgr.nodes[a]
        old_id = old_ref.node_id
        old_uid = old_ref.member_uid
        assert old_ref.process is not None  # one-node-per-process

        mgr.restart_node(a)

        new_ref = mgr.nodes[a]
        assert new_ref is not old_ref               # the ref object was replaced
        assert new_ref.node_id != old_id            # FRESH transport id (no service race)
        assert new_ref.member_uid == old_uid        # stable identity preserved
        assert mgr._refs_by_uid[old_uid] is new_ref  # uid index repointed
        assert new_ref.restart_count == 1           # observability
        # the link survived and was re-wired (no exception re-issuing it)
        assert {"node_out": a, "node_in": b, "slot_out": "out", "slot_in": "val"} in mgr._links
        assert new_ref.wait_for_state(timeout=3.0)  # the respawned node is live + pushing state
        # SHUTDOWN handler re-registered so the new node can still tear the manager down
        assert MessageType.SHUTDOWN in new_ref.callbacks
    finally:
        mgr.terminate(notify_gui=False)


def test_supervisor_detects_killed_process_and_restarts():
    # The end-to-end reliability contract: a node whose OS process is hard-killed
    # is detected dead and respawned by one supervisor sweep.
    mgr = _bare_manager(use_multiprocessing=True)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        old = mgr.nodes[a]
        old.wait_for_state(timeout=2.0)

        old.process.kill()  # hard-kill the worker process (SIGKILL)
        old.process.join(timeout=2.0)
        assert not old.process.is_alive()
        assert Manager._node_is_dead(mgr.nodes[a])  # the supervisor would detect this

        mgr._supervise_once()  # one liveness sweep

        new = mgr.nodes[a]
        assert new is not old
        assert new.restart_count == 1
        assert new.wait_for_state(timeout=3.0)  # the replacement is live and pushing state
    finally:
        mgr.terminate(notify_gui=False)


def test_restart_node_rewires_when_restarted_node_is_the_source():
    # Restarting the SOURCE changes its service names; the downstream must
    # re-subscribe to the new service. Exercises both-direction rewire.
    mgr = _bare_manager(use_multiprocessing=True)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Buffer", "signal")
        mgr.add_link(a, b, "out", "val")
        mgr.restart_node(a)  # a is the source of the only link
        # downstream ref unchanged, link intact, new source id wired
        assert {"node_out": a, "node_in": b, "slot_out": "out", "slot_in": "val"} in mgr._links
        assert mgr.nodes[b].wait_for_state(timeout=3.0)
    finally:
        mgr.terminate(notify_gui=False)
