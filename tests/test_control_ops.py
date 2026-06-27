"""The bridge control ops added for Phase-1 save/load (no live socket)."""
import asyncio
import types
from pathlib import Path

from goofi.bridge.control import ControlHub
from goofi.message import MessageType

from .test_manager import _bare_manager


def _hub(manager=None) -> ControlHub:
    # ControlHub only needs `server.manager` for these ops; no real server.
    # `schedule` swallows the broadcast coroutine (no loop in these tests).
    server = types.SimpleNamespace(manager=manager, schedule=lambda coro: coro.close())
    return ControlHub(server)


def test_list_dir_op(tmp_path: Path):
    (tmp_path / "x.gfi").write_text("x")
    hub = _hub()
    res = asyncio.run(hub._dispatch("list_dir", {"path": str(tmp_path)}))
    assert res["path"] == str(tmp_path.resolve())
    assert any(e["name"] == "x.gfi" for e in res["entries"])


def test_list_examples_op():
    hub = _hub()
    res = asyncio.run(hub._dispatch("list_examples", {}))
    assert "entries" in res


def test_serialize_op_calls_manager():
    class FakeMgr:
        def serialize_patch(self):
            return "nodes: {}\nlinks: []\n"

    hub = _hub(FakeMgr())
    res = asyncio.run(hub._dispatch("serialize", {}))
    assert res["yaml"].startswith("nodes:")


def test_add_node_dispatch_forwards_member_uid():
    """Redo-of-add and undo-of-delete re-create a node with its ORIGINAL uid
    (sent as `member_uid`) so uid-keyed links and panel bindings reconnect. The
    bridge must forward member_uid to the manager; dropping it mints a fresh uid
    and orphans the captured links (KeyError on the replayed add_link)."""
    manager = _bare_manager()
    manager._layout = None
    try:
        hub = _hub(manager)
        want = "deadbeef0001"
        got = asyncio.run(
            hub._dispatch("add_node", {"type": "Oscillator", "category": "inputs", "member_uid": want})
        )
        assert got == want, f"member_uid was not honored (bridge dropped it): {got!r}"
        assert want in manager.nodes
    finally:
        manager.terminate(notify_gui=False)


def test_load_rewires_state_forwarding_for_reused_names(tmp_path: Path):
    """Report B1: a destructive reload tears down old nodes with notify_gui=False
    (so _wired_nodes is never cleared), then re-adds nodes with the SAME display
    names — _wire_node_status then early-returns and the reloaded nodes never get
    their STATE_UPDATE/PROCESSING_ERROR handlers. Live forwarding silently dies."""
    manager = _bare_manager()
    manager._layout = None
    try:
        hub = _hub(manager)
        manager._bridge = types.SimpleNamespace(control=hub)

        osc = manager.add_node("Oscillator", "inputs")
        assert MessageType.STATE_UPDATE in manager.nodes[osc].callbacks
        assert osc in hub._wired_nodes

        gfi = tmp_path / "patch.gfi"
        manager.save(str(gfi))

        # Destructive reload of the same patch reuses the node's display name.
        asyncio.run(hub._dispatch("load", {"path": str(gfi)}))

        assert osc in manager.nodes, "node not reloaded under the same name"
        # The reloaded node's NEW NodeRef must have its forwarding handlers back.
        assert MessageType.STATE_UPDATE in manager.nodes[osc].callbacks, "state forwarding lost after reload (B1)"
        assert MessageType.PROCESSING_ERROR in manager.nodes[osc].callbacks
    finally:
        manager.terminate(notify_gui=False)
