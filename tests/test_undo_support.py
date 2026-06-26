"""Backend contracts the browser undo/redo relies on.

The undo system lives entirely in the frontend (see
``docs/superpowers/specs/2026-06-19-undo-redo-redesign-design.md``). It needs
NO new backend ops — undo-of-delete restores a node by re-adding it under its
*stable uid* (``member_uid`` passthrough), and the bridge ``add_node`` op already
accepts ``name``, ``params`` and ``member_uid``. These tests pin the backend
behaviours that contract depends on:

  1. the stable uid is preserved across a delete→re-add (so links, panel
     bindings and selection — all keyed by uid — reconnect to the same node); and
  2. the auto-numbered display name is reused after a delete (a UX nicety; the
     name is display-only and no longer an identity).
"""
from .test_manager import _bare_manager


def test_uid_is_preserved_across_delete_and_readd():
    """Undo-of-delete re-adds the node with its captured uid. Because everything
    keys on the uid, the restored node IS the same node to links/selection."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        uid = mgr.add_node("Oscillator", "inputs")
        name = mgr.nodes[uid].name
        mgr.remove_node(uid)
        # Undo passes the captured uid back through add_node (member_uid).
        restored = mgr.add_node("Oscillator", "inputs", name=name, member_uid=uid)
        assert restored == uid  # exact identity restored
        assert mgr.nodes[uid].name == name
    finally:
        mgr.terminate(notify_gui=False)


def test_auto_assigned_display_name_is_reused_after_remove():
    """Delete ``oscillator0`` and the next auto-named Oscillator takes the name
    again — a display nicety, independent of the (always-fresh) uid."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        assert mgr.nodes[a].name == "oscillator0"
        b = mgr.add_node("Oscillator", "inputs")
        assert mgr.nodes[b].name == "oscillator1"

        mgr.remove_node(a)
        # The freed display name is reused, not skipped to oscillator2.
        c = mgr.add_node("Oscillator", "inputs")
        assert mgr.nodes[c].name == "oscillator0"
        assert c != a  # ...but it is a brand-new node (fresh uid)
    finally:
        mgr.terminate(notify_gui=False)


def test_duplicate_member_uid_is_reminted():
    """Passing a uid that's already live mints a fresh one instead of colliding —
    the container's single uid index can never hold two nodes under one key."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Oscillator", "inputs", member_uid=a)  # a is still live
        assert b != a and b in mgr.nodes and a in mgr.nodes
    finally:
        mgr.terminate(notify_gui=False)
