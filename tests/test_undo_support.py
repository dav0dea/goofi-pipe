"""Backend contracts the browser undo/redo relies on.

The undo system lives entirely in the frontend (see
``docs/superpowers/specs/2026-06-19-undo-redo-redesign-design.md``). It needs
NO new backend ops — undo-of-delete restores a node by re-adding it under its
*reused display name*, and the bridge ``add_node`` op already accepts ``name``
and ``params``. These tests pin the two backend behaviours that frontend
contract depends on, so a future backend change can't silently break undo:

  1. a freed display name is reused (so undo-of-delete restores identity); and
  2. forcing an already-occupied name raises (so undo-of-delete fails loudly
     rather than corrupting the graph — the frontend surfaces a toast).

Gate decision (plan Phase 3.4): because both hold against the current backend,
the optional bridge changes O1 (member_uid passthrough), O2 (displaced_link in
the add_link response) and O3 (group_nodes rename metadata) are NOT needed for
the core feature and are deferred (spec §7, §11).
"""

import pytest

from goofi.manager import NodeContainer

from .utils import DummyNode


def test_auto_assigned_display_name_is_reused_after_remove():
    """Delete ``dummy0`` and the next auto-named node takes ``dummy0`` again.

    Undo-of-delete re-adds the node under the exact name it had; this reuse is
    what lets links and panel bindings (keyed by display name) reconnect.
    """
    cont = NodeContainer()
    n0 = cont.add_node("dummy", DummyNode.create_local()[0])
    assert n0 == "dummy0"
    n1 = cont.add_node("dummy", DummyNode.create_local()[0])
    assert n1 == "dummy1"

    cont.remove_node("dummy0")
    # The freed slot is reused, not skipped to dummy2.
    reused = cont.add_node("dummy", DummyNode.create_local()[0])
    assert reused == "dummy0"
    for name in ("dummy0", "dummy1"):
        cont[name].terminate()


def test_force_name_restores_exact_freed_name():
    """``force_name=True`` (the path the bridge uses for an explicit ``name``)
    stores the requested name verbatim when it is free — so undo can put the
    node back as e.g. ``oscillator0``."""
    cont = NodeContainer()
    n0 = cont.add_node("dummy", DummyNode.create_local()[0])  # -> dummy0
    cont.remove_node(n0)
    restored = cont.add_node("dummy0", DummyNode.create_local()[0], force_name=True)
    assert restored == "dummy0"
    cont["dummy0"].terminate()


def test_force_name_collision_raises():
    """Forcing a name that is already taken raises — the frontend relies on this
    to fail an undo atomically (and toast) instead of clobbering a live node."""
    cont = NodeContainer()
    cont.add_node("dummy", DummyNode.create_local()[0])  # occupies dummy0
    with pytest.raises(KeyError):
        cont.add_node("dummy0", DummyNode.create_local()[0], force_name=True)
    cont["dummy0"].terminate()
