"""External nd() references from shared sub-patch members (review findings #1, #2).

A shared sub-patch member may reference a node OUTSIDE the sub-patch by nd('name').
The definition stores INTERNAL (intra-patch) refs in template form and EXTERNAL refs
verbatim. Two hazards the flat-name + template-local design must defend against:

  #1 An external ref must not be hijacked by the instantiate-time local->display
     rewrite when the external name happens to equal a member's local template key.
  #2 Renaming an external node must update the ref stored in the shared definition
     (the round-trip source of truth), not just the live members.
"""
from goofi.manager import ROOT_ID

from .test_manager import _bare_manager


def test_external_nd_ref_survives_local_key_collision_on_instantiate():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")  # oscillator0
        b = mgr.add_node("Oscillator", "inputs")  # oscillator1
        inst = mgr.group_nodes([a, b])
        la = mgr._instances[inst].members[a]  # local key 'oscillator0'
        lb = mgr._instances[inst].members[b]
        # free the display name 'oscillator0' (a's local key stays the same)
        mgr.rename_node(a, "carrier")
        assert mgr._instances[inst].members[a] == la
        # an EXTERNAL node now takes the freed name, equal to a's local KEY
        ext = mgr.add_node("Oscillator", "inputs")  # 'oscillator0'
        assert mgr.nodes[ext].name == la == "oscillator0"
        # member b references the EXTERNAL node
        mgr.set_expression(b, "oscillator", "frequency", "nd('oscillator0')", enabled=True)

        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)
        b2 = mgr._member_uid(inst2, lb)
        # b2 must STILL reference the external oscillator0 — NOT inst2's internal copy
        # of member a (whose local key collides with the external name).
        assert mgr.nodes[b2].params["oscillator"]["frequency"].expression == "nd('oscillator0')"
    finally:
        mgr.terminate(notify_gui=False)


def test_rename_external_node_updates_shared_definition_ref(tmp_path):
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        ext = mgr.add_node("Oscillator", "inputs")  # external producer
        b = mgr.add_node("Oscillator", "inputs")  # becomes a member
        mgr.set_expression(b, "oscillator", "frequency", f"nd('{mgr.nodes[ext].name}')", enabled=True)
        inst = mgr.group_nodes([b])
        lb = mgr._instances[inst].members[b]
        def_id = mgr.share_instance(inst)

        mgr.rename_node(ext, "extRenamed")
        # the live member followed the rename...
        assert mgr.nodes[b].params["oscillator"]["frequency"].expression == "nd('extRenamed')"
        # ...and so does a freshly instantiated sibling (the DEFINITION was updated).
        inst2 = mgr.instantiate_definition(def_id)
        b2 = mgr._member_uid(inst2, lb)
        assert mgr.nodes[b2].params["oscillator"]["frequency"].expression == "nd('extRenamed')"

        fp = str(tmp_path / "ext.gfi")
        mgr.save(fp, overwrite=True)
    finally:
        mgr.terminate(notify_gui=False)

    mgr2 = _bare_manager(use_multiprocessing=False)
    try:
        mgr2.load(fp)
        for iid in [i for i in mgr2._instances if i != ROOT_ID]:
            bu = mgr2._member_uid(iid, lb)
            assert mgr2.nodes[bu].params["oscillator"]["frequency"].expression == "nd('extRenamed')"
    finally:
        mgr2.terminate(notify_gui=False)
