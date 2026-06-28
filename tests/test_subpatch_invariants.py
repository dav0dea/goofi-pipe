"""Phase 0 safety net for the recursive-sub-patch redesign.

`assert_subpatch_invariants(mgr)` is the structural rail the later phases lean on:
it asserts the sub-patch state maps are internally consistent and agree with the
live graph. It is written against TODAY's single-level dict structure
(`_membership`/`_instances`/`_definitions`); Phase 1 re-points its *internals* at
the typed objects + indices, but the invariants it enforces (and its name) stay
stable so a regression in any later phase turns these tests red.

The scenario tests pin SEMANTIC outcomes that must survive the redesign (members
stay live + organized, links preserved, shared edits mirror, def GC, boundary
splice resolves) — deliberately NOT the `inst::local` naming or the
instance-id-equals-display-name detail, both of which the redesign intentionally
changes (flat global names; stable minted instance uids).
"""
import pytest

from .test_manager import _bare_manager, _build_grouped_graph, _member


def assert_subpatch_invariants(mgr) -> None:
    """Raise AssertionError if the sub-patch structure is internally inconsistent
    or out of sync with the live graph. Safe to call at rest after any operation."""
    nodes = set(mgr.nodes)
    instances = mgr._instances
    definitions = mgr._definitions
    membership = mgr._membership

    # --- every link endpoint is a live node ---------------------------------
    for link in mgr._links:
        assert link["node_out"] in nodes, f"link out endpoint {link['node_out']} is not a live node"
        assert link["node_in"] in nodes, f"link in endpoint {link['node_in']} is not a live node"

    # --- membership -> instances is consistent, members are live, marker agrees
    for uid, inst_id in membership.items():
        assert inst_id in instances, f"membership[{uid}] points at missing instance {inst_id}"
        members = instances[inst_id].members
        assert uid in members, f"{uid} maps to {inst_id} in membership but is not one of its members"
        assert uid in nodes, f"member {uid} of {inst_id} is not a live node"
        local = members[uid]
        assert mgr.nodes[uid].membership == {"instance": inst_id, "local_name": local}, (
            f"ref.membership marker on {uid} out of sync: {mgr.nodes[uid].membership!r} "
            f"!= {{'instance': {inst_id!r}, 'local_name': {local!r}}}"
        )

    # --- instances -> membership reverse, local-name uniqueness, def + boundary
    for inst_id, inst in instances.items():
        members = inst.members
        for uid, local in members.items():
            assert uid in nodes, f"instance {inst_id} lists member {uid} that is not live"
            assert membership.get(uid) == inst_id, (
                f"reverse membership mismatch: instance {inst_id} owns {uid} "
                f"but membership says {membership.get(uid)!r}"
            )
        locals_ = list(members.values())
        assert len(set(locals_)) == len(locals_), (
            f"duplicate local name within {inst_id}: {locals_}"
        )

        def_id = inst.def_id
        if def_id is not None:
            assert def_id in definitions, f"instance {inst_id} references missing definition {def_id}"
            assert set(members.values()) == set(definitions[def_id].members.keys()), (
                f"shared instance {inst_id} member set diverges from its definition {def_id}"
            )

        for bid, e in inst.interface.items():
            assert e.dir in ("in", "out"), f"boundary {inst_id}:{bid} has bad dir {e.dir!r}"
            inner = e.inner_node
            if inner is not None:
                assert inner in members.values(), (
                    f"boundary {inst_id}:{bid} inner_node {inner!r} is not a member local"
                )
                muid = _member(mgr, inst_id, inner)
                slots = mgr.nodes[muid].input_slots if e.dir == "in" else mgr.nodes[muid].output_slots
                assert e.inner_slot in slots, (
                    f"boundary {inst_id}:{bid} inner_slot {e.inner_slot!r} missing on member {inner!r}"
                )

    # --- definitions are GC'd: every def is referenced by >=1 instance --------
    referenced = {inst.def_id for inst in instances.values()} - {None}
    for def_id in definitions:
        assert def_id in referenced, f"orphan definition {def_id} was not garbage-collected"

    # --- shared siblings strict-mirror their member locals + boundary ids ------
    by_def: dict = {}
    for inst_id, inst in instances.items():
        d = inst.def_id
        if d:
            by_def.setdefault(d, []).append(inst_id)
    for d, sibs in by_def.items():
        local_sets = [set(instances[s].members.values()) for s in sibs]
        assert all(ls == local_sets[0] for ls in local_sets), (
            f"shared siblings of {d} diverge in member locals: {local_sets}"
        )
        bnd_sets = [set(instances[s].interface.keys()) for s in sibs]
        assert all(bs == bnd_sets[0] for bs in bnd_sets), (
            f"shared siblings of {d} diverge in boundary ids: {bnd_sets}"
        )

    # --- display names are globally unique (nd() resolves by name) -------------
    names = [mgr.nodes[u].name for u in nodes]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"duplicate display names break nd() resolution: {dupes}"


# === the checker actually catches corruption (meta-tests) =====================

def test_checker_passes_on_a_clean_grouped_graph():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        _build_grouped_graph(mgr)
        assert_subpatch_invariants(mgr)  # must not raise
    finally:
        mgr.terminate()


def test_checker_catches_orphan_membership():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        _osc, inst = _build_grouped_graph(mgr)
        mgr._membership["ghostuid"] = inst  # uid that is neither a member nor a node
        with pytest.raises(AssertionError):
            assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate()


def test_checker_catches_a_member_with_no_live_node():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        _osc, inst = _build_grouped_graph(mgr)
        # a member listed in the instance + membership but with no backing live node
        mgr._instances[inst].members["phantomuid"] = "phantom0"
        mgr._membership["phantomuid"] = inst
        with pytest.raises(AssertionError):
            assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate()


def test_checker_catches_dangling_definition_reference():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        _osc, inst = _build_grouped_graph(mgr)
        mgr._instances[inst].def_id = "ghostdef"  # references a missing definition
        with pytest.raises(AssertionError):
            assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate()


def test_checker_catches_duplicate_display_name():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)
        # collide two display names (nd() would misroute)
        sel0 = _member(mgr, inst, "select0")
        mgr.nodes[sel0].name = mgr.nodes[osc].name
        with pytest.raises(AssertionError):
            assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate()


# === semantic scenarios with invariants asserted at every step ================

def test_invariants_hold_through_full_lifecycle():
    """group -> share -> instantiate -> add member -> wire output boundary, with the
    structural rail checked after each mutation and the key semantics spot-checked."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)
        assert_subpatch_invariants(mgr)

        def_id = mgr.share_instance(inst)
        assert_subpatch_invariants(mgr)

        inst2 = mgr.instantiate_definition(def_id)
        assert_subpatch_invariants(mgr)
        # both instances are siblings of the same definition
        assert mgr._instances[inst].def_id == def_id == mgr._instances[inst2].def_id

        # a shared edit mirrors to the sibling (semantic, not representation)
        mgr.update_param(_member(mgr, inst, "select0"), "select", "include", "3:9")
        assert mgr.nodes[_member(mgr, inst2, "select0")].params["select"]["include"].value == "3:9"
        assert_subpatch_invariants(mgr)

        # add a member: mirrors into the def + sibling, invariants still hold
        mgr.add_member_node(inst, "Buffer", "signal")
        assert_subpatch_invariants(mgr)

        # wire an output boundary and confirm it resolves to the real inner member
        s1 = _member(mgr, inst, "select1")
        out_slot = list(mgr.nodes[s1].output_slots)[0]
        bnd = mgr.add_boundary(inst, "out", "ARRAY")
        mgr.wire_boundary(inst, bnd, "select1", out_slot)
        assert mgr.resolve_boundary(inst, bnd) == (s1, out_slot)
        assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate()


def test_expand_last_instance_gcs_definition():
    """Expanding the FINAL instance of a shared definition must GC the now-orphaned
    definition, exactly as remove_instance and make_unique do — otherwise dead
    definitions accumulate in the saved patch."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        _osc, inst = _build_grouped_graph(mgr)
        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)
        mgr.expand_instance(inst2)
        assert def_id in mgr._definitions  # still referenced by `inst`
        mgr.expand_instance(inst)
        assert def_id not in mgr._definitions, "orphaned definition not GC'd on expand"
    finally:
        mgr.terminate()


def test_invariants_hold_through_make_unique_and_expand():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)
        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)
        assert_subpatch_invariants(mgr)

        # detach one sibling: def survives (still referenced by `inst`)
        mgr.make_unique(inst2)
        assert mgr._instances[inst2].def_id is None
        assert def_id in mgr._definitions
        assert_subpatch_invariants(mgr)

        # detach the other: def is GC'd
        mgr.make_unique(inst)
        assert def_id not in mgr._definitions
        assert_subpatch_invariants(mgr)

        # expand dissolves a group back to top-level nodes; external link preserved
        members = set(mgr._instances[inst].members)
        restored = set(mgr.expand_instance(inst))
        assert restored == members
        assert inst not in mgr._instances
        links = [dict(l) for l in mgr.links]
        sel0 = next(iter(restored & set(mgr.nodes)))  # at least one member still wired
        assert any(osc == l["node_out"] for l in links)
        assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate()


def test_invariants_hold_across_save_load(tmp_path):
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)
        def_id = mgr.share_instance(inst)
        mgr.instantiate_definition(def_id)
        assert_subpatch_invariants(mgr)
        fp = str(tmp_path / "lifecycle.gfi")
        mgr.save(fp, overwrite=True)
    finally:
        mgr.terminate()

    mgr2 = _bare_manager(use_multiprocessing=False)
    try:
        mgr2.load(fp)
        assert_subpatch_invariants(mgr2)
        # the shared family round-tripped: two instances on one definition
        defs = {i.def_id for i in mgr2._instances.values()}
        assert len(mgr2._instances) == 2 and defs and None not in defs
        # expanding every instance leaves a clean flat graph
        for iid in list(mgr2._instances):
            mgr2.expand_instance(iid)
        assert mgr2._instances == {} and mgr2._membership == {}
        assert_subpatch_invariants(mgr2)
    finally:
        mgr2.terminate()
