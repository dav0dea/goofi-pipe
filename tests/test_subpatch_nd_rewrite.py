"""nd() expression rewrite on group/expand (backlog #1, spec §2.6).

When nodes are grouped, members are renamed `name -> inst::name`. A sibling whose
param expression contains `nd('name')` must have that literal rewritten to the
qualified `nd('inst::name')`, or the cross-reference silently dies (the flat node
directory no longer holds the bare name). Expand reverses it.
"""
from goofi.manager import SUBPATCH_SEP, _rewrite_nd_literal

from .test_manager import _bare_manager


# --- pure literal rewrite ---------------------------------------------------

def test_rewrite_single_quote():
    assert _rewrite_nd_literal("nd('osc0') * 2", {"osc0": "sub0::osc0"}) == "nd('sub0::osc0') * 2"


def test_rewrite_double_quote():
    assert _rewrite_nd_literal('nd("osc0")', {"osc0": "sub0::osc0"}) == 'nd("sub0::osc0")'


def test_rewrite_multiple_refs():
    out = _rewrite_nd_literal("nd('a') + nd('b')", {"a": "s::a", "b": "s::b"})
    assert out == "nd('s::a') + nd('s::b')"


def test_non_member_ref_untouched():
    assert _rewrite_nd_literal("nd('ext')", {"osc0": "s::osc0"}) == "nd('ext')"


def test_no_nd_unchanged():
    assert _rewrite_nd_literal("x * 2", {"osc0": "s::osc0"}) == "x * 2"


def test_substring_name_not_matched():
    # 'osc' is renamed, but 'oscillator' is a different node — must NOT be rewritten.
    assert _rewrite_nd_literal("nd('oscillator')", {"osc": "s::osc"}) == "nd('oscillator')"


def test_none_expression_passthrough():
    assert _rewrite_nd_literal(None, {"a": "b"}) is None


# --- integration: group then expand round-trips the reference ---------------

def test_group_and_expand_rewrite_nd_cross_refs():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Oscillator", "inputs")
        ra, rb = mgr.nodes[a], mgr.nodes[b]
        rb.wait_for_state(timeout=2.0)
        grp, pname = "oscillator", "frequency"
        # nd() resolves by DISPLAY name, so the expression references a's name.
        a_name = ra.name
        rb.set_expression(grp, pname, f"nd('{a_name}')", enabled=True)

        inst = mgr.group_nodes([a, b])  # members referenced by uid
        # display names become qualified; the nd literal follows
        qualified_a = f"{inst}{SUBPATCH_SEP}{a_name}"
        assert mgr.nodes[b].params[grp][pname].expression == f"nd('{qualified_a}')"

        mgr.expand_instance(inst)
        assert mgr.nodes[b].params[grp][pname].expression == f"nd('{a_name}')"
    finally:
        mgr.terminate(notify_gui=False)


def test_external_nd_ref_follows_member_through_group_rename_expand():
    """An EXTERNAL node that references a member by name must have its nd() literal
    rewritten too (group: bare->qualified; rename: re-key; expand: ->bare), not just
    fellow members — else the cross-reference silently dies."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")  # oscillator0 (becomes a member)
        ext = mgr.add_node("Oscillator", "inputs")  # oscillator1 (external referrer)
        mgr.set_expression(ext, "oscillator", "frequency", "nd('oscillator0')", enabled=True)

        inst = mgr.group_nodes([a])
        local = mgr._instances[inst]["members"][a]
        f = mgr.nodes[ext].params["oscillator"]["frequency"]
        assert f.expression == f"nd('{inst}{SUBPATCH_SEP}{local}')"

        mgr.rename_node(a, f"{inst}{SUBPATCH_SEP}renamed")
        assert mgr.nodes[ext].params["oscillator"]["frequency"].expression == f"nd('{inst}{SUBPATCH_SEP}renamed')"

        mgr.expand_instance(inst)
        assert mgr.nodes[ext].params["oscillator"]["frequency"].expression == "nd('renamed')"
    finally:
        mgr.terminate(notify_gui=False)


def test_member_uid_raises_on_duplicate_local_rather_than_silent_misroute():
    """A member's local name is a per-instance key; if two members ever share one
    (corruption), _member_uid must fail loudly instead of returning whichever it
    scans first and silently splicing boundaries/data onto the wrong member."""
    import pytest

    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Buffer", "signal")
        inst = mgr.group_nodes([a, b])
        for uid in list(mgr._instances[inst]["members"]):
            mgr._instances[inst]["members"][uid] = "dup"  # force a collision
        with pytest.raises(RuntimeError):
            mgr._member_uid(inst, "dup")
    finally:
        mgr.terminate(notify_gui=False)
