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
    mgr = _bare_manager()
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Oscillator", "inputs")
        rb = mgr.nodes[b]
        rb.wait_for_state(timeout=2.0)
        grp, pname = "oscillator", "frequency"
        rb.set_expression(grp, pname, f"nd('{a}')", enabled=True)

        inst = mgr.group_nodes([a, b])
        new_b = f"{inst}{SUBPATCH_SEP}{b}"
        qualified_a = f"{inst}{SUBPATCH_SEP}{a}"
        assert mgr.nodes[new_b].params[grp][pname].expression == f"nd('{qualified_a}')"

        mgr.expand_instance(inst)
        assert mgr.nodes[b].params[grp][pname].expression == f"nd('{a}')"
    finally:
        mgr.terminate(notify_gui=False)
