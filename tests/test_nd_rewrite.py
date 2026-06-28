"""AST-based nd() reference rewriting (Phase 2a of the recursive-sub-patch redesign).

`rewrite_nd_refs(source, name_map)` rewrites string-literal `nd('name')` references
whose name is a key of `name_map`, and ONLY those — it must never touch a stray
substring, a name inside another string, a comment, a look-alike call (`gnd`), or a
non-literal argument. It parses the source with `ast`, finds genuine
`Call(func=Name(ND_FUNC_NAME), args=[Constant(str)])` nodes, and surgically replaces
just the matched argument literal, preserving all surrounding source byte-for-byte.

The token `nd` itself lives in one place — `ND_FUNC_NAME` — so renaming the function
later is a one-line change, not a codebase sweep.
"""
from goofi.expression import ND_FUNC_NAME, rewrite_nd_refs


# --- basic rewrites ---------------------------------------------------------

def test_single_quote():
    assert rewrite_nd_refs("nd('osc0') * 2", {"osc0": "osc5"}) == "nd('osc5') * 2"


def test_double_quote_preserved():
    assert rewrite_nd_refs('nd("osc0")', {"osc0": "osc5"}) == 'nd("osc5")'


def test_multiple_refs():
    assert rewrite_nd_refs("nd('a') + nd('b')", {"a": "x", "b": "y"}) == "nd('x') + nd('y')"


def test_same_ref_twice():
    assert rewrite_nd_refs("nd('a') + nd('a')", {"a": "z"}) == "nd('z') + nd('z')"


def test_whitespace_inside_call_tolerated():
    assert rewrite_nd_refs("nd(  'osc0'  )", {"osc0": "osc5"}) == "nd(  'osc5'  )"


def test_attribute_access_after_nd_is_preserved():
    src = "nd('osc0').out.data.mean() + 1"
    assert rewrite_nd_refs(src, {"osc0": "renamed"}) == "nd('renamed').out.data.mean() + 1"


# --- selectivity: must NOT rewrite -----------------------------------------

def test_non_member_ref_untouched():
    assert rewrite_nd_refs("nd('ext')", {"osc0": "osc5"}) == "nd('ext')"


def test_substring_name_not_matched():
    # 'osc' renamed, but 'oscillator' is a different node — must NOT be rewritten.
    assert rewrite_nd_refs("nd('oscillator')", {"osc": "x"}) == "nd('oscillator')"


def test_lookalike_function_not_matched():
    # gnd(...) is a different identifier; only a bare `nd` call counts.
    assert rewrite_nd_refs("gnd('osc0')", {"osc0": "osc5"}) == "gnd('osc0')"


def test_name_inside_a_string_literal_not_matched():
    src = "x = \"nd('osc0')\"\nx"
    assert rewrite_nd_refs(src, {"osc0": "osc5"}) == src


def test_name_inside_a_comment_not_matched():
    src = "y = 1  # nd('osc0') here\ny"
    assert rewrite_nd_refs(src, {"osc0": "osc5"}) == src


def test_non_literal_argument_not_matched():
    # A dynamic arg can't be statically rewritten — leave it alone.
    assert rewrite_nd_refs("nd(name)", {"name": "x"}) == "nd(name)"
    assert rewrite_nd_refs("nd('a' + 'b')", {"a": "x", "ab": "y"}) == "nd('a' + 'b')"
    assert rewrite_nd_refs("nd(f'{p}')", {"p": "x"}) == "nd(f'{p}')"


def test_method_named_nd_not_matched():
    # obj.nd('osc0') is an attribute call, not the global nd() — leave it.
    assert rewrite_nd_refs("obj.nd('osc0')", {"osc0": "osc5"}) == "obj.nd('osc0')"


# --- multi-line / structure-preserving --------------------------------------

def test_multiline_source_preserved_outside_the_ref():
    src = "import numpy as np\nv = nd('osc0').out.data\nnp.mean(v)"
    out = rewrite_nd_refs(src, {"osc0": "renamed"})
    assert out == "import numpy as np\nv = nd('renamed').out.data\nnp.mean(v)"


def test_two_refs_on_different_lines():
    src = "a = nd('p')\nb = nd('q')\na + b"
    assert rewrite_nd_refs(src, {"p": "P", "q": "Q"}) == "a = nd('P')\nb = nd('Q')\na + b"


# --- robustness edges -------------------------------------------------------

def test_none_passthrough():
    assert rewrite_nd_refs(None, {"a": "b"}) is None


def test_empty_map_is_noop():
    assert rewrite_nd_refs("nd('a')", {}) == "nd('a')"


def test_no_nd_unchanged():
    assert rewrite_nd_refs("x * 2 + y", {"a": "b"}) == "x * 2 + y"


def test_invalid_python_returns_unchanged():
    # Can't safely AST-rewrite un-parseable source; return it untouched.
    src = "nd('osc0') +"  # syntax error
    assert rewrite_nd_refs(src, {"osc0": "osc5"}) == src


def test_nd_func_name_is_the_single_token_source():
    # The constant exists and is what the rewriter keys on; renaming it here would
    # retarget both the rewriter and the eval-namespace binding in one place.
    assert ND_FUNC_NAME == "nd"
