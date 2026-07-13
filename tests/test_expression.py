"""Tests for the ExpressionEngine — the per-param Python evaluator that
backs goofi-pipe's TouchDesigner-style param-reference system.

These tests exercise the engine in isolation (no real iceoryx2 traffic):
``slot()`` falls back to the cached value when the service isn't
publishing yet, which is exactly what we want for unit tests."""
from __future__ import annotations

from typing import List

import numpy as np

from goofi.expression import ExpressionEngine


def _make_engine():
    added: List = []
    removed: List = []
    e = ExpressionEngine("test", added.append, removed.append)
    return e, added, removed


def test_last_expression_is_value() -> None:
    e, _, _ = _make_engine()
    e.set_source("x = 5\nx * 2")
    assert e.evaluate() == 10


def test_persistent_namespace_across_evals() -> None:
    e, _, _ = _make_engine()
    e.set_source(
        "import numpy as np\n"
        "def double(x):\n"
        "    return np.asarray(x) * 2\n"
        "double([1, 2, 3]).sum()"
    )
    assert e.evaluate() == 12
    # Run again — imports and def survive, no re-import cost beyond Python's
    # module cache.
    assert e.evaluate() == 12


def test_numpy_preloaded_as_np() -> None:
    e, _, _ = _make_engine()
    e.set_source("np.mean([1.0, 2.0, 3.0])")
    assert e.evaluate() == 2.0


def test_syntax_error_preserves_last_good() -> None:
    e, _, _ = _make_engine()
    e.set_source("42")
    assert e.evaluate() == 42
    e.set_source("def def def")
    assert e.last_error is not None
    # The compiled code is unchanged; evaluation still returns last-good.
    assert e.evaluate() == 42


def test_runtime_error_captures_and_returns_last_good() -> None:
    e, _, _ = _make_engine()
    e.set_source("100")
    assert e.evaluate() == 100
    e.set_source("1 / 0")
    v = e.evaluate()
    assert "ZeroDivisionError" in (e.last_error or "")
    assert v == 100


def test_no_trailing_expression_yields_none() -> None:
    e, _, _ = _make_engine()
    # No trailing Expr → __result__ is never assigned in this pass.
    e.set_source("a = 1\nb = 2")
    assert e.evaluate() is None


def test_clear_source_tears_down() -> None:
    e, _, _ = _make_engine()
    e.set_source("1 + 1")
    e.evaluate()
    e.set_source(None)
    assert e.source is None
    assert e._code is None  # noqa: SLF001 — direct check of cleared state


def test_nd_accessor_returns_none_initially() -> None:
    """`nd('node').slot` returns None on first call when no data has
    arrived yet. The engine doesn't crash; user code that dots through
    a None will raise, fall into the runtime-error path, and last_value
    is preserved."""
    e, added, _ = _make_engine()
    e.set_source("3.14")
    assert e.evaluate() == 3.14
    # Reference a slot — service has no publisher in this test process.
    e.set_source("d = nd('nonexistent_node').out\n0 if d is None else d.data")
    v = e.evaluate()
    assert v == 0


def test_stale_refs_pruned_after_eval() -> None:
    """A reference dropped between evals is unsubscribed.

    We can't easily verify iceoryx2 close in this unit test, but we can
    verify the internal subscribed dict shrinks."""
    e, added, removed = _make_engine()
    e.set_source("a = nd('node_a').out\nb = nd('node_b').out\n0")
    e.evaluate()
    keys_before = set(e._subscribed.keys())  # noqa: SLF001
    assert ("node_a", "out") in keys_before
    assert ("node_b", "out") in keys_before

    # Drop node_b reference.
    e.set_source("a = nd('node_a').out\n0")
    e.evaluate()
    keys_after = set(e._subscribed.keys())  # noqa: SLF001
    assert ("node_a", "out") in keys_after
    assert ("node_b", "out") not in keys_after


def test_nd_reference_resolves_display_name_to_node_id() -> None:
    """`nd('name')` must resolve the user-facing display name to the
    producer's stable transport id before opening a subscriber. The engine
    keys its subscription on the *resolved* id, so the service name matches
    the producer (whose iceoryx2 services use the unique id, not the name)."""
    directory = {"osc": "osc-1a2b3c4d"}
    e = ExpressionEngine(
        "test",
        lambda l: None,
        lambda l: None,
        resolve_node_id=lambda name: directory.get(name, name),
    )
    e.set_source("d = nd('osc').out\n0 if d is None else d.data")
    e.evaluate()
    keys = set(e._subscribed.keys())  # noqa: SLF001
    assert ("osc-1a2b3c4d", "out") in keys
    assert ("osc", "out") not in keys


def test_nd_reference_default_resolver_is_identity() -> None:
    """With no resolver supplied (the unit-test / standalone path), the
    display name is used verbatim — preserving the engine's behaviour for
    callers that don't wire a directory."""
    e, _, _ = _make_engine()
    e.set_source("d = nd('plain_name').out\n0")
    e.evaluate()
    assert ("plain_name", "out") in set(e._subscribed.keys())  # noqa: SLF001


def test_engine_reports_whether_it_references_other_nodes() -> None:
    """The owning node re-applies only the expressions that reference other
    nodes when the name->id directory changes; a self-contained expression
    (no nd()) reports no references so it is left untouched."""
    e_plain, _, _ = _make_engine()
    e_plain.set_source("time.time()")
    e_plain.evaluate()
    assert e_plain.references_other_nodes() is False

    e_ref, _, _ = _make_engine()
    e_ref.set_source("nd('producer').out\n0")
    e_ref.evaluate()
    assert e_ref.references_other_nodes() is True


def test_extract_nd_refs_pulls_node_slot_pairs() -> None:
    from goofi.expression import extract_nd_refs

    # single ref with a slot
    assert extract_nd_refs('nd("delta").normalized.data * 2') == {("delta", "normalized")}
    # multiple refs, chained attrs, mixed with other calls
    src = 'nd("a").out.mean() + nd("b").val.data - foo(nd("a").out)'
    assert extract_nd_refs(src) == {("a", "out"), ("b", "val")}
    # non-literal arg / look-alike / no-attr are ignored (no false positives)
    assert extract_nd_refs("nd(x).out + gnd('y').z + nd('c')") == set()
    # unparseable / empty
    assert extract_nd_refs("nd('a').") == set()
    assert extract_nd_refs("") == set()
    assert extract_nd_refs(None) == set()
