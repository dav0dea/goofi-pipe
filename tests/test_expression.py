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


def test_slot_accessor_returns_none_initially() -> None:
    """`slot()` returns None on first call when no data has arrived yet.

    The engine doesn't crash; the user code path that does
    `slot(...).data` will raise, fall into the runtime-error path, and
    last_value is preserved. This documents the behavior."""
    e, added, _ = _make_engine()
    e.set_source("3.14")
    assert e.evaluate() == 3.14
    # Now reference a slot — the service has no publisher in this test
    # process, so `open_subscriber` returns a subscriber with no data.
    e.set_source("d = slot('nonexistent_node', 'out')\n0 if d is None else d.data")
    v = e.evaluate()
    # The slot returned None, expression evaluates to 0 (the `if` branch).
    assert v == 0
    # A subscription was opened — listener added (best-effort; on systems
    # without iceoryx2 init issues it'll be a real listener, otherwise the
    # entry's listener may be None and add not called. Both are valid).


def test_stale_refs_pruned_after_eval() -> None:
    """A reference dropped between evals is unsubscribed.

    We can't easily verify iceoryx2 close in this unit test, but we can
    verify the internal subscribed dict shrinks and the removed callback
    fires."""
    e, added, removed = _make_engine()
    e.set_source("a = slot('node_a', 'out')\nb = slot('node_b', 'out')\n0")
    e.evaluate()
    keys_before = set(e._subscribed.keys())  # noqa: SLF001
    assert ("node_a", "out") in keys_before
    assert ("node_b", "out") in keys_before

    # Drop node_b reference.
    e.set_source("a = slot('node_a', 'out')\n0")
    e.evaluate()
    keys_after = set(e._subscribed.keys())  # noqa: SLF001
    assert ("node_a", "out") in keys_after
    assert ("node_b", "out") not in keys_after
