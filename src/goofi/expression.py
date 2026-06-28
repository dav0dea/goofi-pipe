"""Expression-bound parameters.

A param can carry a Python snippet that references other nodes' output
slots via the ``nd(node_id).<slot_name>`` accessor:

    nd("oscillator0").out.data.mean()

The snippet runs in the owning node's process and the value of the
trailing expression (Jupyter-style) becomes the param's cached value.

The engine maintains a persistent eval namespace (so ``import`` and
helper ``def``s survive across evaluations) and a per-engine set of
slot subscriptions opened lazily on first attribute access. After
every successful eval, references not touched on that pass are
unsubscribed so a branch-gated reference doesn't leak a subscription
and so editing the source to drop a reference is reflected in wiring.

The engine never attaches its own listeners to a WaitSet — it pushes
listener add/remove callbacks back to the owning node, which manages
the WaitSet membership. This keeps the engine reusable from tests and
keeps WaitSet ownership in one place.
"""

from __future__ import annotations

import ast
import datetime
import math
import random
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Set, Tuple

import numpy as np

from goofi.codec import decode_data
from goofi.transport import (
    Listener,
    Subscriber,
    data_service_name,
    open_subscriber,
)


SlotKey = Tuple[str, str]


# The single source of truth for the expression accessor token. Bound into the eval
# namespace (`_make_namespace`) AND keyed on by `rewrite_nd_refs`, so renaming the
# function is a one-line change here rather than a codebase-wide sweep.
ND_FUNC_NAME = "nd"


def _line_start_offsets(source: str) -> list[int]:
    """Absolute string offset where each 1-based source line begins."""
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def rewrite_nd_refs(source: Optional[str], name_map: Dict[str, str]) -> Optional[str]:
    """Rewrite string-literal ``nd('name')`` references whose name is a key of
    ``name_map`` to the mapped name — and ONLY those.

    Parses ``source`` with ``ast`` and edits just the matched argument literal in
    place, preserving every other byte of the source (whitespace, comments, the rest
    of the expression). A look-alike call (``gnd``), a method call (``obj.nd(...)``),
    a name that merely appears inside another string or a comment, and any non-literal
    argument (``nd(x)``, ``nd('a' + 'b')``, ``nd(f'{x}')``) are all left untouched —
    the AST is the filter, so there are no substring false positives. Un-parseable
    source (a node mid-edit) is returned unchanged. The accessor token is
    ``ND_FUNC_NAME``."""
    if not source or not name_map or ND_FUNC_NAME not in source:
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    line_starts = _line_start_offsets(source)
    edits: list[tuple[int, int, str]] = []  # (start, end, replacement)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == ND_FUNC_NAME):
            continue
        if not node.args:
            continue
        arg = node.args[0]
        if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
            continue
        new = name_map.get(arg.value)
        if new is None:
            continue
        start = line_starts[arg.lineno - 1] + arg.col_offset
        end = line_starts[arg.end_lineno - 1] + arg.end_col_offset
        quote = source[end - 1]  # the literal's closing quote — preserve its style
        edits.append((start, end, f"{quote}{new}{quote}"))

    if not edits:
        return source
    # Apply right-to-left so earlier offsets stay valid as the string is spliced.
    out = source
    for start, end, repl in sorted(edits, key=lambda e: e[0], reverse=True):
        out = out[:start] + repl + out[end:]
    return out


@dataclass
class _SubEntry:
    sub: Subscriber
    listener: Listener
    last: Any  # last decoded Data, or None until first arrival


class _NodeProxy:
    """Returned by ``nd(node_id)`` inside expression eval. Attribute
    access maps each ``<slot_name>`` to the latest cached ``Data`` from
    that slot — opens a subscriber lazily and records the reference on
    the owning engine. Underscored attrs raise AttributeError so the
    proxy stays inspectable in debuggers / reprs without accidentally
    opening subscriptions to internal names.
    """

    __slots__ = ("_engine", "_node_id")

    def __init__(self, engine: "ExpressionEngine", node_id: str) -> None:
        object.__setattr__(self, "_engine", engine)
        object.__setattr__(self, "_node_id", node_id)

    def __getattr__(self, slot_name: str):
        if slot_name.startswith("_"):
            raise AttributeError(slot_name)
        return self._engine._fetch_slot(self._node_id, slot_name)

    def __repr__(self) -> str:
        return f"nd({self._node_id!r})"


class ExpressionEngine:
    """One instance per expression-bound param.

    Parameters
    ----------
    location:
        Used as the compile-time filename so traceback frames carry a
        useful tag (e.g. ``"node:foo.group:freq"``).
    on_listener_added / on_listener_removed:
        Called when a new subscription opens or an existing one is
        pruned. The owning node uses these to keep its WaitSet in sync.
    resolve_node_id:
        Maps the user-facing display name passed to ``nd('name')`` onto the
        producing node's stable transport id (the data services are keyed on
        the id, not the reusable name). Defaults to identity for callers that
        reference nodes by id directly (tests / standalone use).
    """

    def __init__(
        self,
        location: str,
        on_listener_added: Callable[[Listener], None],
        on_listener_removed: Callable[[Listener], None],
        resolve_node_id: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.location = location
        self._on_listener_added = on_listener_added
        self._on_listener_removed = on_listener_removed
        self._resolve_node_id = resolve_node_id or (lambda name: name)
        self._source: Optional[str] = None
        self._code = None
        self._namespace: Dict[str, Any] = self._make_namespace()
        self._subscribed: Dict[SlotKey, _SubEntry] = {}
        self._refs_this_eval: Set[SlotKey] = set()
        self.last_value: Any = None
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def source(self) -> Optional[str]:
        return self._source

    def set_source(self, source: Optional[str]) -> None:
        """Set or clear the expression source.

        Empty / None clears (and tears down all subscriptions). On a
        syntax/compile error the engine retains its previous compiled
        code so the param keeps producing a sensible value.
        """
        if source is None or source.strip() == "":
            self.close()
            self._source = None
            self._code = None
            self.last_error = None
            return

        try:
            tree = ast.parse(source, mode="exec")
        except SyntaxError:
            self.last_error = traceback.format_exc()
            return

        # Last-expression-is-value: if the trailing statement is a bare
        # expression, rewrite it as `__result__ = <expr>` so we can read
        # it back out of the namespace after exec.
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = tree.body[-1]
            tree.body[-1] = ast.Assign(
                targets=[ast.Name(id="__result__", ctx=ast.Store())],
                value=last.value,
            )
            ast.fix_missing_locations(tree)

        try:
            self._code = compile(tree, f"<expr:{self.location}>", "exec")
        except Exception:
            self.last_error = traceback.format_exc()
            return

        self._source = source
        self.last_error = None

    def evaluate(self) -> Any:
        """Run the compiled code once. Returns the trailing-expression value.

        Errors are captured into ``last_error`` (so the owning node can
        surface them) and ``last_value`` is left at its previous setting.
        """
        if self._code is None:
            return self.last_value

        self._refs_this_eval.clear()
        # Clear any prior `__result__` so a code path that omits a
        # trailing expression doesn't accidentally carry a stale value.
        self._namespace["__result__"] = None
        try:
            exec(self._code, self._namespace)
        except Exception:
            self.last_error = traceback.format_exc()
            return self.last_value

        result = self._namespace.get("__result__")
        self.last_value = result
        self.last_error = None
        self._prune_stale()
        return result

    def owns_listener(self, listener: Listener) -> bool:
        return any(entry.listener is listener for entry in self._subscribed.values())

    def references_other_nodes(self) -> bool:
        """True if this engine holds any cross-node slot subscription — i.e.
        its resolution depends on the name->id directory, so the owning node
        should re-apply it when the directory changes."""
        return bool(self._subscribed)

    def close(self) -> None:
        """Tear down every subscription. Safe to call repeatedly."""
        for entry in list(self._subscribed.values()):
            self._tear_down(entry)
        self._subscribed.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_namespace(self) -> Dict[str, Any]:
        return {
            "__builtins__": __builtins__,
            "np": np,
            "numpy": np,
            "math": math,
            "time": time,
            "random": random,
            "datetime": datetime,
            "re": re,
            ND_FUNC_NAME: self._nd,
        }

    def _nd(self, node_id: str) -> _NodeProxy:
        """The ``nd()`` function the user code calls. Returns a proxy
        whose attribute access maps to the cached ``Data`` of that slot.
        """
        return _NodeProxy(self, node_id)

    def _fetch_slot(self, node_ref: str, slot_name: str):
        """Records the (node, slot) tuple as referenced this pass, opens
        a new subscriber on first sight, drains any fresh frame into the
        cache, and returns the latest decoded ``Data`` (or None until
        the first frame arrives).

        ``node_ref`` is the display name from ``nd('name')``; it is resolved
        to the producer's stable transport id so the subscription keys (and
        the service we open) track the current owner of that name. Resolving
        every eval makes the reference self-healing across a node
        delete+recreate: a changed id yields a fresh key + subscription while
        the stale one is pruned.
        """
        node_id = self._resolve_node_id(node_ref)
        key: SlotKey = (node_id, slot_name)
        self._refs_this_eval.add(key)

        if key not in self._subscribed:
            service = data_service_name(node_id, slot_name)
            try:
                sub, listener = open_subscriber(
                    service, in_process=False, latest_wins=True
                )
            except Exception:
                # Service unavailable right now — record the ref so we
                # don't try every eval, but return None to the user code.
                self._subscribed[key] = _SubEntry(sub=None, listener=None, last=None)  # type: ignore[arg-type]
                return None
            self._subscribed[key] = _SubEntry(sub=sub, listener=listener, last=None)
            try:
                self._on_listener_added(listener)
            except Exception:
                pass

        entry = self._subscribed[key]
        if entry.sub is not None:
            try:
                buf = entry.sub.take_latest()
            except Exception:
                buf = None
            if buf is not None:
                try:
                    entry.last = decode_data(buf)
                except Exception:
                    self.last_error = traceback.format_exc()
        return entry.last

    def _prune_stale(self) -> None:
        stale = set(self._subscribed.keys()) - self._refs_this_eval
        for key in stale:
            entry = self._subscribed.pop(key)
            self._tear_down(entry)

    def _tear_down(self, entry: _SubEntry) -> None:
        if entry.listener is not None:
            try:
                self._on_listener_removed(entry.listener)
            except Exception:
                pass
            try:
                entry.listener.close()
            except Exception:
                pass
        if entry.sub is not None:
            try:
                entry.sub.close()
            except Exception:
                pass
