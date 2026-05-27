"""Expression-bound parameters.

A param can carry a Python snippet that references other nodes' output
slots via a ``slot(node_id, slot_name)`` accessor; the snippet runs in
the owning node's process and the value of the trailing expression
(Jupyter-style) becomes the param's cached value.

The engine maintains a persistent eval namespace (so ``import`` and
helper ``def``s survive across evaluations) and a per-engine set of
slot subscriptions opened lazily on first ``slot()`` call. After every
successful eval, references not touched on that pass are unsubscribed
so a branch-gated reference doesn't leak a subscription, and so editing
the source to drop a reference is reflected in the wiring.

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


@dataclass
class _SubEntry:
    sub: Subscriber
    listener: Listener
    last: Any  # last decoded Data, or None until first arrival


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
    """

    def __init__(
        self,
        location: str,
        on_listener_added: Callable[[Listener], None],
        on_listener_removed: Callable[[Listener], None],
    ) -> None:
        self.location = location
        self._on_listener_added = on_listener_added
        self._on_listener_removed = on_listener_removed
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
            "slot": self._slot_accessor,
        }

    def _slot_accessor(self, node_id: str, slot_name: str):
        """The ``slot()`` function the user code calls.

        Records the (node, slot) tuple as referenced this pass, opens a
        new subscriber on first sight, drains any fresh frame into the
        cache, and returns the latest decoded ``Data`` (or None until
        the first frame arrives).
        """
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
