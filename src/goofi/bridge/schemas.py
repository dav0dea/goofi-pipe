"""Serialization helpers for the bridge control plane.

The browser sees a JSON-friendly projection of the manager's state.
`describe_*` helpers convert internal Python types (NodeRef, NodeParams,
Node classes) into plain dicts that round-trip through JSON.
"""
from __future__ import annotations

from typing import Any, Dict, List

from goofi.node_helpers import NodeRef, list_nodes
from goofi.params import (
    BoolParam,
    FloatParam,
    IntParam,
    NodeParams,
    Param,
    StringParam,
)


def describe_param(p: Param) -> Dict[str, Any]:
    """Serialize a single Param to a JSON-safe dict.

    Includes the param type, current value, doc, plus type-specific
    constraints (min/max for numeric, options for string dropdowns,
    trigger flag for bools).
    """
    out: Dict[str, Any] = {
        "value": p._value,
        "doc": p.doc,
        "save_param": p.save_param,
        "expression": getattr(p, "expression", None),
        "expression_enabled": bool(getattr(p, "expression_enabled", False)),
        "expression_triggers_process": bool(
            getattr(p, "expression_triggers_process", False)
        ),
        "expression_autoeval": bool(getattr(p, "expression_autoeval", False)),
    }
    if isinstance(p, FloatParam):
        out["type"] = "float"
        out["vmin"] = p.vmin
        out["vmax"] = p.vmax
    elif isinstance(p, IntParam):
        out["type"] = "int"
        out["vmin"] = p.vmin
        out["vmax"] = p.vmax
    elif isinstance(p, BoolParam):
        out["type"] = "bool"
        out["trigger"] = p.trigger
    elif isinstance(p, StringParam):
        out["type"] = "string"
        out["options"] = p.options
    else:
        out["type"] = "unknown"
    return out


def describe_params(params: NodeParams) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for group_name in params.keys():
        group = params[group_name]
        out[group_name] = {name: describe_param(group[name]) for name in group._fields}
    return out


def describe_node_class(cls: type) -> Dict[str, Any]:
    """Describe a Node *class* for the add-node palette."""
    input_slots, output_slots, default_params = cls._configure()
    return {
        "type": cls.__name__,
        "category": cls.category(),
        "doc": cls.docstring() or "",
        "input_slots": {name: slot.dtype.name for name, slot in input_slots.items()},
        "output_slots": {name: slot.dtype.name for name, slot in output_slots.items()},
        "params": describe_params(default_params),
    }


def describe_instance(manager: Any, inst_id: str, inst: Any) -> Dict[str, Any]:
    """Describe a sub-patch INSTANCE as a computed, recursive record the frontend
    mirrors (rather than re-deriving). The single server source of truth for every
    per-instance field; each computed field replaces a frontend re-derivation:

    - `parent`: the tree edge (None at root) -> frontend parentScope.
    - `members`: inverted+enriched {local: {uid, is_instance}} (locals are unique within
      an instance) -> lets the editor split direct children into nodes vs nested
      collapsed instances, and kills the _memberUid reverse-scan.
    - `interface[bid].dtype`: RESOLVED chain-to-leaf via resolve_boundary + _slot_dtype
      (authoritative even for legacy/unwired-then-rewired entries); falls back to the
      stored dtype when unwired/gone (guarded so one bad boundary can't drop the record).
    - `slots`: {input,output} computed from WIRED boundaries -> the synth node's external
      ports become a pure passthrough.
    - `siblings`: _shared_siblings (def-scoped strict-mirror family); [] for unique.
    - `error`: first errored DESCENDANT across the whole subtree (recursion-correct — a
      deeply-nested member's error surfaces on the collapsed ancestor).
    - `member_count`, `viewers`: direct reads (viewers round-trips the persisted state).
    """
    interface: Dict[str, Any] = {}
    slots: Dict[str, Dict[str, str]] = {"input": {}, "output": {}}
    for bid, b in inst.interface.items():
        dtype = b.dtype
        if b.inner_node is not None:
            try:
                leaf_uid, leaf_slot = manager.resolve_boundary(inst_id, bid)
                dtype = manager._slot_dtype(leaf_uid, leaf_slot, b.dir)
            except (KeyError, ValueError):
                pass  # unwired/gone chain — keep the stored dtype, never throw
            slots["input" if b.dir == "in" else "output"][bid] = dtype or "ARRAY"
        interface[bid] = {
            "dir": b.dir,
            "dtype": dtype,
            "inner_node": b.inner_node,
            "inner_slot": b.inner_slot,
            "pos": list(b.pos),
        }
    error = next(
        (manager.nodes[u].last_error for u in manager.nodes
         if manager.nodes[u].last_error and manager._within_subtree(u, inst_id)),
        None,
    )
    return {
        "uid": inst.uid,
        "name": inst.name,
        "kind": inst.kind,
        "def_id": inst.def_id,
        "parent": inst.parent,
        "pos": list(inst.pos),
        "viewers": dict(inst.viewers or {}),
        "members": {
            local: {"uid": uid, "is_instance": uid in manager._instances}
            for uid, local in inst.members.items()
        },
        "interface": interface,
        "slots": slots,
        "siblings": manager._shared_siblings(inst_id),
        "error": error,
        "member_count": len(inst.members),
    }


def describe_node_instance(uid: str, ref: NodeRef) -> Dict[str, Any]:
    """Describe a *live* node (instance) on the current graph.

    `uid` is the universal identity (the key everything references); `name` is the
    mutable display label."""
    gk = ref.gui_kwargs or {}
    return {
        "uid": uid,
        "name": ref.name,
        "type": ref.node_class.__name__,
        "category": ref.category,
        "doc": ref.node_class.docstring() or "",
        "input_slots": {n: dt.name for n, dt in ref.input_slots.items()},
        "output_slots": {n: dt.name for n, dt in ref.output_slots.items()},
        "params": describe_params(ref.params),
        "pos": list(gk.get("pos", (0, 0))),
        # Per-output-slot view state persisted in the .gfi patch:
        #   { "<slot>": {"collapsed": <bool>} }
        # The frontend uses this to restore expand state on patch load.
        "viewers": dict(gk.get("viewers") or {}),
        # Sub-patch membership marker (None for a top-level node): the editor
        # hides members of a collapsed instance and shows the group node instead.
        "membership": ref.membership,
        "error": ref.last_error,
        # Rolling execution telemetry the node pushes on the status plane
        # (~1 Hz). None until the node's first NODE_STATS push.
        "stats": ref.node_stats,
        # How many times the supervisor has auto-restarted this node after a
        # process crash; lets a freshly-connected client render the crash badge.
        "restarts": ref.restart_count,
        # Peer-to-peer SSE log endpoint the node advertises via STATE_UPDATE.
        # None until the node's first state push (or when capture is off).
        "log_endpoint": (ref.serialized_state or {}).get("log_endpoint"),
    }


def list_node_types() -> List[Dict[str, Any]]:
    """Return all registered node classes as a JSON-safe list."""
    out: List[Dict[str, Any]] = []
    for cls in list_nodes():
        try:
            out.append(describe_node_class(cls))
        except Exception as e:
            print(f"bridge: failed to describe {cls.__name__}: {e}")
    out.sort(key=lambda x: (x["category"], x["type"]))
    return out
