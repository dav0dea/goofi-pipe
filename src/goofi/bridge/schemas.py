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


def describe_node_instance(name: str, ref: NodeRef) -> Dict[str, Any]:
    """Describe a *live* node (instance) on the current graph."""
    gk = ref.gui_kwargs or {}
    return {
        "name": name,
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
        "error": ref.last_error,
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
