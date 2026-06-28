"""Versioned `.gfi` patch-format boundary.

One place that knows the on-disk shape. `normalize_loaded` validates a read
document; `build_v2` emits it; `read_graph` splits a v2 document into the pieces
`Manager._expand_doc` splices into the live graph.

v2 shape — uid-keyed (the node uid is the universal identity; the display name
rides inside each node record, and link endpoints are uids):

    version: 2
    definitions: {def_id: {members, links, interface}}   # shared sub-patch graphs
    root:
      nodes:     {uid: {name, _type, category, params, gui_kwargs, ...}}
      links:     [{node_out: uid, node_in: uid, slot_out, slot_in}, ...]
      instances: {inst_id: {kind, def|inline, gui_kwargs, members}}
    layout: <opaque>   # optional
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

CURRENT_VERSION = 2


def normalize_loaded(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return `raw` as a current-version (v2) document. v2 is uid-keyed (nodes
    and link endpoints are node uids); the legacy flat/name-keyed v1 format is no
    longer supported."""
    version = raw.get("version")
    if version == CURRENT_VERSION:
        return raw
    raise ValueError(f"unsupported patch version {version!r}")


def build_v2(
    nodes: Dict[str, Any],
    links: List[Dict[str, str]],
    layout: Any,
    definitions: Dict[str, Any],
    instances: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble a v2 document. `version` is first so yaml.dump is stable."""
    doc: Dict[str, Any] = {
        "version": CURRENT_VERSION,
        "definitions": definitions,
        "root": {"nodes": nodes, "links": links, "instances": instances},
    }
    if layout is not None:
        doc["layout"] = layout
    return doc


def read_graph(doc: Dict[str, Any]) -> Tuple[Dict, List, Dict, Dict, Any]:
    """Read a v2 document into (root_nodes, root_links, instances, definitions,
    layout); the caller (Manager._expand_doc) splices the instances into the flat
    live graph.
    """
    root = doc.get("root") or {}
    return (
        root.get("nodes", {}),
        root.get("links", []),
        root.get("instances", {}),
        doc.get("definitions") or {},
        doc.get("layout"),
    )
