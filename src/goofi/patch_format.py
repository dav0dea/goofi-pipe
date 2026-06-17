"""Versioned `.gfi` patch-format boundary.

One place that knows the on-disk shape. `normalize_loaded` upgrades any read
document to the current in-memory v2 shape; `build_v2` emits it; `flat_view`
projects a v2 document's root graph for the (still flat) live manager.

v2 shape (recursive-ready, no embedded node source):

    version: 2
    definitions: {def_id: {members, links, interface}}   # shared sub-patch graphs
    root:
      nodes:     {name: {uid, _type, category, params, gui_kwargs, ...}}
      links:     [{node_out, node_in, slot_out, slot_in}, ...]
      instances: {inst_id: {kind, def|inline, gui_kwargs, members}}
    layout: <opaque>   # optional

Until the recursive expander lands (grouping runtime), `flat_view` HARD-FAILS on
a document that actually carries definitions/instances, so a real sub-patch file
can never silently load as a partial flat graph.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

CURRENT_VERSION = 2


def normalize_loaded(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return `raw` as a current-version (v2) document."""
    version = raw.get("version")
    if version is None:
        return _v1_to_v2(raw)
    if version == CURRENT_VERSION:
        return raw
    raise ValueError(f"unsupported patch version {version!r}")


def _v1_to_v2(raw: Dict[str, Any]) -> Dict[str, Any]:
    """A flat v1 patch is a v2 document whose root has no sub-patches."""
    doc: Dict[str, Any] = {
        "version": CURRENT_VERSION,
        "definitions": {},
        "root": {"nodes": raw.get("nodes", {}), "links": raw.get("links", []), "instances": {}},
    }
    if raw.get("layout") is not None:
        doc["layout"] = raw["layout"]
    return doc


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


def flat_view(doc: Dict[str, Any]) -> Tuple[Dict, List, Dict, Dict, Any]:
    """Project a v2 document's root graph: (nodes, links, instances, definitions, layout).

    HARD-FAILS until the recursive expander exists: a document carrying real
    definitions or instances must not load as a partial flat graph.
    """
    defs = doc.get("definitions") or {}
    root = doc.get("root") or {}
    instances = root.get("instances") or {}
    if defs or instances:
        raise ValueError(
            "sub-patch definitions/instances require recursive expansion "
            "(not enabled in this build)"
        )
    return root.get("nodes", {}), root.get("links", []), instances, defs, doc.get("layout")


def read_graph(doc: Dict[str, Any]) -> Tuple[Dict, List, Dict, Dict, Any]:
    """Read a v2 document into (root_nodes, root_links, instances, definitions,
    layout). Unlike `flat_view`, this does NOT reject sub-patch content — the
    caller (Manager._expand_doc) knows how to splice instances into the flat
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


# ---------------------------------------------------------------------------
# One-shot v1 -> v2 converter (CLI)
# ---------------------------------------------------------------------------


def _migrate_file(path) -> bool:
    """Rewrite a single `.gfi` to v2 in place. Returns True if it changed.

    Idempotent: a file already at v2 is left untouched.
    """
    import yaml
    from pathlib import Path

    path = Path(path)
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    if not isinstance(raw, dict) or raw.get("version") == CURRENT_VERSION:
        return False
    doc = normalize_loaded(raw)
    path.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    return True


def main(argv=None) -> None:
    import argparse
    import glob as _glob
    from pathlib import Path

    parser = argparse.ArgumentParser(description="goofi .gfi format tools")
    parser.add_argument("command", choices=["migrate"], help="migrate: rewrite flat .gfi files to v2")
    parser.add_argument(
        "globs",
        nargs="*",
        default=["examples/*.gfi"],
        help="file globs to migrate (default: examples/*.gfi)",
    )
    args = parser.parse_args(argv)

    paths = sorted({p for g in args.globs for p in _glob.glob(g)})
    if not paths:
        print("no files matched")
        return

    changed = 0
    for p in paths:
        try:
            if _migrate_file(Path(p)):
                changed += 1
                print(f"migrated {p}")
            else:
                print(f"skipped  {p} (already v2)")
        except Exception as e:  # one bad file must not abort the batch
            print(f"FAILED   {p}: {e}")
    print(f"done — {changed} migrated, {len(paths) - changed} unchanged")


if __name__ == "__main__":
    main()
