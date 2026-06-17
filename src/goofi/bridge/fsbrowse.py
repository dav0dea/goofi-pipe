"""Filesystem-browse helpers for the bridge control plane.

Pure functions: turn a path into a JSON-safe directory listing the browser
renders as a file picker. NO jail — goofi-pipe is a trusted single-user local
app and the user explicitly wants full-filesystem access (see the persistence
design spec, decision-log item 4). Device auth is future work.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def examples_dir() -> Optional[Path]:
    """The repo `examples/` dir, or None under a wheel that doesn't ship it."""
    cand = Path(__file__).resolve().parents[3] / "examples"
    return cand if cand.is_dir() else None


def _entry(child: Path) -> Optional[Dict[str, Any]]:
    try:
        st = child.stat()
        is_dir = child.is_dir()
    except OSError:
        return None
    return {
        "name": child.name,
        "path": str(child),
        "kind": "dir" if is_dir else "file",
        "is_gfi": child.suffix == ".gfi",
        "hidden": child.name.startswith("."),
        "size": st.st_size,
        "mtime": st.st_mtime,
    }


def _roots() -> List[Dict[str, str]]:
    roots = [{"label": "Home", "path": str(Path.home())}]
    ex = examples_dir()
    if ex is not None:
        roots.append({"label": "Examples", "path": str(ex)})
    roots.append({"label": "Working dir", "path": str(Path.cwd())})
    return roots


def list_dir(path: Optional[str]) -> Dict[str, Any]:
    """List one directory level. `path` None → home. A file path → its parent."""
    base = Path(path).expanduser() if path else Path.home()
    base = base.resolve()
    if base.is_file():
        base = base.parent

    entries: List[Dict[str, Any]] = []
    try:
        children = list(base.iterdir())
    except OSError:
        children = []
    # Dirs before files, each case-insensitively name-sorted.
    children.sort(key=lambda p: (p.is_file(), p.name.lower()))
    for child in children:
        e = _entry(child)
        if e is not None:
            entries.append(e)

    parent = base.parent
    return {
        "path": str(base),
        "parent": str(parent) if parent != base else None,
        "entries": entries,
        "roots": _roots(),
    }


def list_examples() -> Dict[str, Any]:
    ex = examples_dir()
    if ex is None:
        return {"entries": []}
    entries: List[Dict[str, Any]] = []
    for child in sorted(ex.glob("*.gfi"), key=lambda p: p.name.lower()):
        e = _entry(child)
        if e is not None:
            entries.append(e)
    return {"entries": entries}
