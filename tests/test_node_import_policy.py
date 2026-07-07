"""The no-import-guards policy for the node library.

A node's module-level imports are plain and unconditional: a missing dependency
greys the palette entry (the availability probe reads the AST imports), and a
broken-but-installed dependency fails the child-process bootstrap and surfaces
on the node's error channel. A module-level ``try/except`` around an import both
hides the dep from the probe (so the palette lies) and can chain into a second,
misleading failure. Config hooks may still ``try/except`` a device-enumeration
import INSIDE their own body — that is the intended ``dynamic`` pattern.
"""
import ast
import pathlib

import goofi.nodes


def test_no_module_level_try_except_imports():
    root = pathlib.Path(goofi.nodes.__file__).parent
    offenders = []
    for p in sorted(root.rglob("*.py")):
        if any(part.startswith("_") for part in p.relative_to(root).parts):
            continue
        tree = ast.parse(p.read_bytes())
        for stmt in tree.body:  # module scope only — hook bodies are exempt
            if isinstance(stmt, ast.Try) and any(
                isinstance(sub, (ast.Import, ast.ImportFrom)) for sub in ast.walk(stmt)
            ):
                offenders.append(str(p.relative_to(root)))
                break
    assert offenders == [], f"module-level try/except imports (no-guards policy): {offenders}"
