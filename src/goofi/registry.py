"""AST-based node catalog — the single source of truth for node types.

The manager NEVER imports node modules. Discovery parses each file under
``goofi/nodes/`` with :mod:`ast`, finds every ``class X(Node)``, and compiles
ONLY the three config hooks (plus any same-file top-level constants they
need) into a whitelisted namespace. Executing a hook therefore runs the real
declaration code with real Python semantics, without touching the module's
imports — the import cost lands in the node's own process at spawn time.

Purity contract (enforced here + by tests/test_registry_parity.py):
config hooks may reference only builtins, goofi declaration symbols
(goofi.params, DataType, InputSlot, OutputSlot), and same-file top-level
constants that are themselves static. ``__import__`` is blocked inside the
namespace, so a hook that reaches runtime state (device enumeration) either
takes its own fallback path (a ``dynamic`` spec — the live node reports the
real options later) or is excluded from the catalog with an error naming the
node, hook, and symbol. Hooks must be declared in the node class's OWN body:
an inherited hook is invisible to the AST and falls back to ``{}``.
"""

import ast
import builtins
import importlib
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import goofi.params as _params_mod
from goofi.data import DataType
from goofi.node_helpers import InputSlot, OutputSlot, normalize_config

HOOK_NAMES = ("config_input_slots", "config_output_slots", "config_params")


def _blocked_import(name, *args, **kwargs):
    raise ImportError(
        f"config hooks must not import ('{name}') — declare a static fallback "
        "and report live options from the node process (see goofi/registry.py)"
    )


def _whitelist_namespace() -> dict:
    ns = {
        "DataType": DataType,
        "InputSlot": InputSlot,
        "OutputSlot": OutputSlot,
        "__builtins__": {**vars(builtins), "__import__": _blocked_import},
    }
    for name in dir(_params_mod):
        if not name.startswith("_"):
            ns[name] = getattr(_params_mod, name)
    return ns


@dataclass(frozen=True)
class NodeSpec:
    type: str
    category: str
    module: str
    cls_name: str
    doc: str
    available: bool = True
    dynamic: bool = False
    missing_deps: Tuple[str, ...] = ()
    hooks: Tuple[Callable, Callable, Callable] = field(default=None, repr=False, compare=False)

    def configure(self):
        """Fresh (input_slots, output_slots, NodeParams) — mirrors Node._configure."""
        in_hook, out_hook, params_hook = self.hooks
        return normalize_config(in_hook(), out_hook(), params_hook())

    def load_class(self) -> type:
        """Import the implementation. Only local-mode / group hosts / tests call this —
        never the manager's spawn path."""
        return getattr(importlib.import_module(self.module), self.cls_name)

    @classmethod
    def from_class(cls, node_cls: type) -> "NodeSpec":
        """Wrap an already-imported class (tests, Node.create). Uniform shape:
        the hooks ARE the class's hooks, so configure() is exactly _configure()."""
        return cls(
            type=node_cls.__name__,
            category=node_cls.category(),
            module=node_cls.__module__,
            cls_name=node_cls.__name__,
            doc=node_cls.docstring(),
            hooks=(node_cls.config_input_slots, node_cls.config_output_slots, node_cls.config_params),
        )


def _compile_function(fd: ast.FunctionDef, filename: str, ns: dict) -> Callable:
    """Compile a single function def (decorators stripped) into `ns` and return it."""
    clean = ast.FunctionDef(
        name=fd.name,
        args=fd.args,
        body=fd.body,
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    ast.copy_location(clean, fd)
    mod = ast.Module(body=[clean], type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, filename, "exec"), ns)
    return ns[fd.name]


def _exec_static_constants(tree: ast.Module, filename: str, ns: dict) -> None:
    """Execute top-level assignments that evaluate under the whitelist, in file
    order. Assignments touching non-whitelisted names are skipped — a hook that
    needs one will raise NameError, which is the contract signal."""
    for stmt in tree.body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            mod = ast.Module(body=[stmt], type_ignores=[])
            ast.fix_missing_locations(mod)
            try:
                exec(compile(mod, filename, "exec"), ns)
            except Exception:
                continue


def _hook_is_dynamic(fd: ast.FunctionDef, ns: dict) -> bool:
    """True when the hook reaches outside the whitelist: any import statement in
    its body, or a loaded name bound neither locally, in the namespace, nor in
    builtins. Such a hook must carry its own fallback path to stay evaluable."""
    bound = set()
    for node in ast.walk(fd):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return True
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.Lambda)):
            for a in getattr(node.args, "args", []):
                bound.add(a.arg)
    for node in ast.walk(fd):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in bound or node.id in ns or hasattr(builtins, node.id):
                continue
            return True
    return False


def _module_deps(tree: ast.Module) -> List[str]:
    """Top-level package names imported unconditionally at module scope.
    Imports inside try/except (optional deps) or function bodies don't count."""
    deps = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            deps.extend(alias.name.split(".")[0] for alias in stmt.names)
        elif isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and stmt.module:
            deps.append(stmt.module.split(".")[0])
    return [d for d in dict.fromkeys(deps) if d != "goofi"]


def _probe_available(deps: List[str]) -> Tuple[bool, Tuple[str, ...]]:
    missing = []
    for dep in deps:
        try:
            if importlib.util.find_spec(dep) is None:
                missing.append(dep)
        except (ImportError, ValueError):
            missing.append(dep)
    return (not missing, tuple(missing))


def build_catalog(
    nodes_root: Optional[Path] = None,
    package: str = "goofi.nodes",
) -> Tuple[Dict[str, NodeSpec], List[str]]:
    """Walk `nodes_root`, AST-extract every Node subclass, and return
    (catalog keyed by type name, human-readable errors). Nothing is imported."""
    if nodes_root is None:
        import goofi.nodes

        nodes_root = Path(goofi.nodes.__file__).parent

    catalog: Dict[str, NodeSpec] = {}
    origin: Dict[str, Path] = {}
    errors: List[str] = []

    for path in sorted(nodes_root.rglob("*.py")):
        rel = path.relative_to(nodes_root)
        if any(part.startswith("_") for part in rel.parts):
            continue
        module = f"{package}." + ".".join(rel.with_suffix("").parts)
        category = rel.parts[0] if len(rel.parts) > 1 else ""
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as e:
            errors.append(f"{rel}: syntax error: {e}")
            continue

        available, missing = _probe_available(_module_deps(tree))

        for cls_node in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            if not any(isinstance(b, ast.Name) and b.id == "Node" for b in cls_node.bases):
                continue
            ns = _whitelist_namespace()
            _exec_static_constants(tree, str(path), ns)

            hooks: Dict[str, Callable] = {}
            dynamic = False
            for item in cls_node.body:
                if isinstance(item, ast.FunctionDef) and item.name in HOOK_NAMES:
                    dynamic = dynamic or _hook_is_dynamic(item, ns)
                    hooks[item.name] = _compile_function(item, str(path), ns)

            # Validate the contract now, naming the failing hook: it must
            # evaluate under the whitelist (its own fallback path counts).
            # A violating node is excluded from the catalog, not fatal.
            failure = None
            for hook_name in HOOK_NAMES:
                if hook_name in hooks:
                    try:
                        hooks[hook_name]()
                    except Exception as e:
                        failure = (hook_name, e)
                        break
            if failure is not None:
                hook_name, exc = failure
                errors.append(
                    f"{rel}: {cls_node.name}.{hook_name} is not statically evaluable "
                    f"({type(exc).__name__}: {exc}) — see the purity contract in goofi/registry.py"
                )
                continue

            spec = NodeSpec(
                type=cls_node.name,
                category=category,
                module=module,
                cls_name=cls_node.name,
                doc=ast.get_docstring(cls_node) or "",
                available=available,
                dynamic=dynamic,
                missing_deps=missing,
                hooks=(
                    hooks.get("config_input_slots", dict),
                    hooks.get("config_output_slots", dict),
                    hooks.get("config_params", dict),
                ),
            )
            try:
                spec.configure()  # normalization must hold too (slot dtypes, params shape)
            except Exception as e:
                errors.append(
                    f"{rel}: {cls_node.name} config does not normalize "
                    f"({type(e).__name__}: {e})"
                )
                continue

            if cls_node.name in catalog:
                errors.append(
                    f"duplicate node type '{cls_node.name}': {origin[cls_node.name]} and {rel} "
                    "(type names must be globally unique; first one wins)"
                )
                continue
            catalog[cls_node.name] = spec
            origin[cls_node.name] = rel

    return catalog, errors
