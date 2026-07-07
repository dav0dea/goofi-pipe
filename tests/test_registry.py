"""Registry: AST-based node catalog. No node module is imported by the manager."""

from pathlib import Path

from goofi.registry import NodeSpec, build_catalog


def make_nodes_tree(tmp_path: Path, files: dict) -> Path:
    """Write {relpath: source} under tmp_path/nodes and return that root."""
    root = tmp_path / "nodes"
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
    return root


BASIC = '''
import numpy as np
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam

class Osc(Node):
    """Makes waves."""
    def config_output_slots():
        return {"out": DataType.ARRAY}
    def config_params():
        return {"osc": {"frequency": FloatParam(1.0, 0.01, 100.0)}}
    def process(self):
        return {"out": (np.zeros(1), {})}
'''


def test_basic_spec(tmp_path):
    root = make_nodes_tree(tmp_path, {"inputs/osc.py": BASIC})
    catalog, errors = build_catalog(root, package="fixture.nodes")
    assert errors == []
    spec = catalog["Osc"]
    assert spec.category == "inputs"
    assert spec.module == "fixture.nodes.inputs.osc"
    assert spec.cls_name == "Osc"
    assert spec.doc == "Makes waves."
    assert spec.dynamic is False
    in_slots, out_slots, params = spec.configure()
    assert in_slots == {}
    assert set(out_slots) == {"out"}
    assert params["osc"]["frequency"].value == 1.0


def test_configure_returns_fresh_objects(tmp_path):
    root = make_nodes_tree(tmp_path, {"inputs/osc.py": BASIC})
    catalog, _ = build_catalog(root, package="fixture.nodes")
    _, _, p1 = catalog["Osc"].configure()
    p1["osc"]["frequency"].value = 42.0
    _, _, p2 = catalog["Osc"].configure()
    assert p2["osc"]["frequency"].value == 1.0


MULTI = '''
from goofi.data import DataType
from goofi.node import Node

class A(Node):
    def config_output_slots():
        return {"a": DataType.ARRAY}
    def process(self):
        pass

class B(Node):
    def config_input_slots():
        return {"b": DataType.STRING}
    def process(self):
        pass
'''


def test_multiple_nodes_per_file(tmp_path):
    root = make_nodes_tree(tmp_path, {"misc/multi.py": MULTI})
    catalog, errors = build_catalog(root, package="fixture.nodes")
    assert errors == []
    assert set(catalog) == {"A", "B"}
    assert catalog["A"].module == catalog["B"].module == "fixture.nodes.misc.multi"


def test_duplicate_type_name_is_error(tmp_path):
    root = make_nodes_tree(
        tmp_path,
        {"misc/one.py": MULTI, "signal/two.py": MULTI.replace("class B", "class C")},
    )
    catalog, errors = build_catalog(root, package="fixture.nodes")
    # first-seen wins; the collision is reported naming both files
    assert "A" in catalog
    assert any("A" in e and "one.py" in e and "two.py" in e for e in errors)


def test_underscore_modules_skipped(tmp_path):
    root = make_nodes_tree(tmp_path, {"misc/_helper.py": BASIC, "misc/osc.py": BASIC})
    catalog, _ = build_catalog(root, package="fixture.nodes")
    assert catalog["Osc"].module == "fixture.nodes.misc.osc"


CONSTANTS = '''
from goofi.node import Node
from goofi.params import StringParam

METRICS = ["euclidean", "manhattan"]
DEFAULTS = {"metric": METRICS[0]}

class Reducer(Node):
    def config_params():
        return {"r": {"metric": StringParam(DEFAULTS["metric"], options=METRICS)}}
    def process(self):
        pass
'''


def test_same_file_constants_resolve(tmp_path):
    root = make_nodes_tree(tmp_path, {"analysis/red.py": CONSTANTS})
    catalog, errors = build_catalog(root, package="fixture.nodes")
    assert errors == []
    _, _, params = catalog["Reducer"].configure()
    assert params["r"]["metric"].options == ["euclidean", "manhattan"]
    assert catalog["Reducer"].dynamic is False


HELPER_FN = '''
from goofi.node import Node
from goofi.params import IntParam

def _shared_params():
    return {"g": {"size": IntParam(8, 1, 64)}}

class First(Node):
    def config_params():
        return _shared_params()
    def process(self):
        pass

class Second(Node):
    def config_params():
        return _shared_params()
    def process(self):
        pass
'''


def test_same_file_helper_function_shared_by_hooks(tmp_path):
    root = make_nodes_tree(tmp_path, {"signal/pair.py": HELPER_FN})
    catalog, errors = build_catalog(root, package="fixture.nodes")
    assert errors == []
    _, _, p1 = catalog["First"].configure()
    _, _, p2 = catalog["Second"].configure()
    assert p1["g"]["size"].value == p2["g"]["size"].value == 8
    # fresh Param objects per configure() — no shared mutable state
    assert p1["g"]["size"] is not p2["g"]["size"]
    assert catalog["First"].dynamic is False


DYNAMIC = '''
from goofi.node import Node
from goofi.params import StringParam

class MidiThing(Node):
    def config_params():
        try:
            import mido
            ports = mido.get_output_names()
        except Exception:
            ports = ["fallback"]
        return {"m": {"port": StringParam(ports[0], options=ports)}}
    def process(self):
        pass
'''


def test_dynamic_hook_with_fallback(tmp_path):
    root = make_nodes_tree(tmp_path, {"outputs/midithing.py": DYNAMIC})
    catalog, errors = build_catalog(root, package="fixture.nodes")
    assert errors == []
    spec = catalog["MidiThing"]
    assert spec.dynamic is True  # references non-whitelisted `mido`
    _, _, params = spec.configure()  # fallback path evaluates statically
    assert params["m"]["port"].options == ["fallback"]


VIOLATION = '''
import numpy as np
from goofi.node import Node

class Bad(Node):
    def config_params():
        return {"b": {"n": int(np.pi)}}  # heavy dep, no fallback
    def process(self):
        pass
'''


def test_contract_violation_excluded_with_error(tmp_path):
    root = make_nodes_tree(tmp_path, {"misc/bad.py": VIOLATION})
    catalog, errors = build_catalog(root, package="fixture.nodes")
    assert "Bad" not in catalog
    assert any("Bad" in e and "config_params" in e and "np" in e for e in errors)


AVAIL = '''
import numpy as np
import definitely_not_installed_pkg
from goofi.node import Node

try:
    import also_not_installed_but_optional
except ImportError:
    also_not_installed_but_optional = None

class Needy(Node):
    def config_params():
        return {}
    def process(self):
        pass
'''


def test_availability_probe(tmp_path):
    root = make_nodes_tree(tmp_path, {"misc/needy.py": AVAIL})
    catalog, _ = build_catalog(root, package="fixture.nodes")
    spec = catalog["Needy"]
    assert spec.available is False
    assert "definitely_not_installed_pkg" in spec.missing_deps
    # try/except-wrapped imports are optional — never availability-blocking
    assert "also_not_installed_but_optional" not in spec.missing_deps


NO_MP = '''
from goofi.node import Node

class Fussy(Node):
    NO_MULTIPROCESSING = True
    def config_params():
        return {}
    def process(self):
        pass
'''


def test_no_multiprocessing_flag_extracted(tmp_path):
    root = make_nodes_tree(tmp_path, {"misc/fussy.py": NO_MP, "misc/osc.py": BASIC})
    catalog, _ = build_catalog(root, package="fixture.nodes")
    assert catalog["Fussy"].no_multiprocessing is True
    assert catalog["Osc"].no_multiprocessing is False


NON_LITERAL_NO_MP = '''
from goofi.node import Node

class BadFlag(Node):
    NO_MULTIPROCESSING = bool(1)  # not a plain literal -> ast.literal_eval raises
    def config_params():
        return {}
    def process(self):
        pass
'''


def test_non_literal_no_multiprocessing_excluded_not_fatal(tmp_path):
    # A node whose NO_MULTIPROCESSING is a non-literal expression must be
    # EXCLUDED with a named error — never crash the whole catalog build (and
    # with it manager startup). Sibling nodes still land.
    root = make_nodes_tree(tmp_path, {"misc/badflag.py": NON_LITERAL_NO_MP, "misc/osc.py": BASIC})
    catalog, errors = build_catalog(root, package="fixture.nodes")
    assert "BadFlag" not in catalog
    assert "Osc" in catalog
    assert any("BadFlag" in e and "NO_MULTIPROCESSING" in e for e in errors)


UTF8_DOC = '''
from goofi.node import Node

class Cafe(Node):
    """Réduit le café — accented docstring with a typographic dash."""
    def config_params():
        return {}
    def process(self):
        pass
'''


def test_build_catalog_reads_utf8_regardless_of_locale(tmp_path, monkeypatch):
    # Node sources are UTF-8 by the import system's contract (PEP 263). The
    # catalog must decode them as UTF-8, not via the locale codec — else a
    # legacy locale crashes startup on the first multibyte docstring. We prove
    # the read never goes through the locale-dependent Path.read_text().
    root = make_nodes_tree(tmp_path, {"misc/cafe.py": UTF8_DOC})

    def boom(self, *a, **k):  # any non-UTF-8 locale would raise here
        raise UnicodeDecodeError("ascii", b"", 0, 1, "locale codec cannot decode node source")

    monkeypatch.setattr(Path, "read_text", boom)
    catalog, errors = build_catalog(root, package="fixture.nodes")
    assert errors == []
    assert "Cafe" in catalog
    assert "café" in catalog["Cafe"].doc


def test_availability_probe_namespace_package_submodule(tmp_path, monkeypatch):
    # A PEP 420 namespace package (a bare dir on sys.path, no __init__.py) makes
    # find_spec(top-level) succeed even though the real dependency is a SUBMODULE
    # that isn't installed — the google/google.generativeai shape. The probe must
    # see through it and report the node unavailable, naming the dotted gap.
    import importlib

    site = tmp_path / "site"
    (site / "acme_ns").mkdir(parents=True)  # namespace portion, no __init__.py
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()

    src = '''
import acme_ns.generativeai as gen
from goofi.node import Node

class NS(Node):
    def config_params():
        return {}
    def process(self):
        pass
'''
    root = make_nodes_tree(tmp_path, {"misc/ns.py": src})
    catalog, _ = build_catalog(root, package="fixture.nodes")
    spec = catalog["NS"]
    assert spec.available is False
    assert "acme_ns.generativeai" in spec.missing_deps


SHADOW = '''
from goofi.node import Node
from goofi.params import StringParam

# A module-level constant that shadows a registry-whitelist name. Under a real
# import the file's binding wins; the AST catalog must agree, or it silently
# feeds the palette the whitelist object instead of the node's own list.
FloatParam = ["red", "green", "blue"]

class Painter(Node):
    def config_params():
        return {"p": {"color": StringParam(FloatParam[0], options=FloatParam)}}
    def process(self):
        pass
'''


def test_same_file_declaration_shadows_whitelist(tmp_path):
    root = make_nodes_tree(tmp_path, {"misc/painter.py": SHADOW})
    catalog, errors = build_catalog(root, package="fixture.nodes")
    assert errors == []
    _, _, params = catalog["Painter"].configure()
    assert params["p"]["color"].options == ["red", "green", "blue"]
    assert params["p"]["color"].value == "red"


BAD_CALLBACKS = '''
from goofi.node import Node
from goofi.params import IntParam, StringParam

class Cbs(Node):
    def config_params():
        return {"g": {"size": IntParam(4, 1, 8), "mode": StringParam("a", options=["a", "b"])}}
    def process(self):
        pass
    def g_size_changed(self, value):        # OK: matches g.size, takes (self, value)
        pass
    def g_nonexistent_changed(self, value): # dead: no g.nonexistent param
        pass
    def g_mode_changed(self):               # wrong arity: matches g.mode but no value arg
        pass
    def common_autotrigger_changed(self, value):  # OK: the injected common param
        pass
'''


def test_param_changed_callbacks_validated(tmp_path):
    # A `{group}_{name}_changed` method that maps to no real param is silently
    # never dispatched; one that omits the value arg TypeErrors on every change.
    # Both are surfaced as (non-fatal) errors — the node still configures.
    root = make_nodes_tree(tmp_path, {"misc/cbs.py": BAD_CALLBACKS})
    catalog, errors = build_catalog(root, package="fixture.nodes")
    assert "Cbs" in catalog  # a dead callback doesn't cripple the node
    assert any("g_nonexistent_changed" in e for e in errors), "dead callback name must be reported"
    assert any("g_mode_changed" in e for e in errors), "wrong-arity callback must be reported"
    # the correct ones are never flagged
    assert not any("g_size_changed" in e for e in errors)
    assert not any("common_autotrigger_changed" in e for e in errors)


def test_from_class_wraps_a_real_class():
    from goofi.nodes.inputs.constantarray import ConstantArray

    spec = NodeSpec.from_class(ConstantArray)
    assert spec.type == "ConstantArray"
    assert spec.category == "inputs"
    assert spec.module == "goofi.nodes.inputs.constantarray"
    in_slots, out_slots, params = spec.configure()
    ri, ro, rp = ConstantArray._configure()
    assert {n: s.dtype for n, s in in_slots.items()} == {n: s.dtype for n, s in ri.items()}
    assert {n: s.dtype for n, s in out_slots.items()} == {n: s.dtype for n, s in ro.items()}
    assert params.serialize() == rp.serialize()


def test_load_class_real_library():
    catalog, _ = build_catalog()
    cls = catalog["ConstantArray"].load_class()
    assert cls.__name__ == "ConstantArray"


def test_describe_node_spec_payload():
    from goofi.bridge.schemas import describe_node_spec

    catalog, _ = build_catalog()
    d = describe_node_spec(catalog["ConstantArray"])
    assert d["type"] == "ConstantArray"
    assert d["category"] == "inputs"
    assert d["available"] is True
    assert d["dynamic"] is False
    assert d["missing_deps"] == []
    assert len(d["output_slots"]) >= 1
    assert isinstance(d["params"], dict)
