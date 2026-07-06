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
