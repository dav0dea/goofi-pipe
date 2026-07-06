"""The purity contract's teeth: AST-derived metadata must equal the real
cls._configure() output for every node in the library. Dynamic specs (device
enumeration) are compared structurally — names/types, not options/defaults."""

import pytest

from goofi.registry import build_catalog

CATALOG, ERRORS = build_catalog()


def test_catalog_builds_clean():
    assert ERRORS == []


def test_catalog_covers_the_library():
    # 145 modules today, >= that many classes; guards against a walk regression.
    assert len(CATALOG) >= 145


def load_or_skip(spec):
    """Import the spec's class, or skip: a missing dep (probe) or a
    broken-but-installed package (import raises) means the node simply
    doesn't instantiate in this environment — by design, the child-process
    bootstrap surfaces that on the node's error channel."""
    if not spec.available:
        pytest.skip(f"missing deps: {', '.join(spec.missing_deps)}")
    try:
        return spec.load_class()
    except Exception as e:
        pytest.skip(f"implementation not importable in this environment: {type(e).__name__}: {e}")


@pytest.mark.parametrize("spec", CATALOG.values(), ids=lambda s: s.type)
def test_ast_matches_import(spec):
    cls = load_or_skip(spec)
    assert cls.__name__ == spec.cls_name
    assert cls.category() == spec.category
    assert cls.docstring() == spec.doc

    ast_in, ast_out, ast_params = spec.configure()
    real_in, real_out, real_params = cls._configure()

    assert {n: s.dtype for n, s in ast_in.items()} == {n: s.dtype for n, s in real_in.items()}
    assert {n: s.trigger_process for n, s in ast_in.items()} == {
        n: s.trigger_process for n, s in real_in.items()
    }
    assert {n: s.dtype for n, s in ast_out.items()} == {n: s.dtype for n, s in real_out.items()}

    ast_ser, real_ser = ast_params.serialize(), real_params.serialize()
    if spec.dynamic:
        # device lists differ per machine — structure only
        assert set(ast_ser) == set(real_ser)
        for group in ast_ser:
            assert set(ast_ser[group]) == set(real_ser[group])
    else:
        assert ast_ser == real_ser


def test_the_dynamic_set_is_exactly_the_device_nodes():
    dynamic = sorted(s.type for s in CATALOG.values() if s.dynamic)
    assert dynamic == ["AudioOut", "AudioStream", "MidiCCout", "MidiOut"]
