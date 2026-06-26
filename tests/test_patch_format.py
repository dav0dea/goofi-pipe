"""Unit tests for the versioned .gfi patch-format boundary (uid-keyed v2)."""
import pytest

from goofi.patch_format import build_v2, normalize_loaded, read_graph


def test_normalize_v2_is_identity():
    doc = build_v2(nodes={}, links=[], layout=None, definitions={}, instances={})
    assert normalize_loaded(doc) is doc


def test_normalize_rejects_legacy_v1():
    # The flat/name-keyed v1 format (no `version`) is no longer supported.
    with pytest.raises(ValueError):
        normalize_loaded({"nodes": {}, "links": []})


def test_normalize_rejects_unknown_version():
    with pytest.raises(ValueError):
        normalize_loaded({"version": 99, "root": {}})


def test_read_graph_returns_root_pieces():
    doc = build_v2(
        nodes={"u0": {"name": "osc0", "_type": "Oscillator", "category": "inputs"}},
        links=[{"node_out": "u0", "node_in": "u1", "slot_out": "out", "slot_in": "in"}],
        layout={"x": 1},
        definitions={},
        instances={},
    )
    nodes, links, instances, defs, layout = read_graph(doc)
    assert "u0" in nodes and nodes["u0"]["name"] == "osc0"
    assert len(links) == 1
    assert instances == {} and defs == {}
    assert layout == {"x": 1}


def test_read_graph_passes_subpatch_content_through():
    # Unlike the removed flat_view, read_graph hands sub-patch content to the
    # caller (Manager._expand_doc) rather than rejecting it.
    doc = build_v2(nodes={}, links=[], layout=None, definitions={"d": {}}, instances={"i": {}})
    nodes, links, instances, defs, layout = read_graph(doc)
    assert defs == {"d": {}} and instances == {"i": {}}


def test_build_v2_omits_layout_when_none():
    doc = build_v2(nodes={}, links=[], layout=None, definitions={}, instances={})
    assert "layout" not in doc
    assert list(doc.keys())[0] == "version"  # version first for stable yaml
