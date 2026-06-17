"""Unit tests for the versioned .gfi patch-format boundary."""
import pytest

from goofi.patch_format import build_v2, flat_view, normalize_loaded


def test_normalize_v1_wraps_to_v2():
    raw = {"nodes": {"osc0": {"_type": "Oscillator", "category": "inputs"}}, "links": []}
    doc = normalize_loaded(raw)
    assert doc["version"] == 2
    assert doc["definitions"] == {}
    assert doc["root"]["nodes"] == raw["nodes"]
    assert doc["root"]["links"] == []


def test_normalize_v1_preserves_layout():
    raw = {"nodes": {}, "links": [], "layout": {"workspaces": []}}
    doc = normalize_loaded(raw)
    assert doc["layout"] == {"workspaces": []}


def test_normalize_v2_is_identity():
    doc = build_v2(nodes={}, links=[], layout=None, definitions={}, instances={})
    assert normalize_loaded(doc) is doc


def test_normalize_rejects_unknown_version():
    with pytest.raises(ValueError):
        normalize_loaded({"version": 99, "root": {}})


def test_flat_view_returns_root_pieces():
    doc = build_v2(
        nodes={"osc0": {"_type": "Oscillator", "category": "inputs"}},
        links=[{"node_out": "osc0", "node_in": "b", "slot_out": "out", "slot_in": "in"}],
        layout={"x": 1},
        definitions={},
        instances={},
    )
    nodes, links, instances, defs, layout = flat_view(doc)
    assert "osc0" in nodes
    assert len(links) == 1
    assert instances == {} and defs == {}
    assert layout == {"x": 1}


def test_flat_view_hard_fails_on_subpatch_content():
    doc = build_v2(nodes={}, links=[], layout=None, definitions={"d": {}}, instances={})
    with pytest.raises(ValueError):
        flat_view(doc)
    doc2 = build_v2(nodes={}, links=[], layout=None, definitions={}, instances={"i": {}})
    with pytest.raises(ValueError):
        flat_view(doc2)


def test_build_v2_omits_layout_when_none():
    doc = build_v2(nodes={}, links=[], layout=None, definitions={}, instances={})
    assert "layout" not in doc
    assert list(doc.keys())[0] == "version"  # version first for stable yaml


def test_migrate_file_is_idempotent(tmp_path):
    import yaml as _yaml

    from goofi.patch_format import _migrate_file

    p = tmp_path / "x.gfi"
    p.write_text(
        _yaml.dump({"nodes": {"a": {"_type": "T", "category": "c"}}, "links": []}, sort_keys=False)
    )
    assert _migrate_file(p) is True
    first = p.read_text()
    assert "version: 2" in first
    # second run is a no-op (already v2) and leaves the file byte-identical
    assert _migrate_file(p) is False
    assert p.read_text() == first
