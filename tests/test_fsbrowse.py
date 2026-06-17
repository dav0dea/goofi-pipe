"""Unit tests for the bridge filesystem-browse helpers."""
from pathlib import Path

from goofi.bridge import fsbrowse


def test_list_dir_lists_entries_dirs_first(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.gfi").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    res = fsbrowse.list_dir(str(tmp_path))
    assert res["path"] == str(tmp_path.resolve())
    names = [e["name"] for e in res["entries"]]
    # dirs sort before files
    assert names[0] == "sub"
    assert set(names) == {"sub", "a.gfi", "b.txt"}
    gfi = next(e for e in res["entries"] if e["name"] == "a.gfi")
    assert gfi["kind"] == "file" and gfi["is_gfi"] is True
    assert next(e for e in res["entries"] if e["name"] == "sub")["kind"] == "dir"
    assert "roots" in res and any(r["label"] == "Home" for r in res["roots"])


def test_list_dir_on_a_file_returns_its_parent(tmp_path: Path):
    f = tmp_path / "p.gfi"
    f.write_text("x")
    res = fsbrowse.list_dir(str(f))
    assert res["path"] == str(tmp_path.resolve())


def test_list_dir_parent_is_none_at_root():
    res = fsbrowse.list_dir("/")
    assert res["parent"] is None


def test_list_dir_none_defaults_to_home():
    res = fsbrowse.list_dir(None)
    assert res["path"] == str(Path.home().resolve())


def test_list_dir_flags_hidden(tmp_path: Path):
    (tmp_path / ".secret").write_text("x")
    res = fsbrowse.list_dir(str(tmp_path))
    assert next(e for e in res["entries"] if e["name"] == ".secret")["hidden"] is True


def test_list_examples_returns_gfi_entries():
    res = fsbrowse.list_examples()
    # In a checkout the examples dir exists and holds .gfi files.
    if fsbrowse.examples_dir() is not None:
        assert all(e["is_gfi"] for e in res["entries"])
        assert len(res["entries"]) > 0
    else:
        assert res["entries"] == []
