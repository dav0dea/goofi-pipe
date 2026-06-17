"""The bridge control ops added for Phase-1 save/load (no live socket)."""
import asyncio
import types
from pathlib import Path

from goofi.bridge.control import ControlHub


def _hub(manager=None) -> ControlHub:
    # ControlHub only needs `server.manager` for these ops; no real server.
    return ControlHub(types.SimpleNamespace(manager=manager))


def test_list_dir_op(tmp_path: Path):
    (tmp_path / "x.gfi").write_text("x")
    hub = _hub()
    res = asyncio.run(hub._dispatch("list_dir", {"path": str(tmp_path)}))
    assert res["path"] == str(tmp_path.resolve())
    assert any(e["name"] == "x.gfi" for e in res["entries"])


def test_list_examples_op():
    hub = _hub()
    res = asyncio.run(hub._dispatch("list_examples", {}))
    assert "entries" in res


def test_serialize_op_calls_manager():
    class FakeMgr:
        def serialize_patch(self):
            return "nodes: {}\nlinks: []\n"

    hub = _hub(FakeMgr())
    res = asyncio.run(hub._dispatch("serialize", {}))
    assert res["yaml"].startswith("nodes:")
