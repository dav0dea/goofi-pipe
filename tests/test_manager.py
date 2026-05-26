"""Tests for the Manager after the iceoryx2 refactor."""
import time
import uuid
from os import path

import pytest

import goofi
from goofi.manager import Manager, NodeContainer
from goofi.transport import (
    WaitSet,
    data_service_name,
    open_subscriber,
    set_instance_id,
)

MANAGER_TEST_DURATION = 0.2


def _bare_manager(use_multiprocessing: bool = True) -> Manager:
    """Construct a Manager without entering its blocking event loop."""
    import os, atexit
    from goofi.manager import _cleanup_iceoryx2_shm
    from goofi.node_helpers import NodeProcessRegistry, list_nodes

    mgr = Manager.__new__(Manager)
    instance_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    set_instance_id(instance_id)
    atexit.register(_cleanup_iceoryx2_shm, instance_id)
    list_nodes(verbose=False)
    mgr._instance_id = instance_id
    mgr._headless = True
    mgr._use_multiprocessing = use_multiprocessing
    mgr._running = True
    mgr.nodes = NodeContainer()
    mgr._node_groups = {}
    mgr._links = []
    NodeProcessRegistry().headless = True
    mgr._save_path = None
    mgr._unsaved_changes = False
    mgr._bridge = None
    return mgr


def _build_simple_graph(mgr: Manager) -> tuple[str, str]:
    """Oscillator → Select with a 0:5 include filter."""
    osc = mgr.add_node("Oscillator", "inputs")
    sel = mgr.add_node("Select", "array", params={"select": {"include": "0:5"}})
    mgr.add_link(osc, sel, "out", "data")
    return osc, sel


def test_creation_smoke():
    """Manager(duration=...) ought to start, run briefly, and shut down cleanly."""
    Manager(duration=MANAGER_TEST_DURATION)


def test_main_entrypoint():
    """The CLI entry point should accept --headless and exit cleanly."""
    goofi.manager.main(MANAGER_TEST_DURATION, ["--headless"])


def test_simple_chain_dataflow():
    """End-to-end: data passes through a 2-node chain via iceoryx2."""
    mgr = _bare_manager()
    try:
        osc, sel = _build_simple_graph(mgr)

        # External subscriber acting like a GUI viewer.
        service = data_service_name(sel, "out")
        sub, listener = open_subscriber(service, in_process=False, latest_wins=True)
        ws = WaitSet()
        ws.attach(listener)
        mgr.nodes[sel].register_subscriber("out")

        from goofi.codec import decode_data

        received = []
        deadline = time.time() + 3.0
        while time.time() < deadline and len(received) < 5:
            if ws.wait(0.25):
                buf = sub.take_latest()
                if buf is not None:
                    received.append(decode_data(buf))
        assert len(received) >= 1, "no data reached the downstream subscriber"
        # The Select node was configured with include=0:5, so the latest
        # frame must have at most 5 elements along axis 0.
        assert received[-1].data.shape[0] == 5
    finally:
        mgr.terminate(notify_gui=False)


def test_save_empty(tmpdir):
    mgr = _bare_manager()
    try:
        mgr.save(path.join(str(tmpdir), "test.gfi"))
        assert path.exists(path.join(str(tmpdir), "test.gfi"))
    finally:
        mgr.terminate(notify_gui=False)


def test_save_extension(tmpdir):
    mgr = _bare_manager()
    try:
        _build_simple_graph(mgr)
        mgr.save(path.join(str(tmpdir), "no_extension"))
        assert path.exists(path.join(str(tmpdir), "no_extension.gfi"))
    finally:
        mgr.terminate(notify_gui=False)


@pytest.mark.parametrize("overwrite", [True, False])
def test_save_overwrite(overwrite, tmpdir):
    mgr = _bare_manager()
    tmpdir = str(tmpdir)
    try:
        _build_simple_graph(mgr)
        target = path.join(tmpdir, "test.gfi")
        mgr.save(target, overwrite=overwrite)
        assert path.exists(target)
        if overwrite:
            mgr.save(target, overwrite=True)
            assert path.exists(target)
        else:
            with pytest.raises(FileExistsError):
                mgr.save(target)
    finally:
        mgr.terminate(notify_gui=False)


def test_save_to_directory(tmpdir):
    mgr = _bare_manager()
    tmpdir = str(tmpdir)
    try:
        _build_simple_graph(mgr)
        with pytest.raises(ValueError):
            mgr.save(123)  # not a str
        mgr.save(tmpdir)
        assert path.exists(path.join(tmpdir, "untitled0.gfi"))
        mgr.save(tmpdir)
        assert path.exists(path.join(tmpdir, "untitled1.gfi"))
    finally:
        mgr.terminate(notify_gui=False)
