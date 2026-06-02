"""Per-node stdout/stderr capture + SSE log server (`goofi.node_log`).

These exercise the capture/attribution and the SSE endpoint directly (no
iceoryx2, no Manager) so they're fast and deterministic. The headline case is
**two nodes hosted in one process** (the process-group shape): each must get
only its own lines, proving per-node attribution inside a shared process.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request

import pytest

from goofi import node_log


@pytest.fixture(autouse=True)
def _reset_capture():
    yield
    node_log._reset_for_tests()


def _run_as_node(node_id: str, fn) -> None:
    """Run `fn` on a thread bound to `node_id` (mimics a node worker thread)."""
    t = threading.Thread(target=lambda: (node_log.bind_thread_node(node_id), fn()))
    t.start()
    t.join()


def _texts(node_id: str):
    return [(r["stream"], r["text"]) for r in list(node_log._buffers[node_id].records)]


def _read_sse(url: str, n: int, timeout: float = 3.0):
    """Read up to `n` SSE `data:` records from `url`."""
    out = []
    req = urllib.request.urlopen(url, timeout=timeout)
    try:
        while len(out) < n:
            line = req.readline()
            if not line:
                break
            if line.startswith(b"data:"):
                out.append(json.loads(line[5:].decode().strip()))
    finally:
        req.close()
    return out


def test_per_thread_attribution():
    node_log.register_node("A")
    node_log.register_node("B")
    _run_as_node("A", lambda: print("from A"))
    _run_as_node("B", lambda: (print("from B"), print("err B", file=sys.stderr)))

    assert ("stdout", "from A") in _texts("A")
    assert ("stdout", "from A") not in _texts("B")
    assert ("stdout", "from B") in _texts("B")
    assert ("stderr", "err B") in _texts("B")


def test_unbound_thread_not_captured():
    """A thread with no bound node and no process-default (group/LOCAL shape)
    falls through to the real stream — the manager's own output is never
    attributed to a node."""
    node_log.register_node("A")
    print("manager-ish line")
    time.sleep(0.05)
    assert all("manager-ish" not in text for _, text in _texts("A"))


def test_process_default_attributes_unbound(monkeypatch):
    """Single-node multiprocessing: the process default catches threads the node
    spawns itself (which aren't explicitly bound)."""
    node_log.register_node("solo")
    node_log.set_process_default_node("solo")
    done = threading.Event()
    threading.Thread(target=lambda: (print("spawned-by-node"), done.set())).start()
    done.wait(1.0)
    assert ("stdout", "spawned-by-node") in _texts("solo")


def test_sse_serves_only_that_node():
    """Two nodes in ONE process — each endpoint serves only its own lines."""
    ep_a = node_log.register_node("A")
    ep_b = node_log.register_node("B")
    _run_as_node("A", lambda: print("alpha"))
    _run_as_node("B", lambda: print("beta"))

    a = _read_sse(ep_a, 1)
    b = _read_sse(ep_b, 1)
    assert a and all(d["node"] == "A" for d in a)
    assert any(d["text"] == "alpha" for d in a)
    assert b and all(d["node"] == "B" for d in b)
    assert any(d["text"] == "beta" for d in b)


def test_sse_resumes_after_last_event_id():
    """Reconnect with Last-Event-ID replays only what was missed."""
    ep = node_log.register_node("A")
    _run_as_node("A", lambda: (print("one"), print("two"), print("three")))
    req = urllib.request.Request(ep, headers={"Last-Event-ID": "0"})
    resp = urllib.request.urlopen(req, timeout=3.0)
    try:
        seqs = []
        while len(seqs) < 2:
            line = resp.readline()
            if not line:
                break
            if line.startswith(b"data:"):
                seqs.append(json.loads(line[5:].decode().strip())["seq"])
    finally:
        resp.close()
    # seq 0 ("one") was acknowledged; resume starts at 1.
    assert seqs[0] >= 1
