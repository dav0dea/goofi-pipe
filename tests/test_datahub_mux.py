"""Unit tests for the data-plane per-slot multiplexer (Option C relay).

The mux owns ONE NodeRef view subscription per (uid, slot). The node has already
reduced + encoded the frame, so `dispatch` fans the node-reduced bytes to every
connected forwarder VERBATIM — no manager-side decode/adapt/re-encode. The mux
also folds the connected forwarders' ViewSpecs and pushes the fold to the node.
"""
import asyncio
import types

from goofi.bridge.data import DataHub, _SlotMux


class _FakeFwd:
    def __init__(self, spec=None):
        self.spec = spec or {"axes": [], "version": 0}
        self.frames = []

    def push_threadsafe(self, buf: bytes) -> None:
        self.frames.append(buf)


class _FakeRef:
    def __init__(self):
        self.specs = []

    def set_viewspec(self, slot, spec):
        self.specs.append((slot, spec))


class _RecordingRef:
    """A NodeRef stand-in that records view-plane (re)registration + viewspec pushes."""

    def __init__(self, uid):
        self.uid = uid
        self.handlers = []  # (slot, fn, raw, view)
        self.specs = []

    def set_data_handler(self, slot, fn, raw=False, view=False):
        self.handlers.append((slot, fn, raw, view))

    def set_viewspec(self, slot, spec):
        self.specs.append((slot, spec))


def _hub_with_nodes(nodes):
    server = types.SimpleNamespace(manager=types.SimpleNamespace(nodes=nodes))
    return DataHub(server)


def test_rewire_node_repoints_mux_and_reregisters_handler_on_the_new_ref():
    """After restart_node replaces a node's NodeRef, its mux still holds the OLD (dead)
    ref and the view handler is bound to it — so the new node publishes into nothing and
    viewers freeze. rewire_node must re-point the mux at the new ref, re-register the
    view-plane handler on it, and re-push the folded viewspec so frames flow again."""
    old = _RecordingRef("n1")
    new = _RecordingRef("n1")  # same uid, fresh ref after restart
    hub = _hub_with_nodes({"n1": new})

    mux = _SlotMux(ref=old, slot="out")
    fwd = _FakeFwd({"axes": [{"axis": -1, "max": 800, "method": "envelope"}], "version": 1})
    mux.add(fwd)
    hub._muxes[("n1", "out")] = mux

    asyncio.run(hub.rewire_node("n1"))

    assert mux.ref is new  # re-pointed at the live ref
    assert any(slot == "out" and raw and view for (slot, _fn, raw, view) in new.handlers), (
        "view-plane handler not re-registered on the new ref"
    )
    assert any(slot == "out" for (slot, _spec) in new.specs), "folded viewspec not re-pushed to the new node"

    # Frames published by the restarted node now reach the still-connected viewer.
    on_frame = next(fn for (slot, fn, _r, _v) in new.handlers if slot == "out")
    on_frame(new, "out", b"FRESH")
    assert fwd.frames == [b"FRESH"]


def test_detach_unregisters_the_current_ref_not_a_stale_capture():
    """When the last viewer of a slot disconnects, the unregister must target the CURRENT
    ref (mux.ref) — which rewire_node may have re-pointed at the restarted node — not the
    ref captured when the connection opened. Using the stale capture leaves the new ref's
    view handler + reducer orphaned (the round-1 restart fix's regression)."""
    old = _RecordingRef("n1")  # the ref captured at connect time (now dead after a restart)
    new = _RecordingRef("n1")  # the live ref rewire_node re-pointed the mux at
    hub = _hub_with_nodes({"n1": new})
    mux = _SlotMux(ref=new, slot="out")
    fwd = _FakeFwd()
    mux.add(fwd)
    hub._muxes[("n1", "out")] = mux

    asyncio.run(hub._detach(("n1", "out"), mux, fwd))

    # The last viewer left → unregister the handler on the CURRENT ref and drop the mux.
    assert any(slot == "out" and fn is None for (slot, fn, _r, _v) in new.handlers), (
        "did not unregister the current (mux.ref) handler"
    )
    assert old.handlers == [], "touched the stale captured ref instead of mux.ref"
    assert ("n1", "out") not in hub._muxes


def test_dispatch_fans_raw_bytes_verbatim():
    mux = _SlotMux(ref=_FakeRef(), slot="out")
    a, b = _FakeFwd(), _FakeFwd()
    mux.add(a)
    mux.add(b)

    mux.dispatch(b"REDUCED")

    assert a.frames == [b"REDUCED"]
    assert b.frames == [b"REDUCED"]
    assert a.frames[0] is b.frames[0]  # same bytes fanned out, never re-encoded


def test_remove_keeps_others_and_reports_empty():
    mux = _SlotMux(ref=_FakeRef(), slot="out")
    a, b = _FakeFwd(), _FakeFwd()
    mux.add(a)
    mux.add(b)
    assert mux.remove(a) is False  # b still connected
    mux.dispatch(b"x")
    assert a.frames == []  # removed → no frames
    assert b.frames == [b"x"]
    assert mux.remove(b) is True  # last one out → empty


def test_push_spec_if_changed_folds_richest_and_dedups():
    ref = _FakeRef()
    mux = _SlotMux(ref=ref, slot="out")
    a = _FakeFwd({"axes": [{"axis": -1, "max": 800, "method": "envelope"}], "version": 1})
    b = _FakeFwd({"axes": [{"axis": -1, "max": 2000, "method": "envelope"}], "version": 2})
    mux.add(a)
    mux.add(b)

    mux.push_spec_if_changed()
    assert len(ref.specs) == 1
    slot, folded = ref.specs[0]
    assert slot == "out"
    assert folded["axes"] == [{"axis": -1, "max": 2000, "method": "envelope"}]  # richest

    mux.push_spec_if_changed()  # unchanged fold → no extra push
    assert len(ref.specs) == 1

    mux.remove(b)  # drop the richer viewer → fold narrows → push
    mux.push_spec_if_changed()
    assert len(ref.specs) == 2
    assert ref.specs[-1][1]["axes"][0]["max"] == 800


def test_connect_path_registration_failure_cleans_up_without_leaking():
    """The first-viewer registration (set_data_handler view=True) does blocking IPC that
    can fault (iceoryx2 subscriber build, ctrl publish). If it raises, the handler must
    undo any partial REGISTER_VIEWER (so the node doesn't reduce into a dead subscriber)
    and tear down the started forwarder + ws — not let the exception escape past the
    try/finally and leak the parked sender task + a stranded viewer registration."""
    import asyncio
    from types import SimpleNamespace

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from goofi.bridge.data import DataHub

    class _FailingRef:
        uid = "n1"
        output_slots = {"out": "ARRAY"}

        def __init__(self):
            self.unregister_calls = []

        def set_data_handler(self, slot, fn, raw=False, view=False):
            if view and fn is not None:
                raise RuntimeError("iceoryx2 .view subscriber build failed")
            self.unregister_calls.append(slot)  # the cleanup unregister (fn=None)

        def set_viewspec(self, slot, spec):
            pass

    async def run():
        ref = _FailingRef()
        manager = SimpleNamespace(nodes={"n1": ref}, _instances={})
        server = SimpleNamespace(host="127.0.0.1", manager=manager)
        hub = DataHub(server)
        app = web.Application()
        app.add_routes([web.get("/data/{node}/{slot}/{kind}", hub.handler)])
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/data/n1/out/line")
            await ws.receive()  # server closes the ws after the failed registration
            await ws.close()
            await asyncio.sleep(0.05)
            # No mux stored, and the partial registration was undone (orphan-viewer guard).
            assert hub._muxes == {}, "a mux was stored despite a failed registration"
            assert ref.unregister_calls == ["out"], "partial REGISTER_VIEWER not cleaned up"

    asyncio.new_event_loop().run_until_complete(run())


def test_push_spec_dedups_on_axes_ignoring_version():
    """The node ignores ViewSpec.version, so a version-only bump (identical axes)
    must NOT trigger a redundant set_viewspec ctrl publish."""
    ref = _FakeRef()
    mux = _SlotMux(ref=ref, slot="out")
    a = _FakeFwd({"axes": [{"axis": -1, "max": 800, "method": "envelope"}], "version": 1})
    mux.add(a)
    mux.push_spec_if_changed()
    assert len(ref.specs) == 1

    a.spec = {"axes": [{"axis": -1, "max": 800, "method": "envelope"}], "version": 7}
    mux.push_spec_if_changed()  # only version changed → axes identical → no push
    assert len(ref.specs) == 1
