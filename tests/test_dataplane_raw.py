"""Raw vs decoded data-handler contract on the NodeRef data plane (report B4).

`set_data_handler(raw=True)` hands the pump's GOOF wire bytes straight to the
callback (no `decode_data`); `raw=False` decodes once and hands the callback a
`Data`. Both are supported NodeRef capabilities. The bridge data plane uses
`raw=False` — it decodes each slot once and the viewer adapters re-encode per
kind (see test_dataplane_perf); the `raw=True` path remains for any consumer
that wants the producer bytes untouched.
"""
import time

from goofi.codec import decode_data
from goofi.data import Data, DataType

from .test_manager import _bare_manager


def _wait_until(pred, timeout: float) -> None:
    deadline = time.time() + timeout
    while not pred() and time.time() < deadline:
        time.sleep(0.02)


def test_raw_handler_delivers_wire_bytes():
    mgr = _bare_manager()
    try:
        osc = mgr.add_node("Oscillator", "inputs")
        ref = mgr.nodes[osc]
        ref.wait_for_state(timeout=2.0)

        frames: list = []
        ref.set_data_handler("out", lambda _nr, _slot, buf: frames.append(buf), raw=True)
        _wait_until(lambda: len(frames) >= 2, 3.0)
        ref.set_data_handler("out", None)

        assert frames, "raw handler received no frames"
        assert all(isinstance(f, (bytes, bytearray)) for f in frames)
        # The raw bytes are a valid GOOF frame that decodes to the array output.
        decoded = decode_data(frames[-1])
        assert decoded.dtype == DataType.ARRAY
    finally:
        mgr.terminate(notify_gui=False)


def test_decoded_handler_still_delivers_data():
    mgr = _bare_manager()
    try:
        osc = mgr.add_node("Oscillator", "inputs")
        ref = mgr.nodes[osc]
        ref.wait_for_state(timeout=2.0)

        frames: list = []
        ref.set_data_handler("out", lambda _nr, _slot, data: frames.append(data))
        _wait_until(lambda: len(frames) >= 2, 3.0)
        ref.set_data_handler("out", None)

        assert frames, "decoded handler received no frames"
        assert all(isinstance(x, Data) for x in frames)
        assert frames[-1].dtype == DataType.ARRAY
    finally:
        mgr.terminate(notify_gui=False)


def test_unregister_closes_subscriber_and_listener():
    """report B4: unregistering a data handler must release BOTH the iceoryx2
    subscriber and its listener — previously only the subscriber was closed,
    leaking a listener per subscribe/unsubscribe cycle."""
    mgr = _bare_manager()
    try:
        osc = mgr.add_node("Oscillator", "inputs")
        ref = mgr.nodes[osc]
        ref.wait_for_state(timeout=2.0)

        ref.set_data_handler("out", lambda *_a: None, raw=True)
        sub, listener = ref._data_handlers["out"][0], ref._data_handlers["out"][1]
        ref.set_data_handler("out", None)

        # close() nulls the inner handle on both IPC endpoints.
        assert sub._sub is None, "subscriber not closed on unregister"
        assert listener._listener is None, "listener leaked on unregister"
    finally:
        mgr.terminate(notify_gui=False)
