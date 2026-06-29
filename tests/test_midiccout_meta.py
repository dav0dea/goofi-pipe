"""MidiCCout must build its status meta from a CC input that is actually present.

The node has five independent triggering CC inputs (cc1..cc5). The send loop
correctly guards each input with ``data is not None``, but both return
statements built the status tuple from ``cc1.meta`` unconditionally. Wiring only
cc2 (cc1 unwired -> None) made ``cc1.meta`` raise AttributeError on every tick,
*after* the CC messages had already been sent, so the node reported a processing
error continuously and never emitted a status. These tests pin the status meta
to a present input's meta (or {} when none is present).
"""
import numpy as np

from goofi.data import Data, DataType
from goofi.node import NodeEnv
from goofi.nodes.outputs.midiccout import MidiCCout


def _standalone(cls):
    i, o, p = cls._configure()
    return cls(None, i, o, p, NodeEnv.STANDALONE)


class _FakeOutport:
    """Records sends so the test exercises the meta path without a MIDI device."""

    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)

    def close(self):
        pass


class _FakeMido:
    def __init__(self):
        self.outport = _FakeOutport()

    def open_output(self, name):
        return self.outport

    def Message(self, *args, **kwargs):
        return (args, kwargs)


def _midiccout():
    # STANDALONE skips auto-setup(), so the real mido backend is never opened.
    # Stub self.mido so the send loop reaches the return without a MIDI device.
    node = _standalone(MidiCCout)
    node.mido = _FakeMido()
    return node


def test_cc1_unwired_cc2_present_does_not_raise_and_uses_cc2_meta():
    # The reproducer: cc1 unwired (None), only cc2 present. The buggy returns
    # dereferenced cc1.meta -> AttributeError on None every tick.
    node = _midiccout()
    cc2 = Data(DataType.ARRAY, np.array([64.0]), {"m": 1})
    out = node.process(cc1=None, cc2=cc2, cc3=None, cc4=None, cc5=None)
    value, meta = out["midi_status"]
    assert value == "CC messages sent successfully"
    # Data normalizes meta (adds channels/dtype/shape), so compare against the
    # present input's actual meta rather than the literal passed in.
    assert meta == cc2.meta and meta["m"] == 1, "status must carry a present cc input's meta"
    # the present input's value was actually sent
    assert len(node.mido.outport.sent) == 1


def test_no_cc_inputs_present_falls_back_to_empty_meta():
    # When every cc input is None there is no meta to forward; the status must
    # still be emitted with an empty meta rather than raising.
    node = _midiccout()
    out = node.process(cc1=None, cc2=None, cc3=None, cc4=None, cc5=None)
    value, meta = out["midi_status"]
    assert value == "CC messages sent successfully"
    assert meta == {}
