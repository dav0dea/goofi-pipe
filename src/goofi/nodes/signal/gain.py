import numpy as np

from goofi.data import DataType
from goofi.node import InputSlot, Node
from goofi.params import FloatParam


class Gain(Node):
    """
    This node scales an incoming signal by a constant gain factor. It is a
    length-preserving mid-chain processor: its single input consumes losslessly
    (a queue slot, so no frames are dropped), and because the output frame count
    matches the input, the framework copies the input's continuity index straight
    through to the output.

    Inputs:
    - signal: Array to scale.

    Outputs:
    - out: The input array multiplied by the gain, carrying the input's sfreq.
    """

    def config_input_slots():
        return {"signal": InputSlot(DataType.ARRAY, queue=True)}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"gain": {"gain": FloatParam(1.0, 0.0, 10.0)}}

    def process(self, signal):
        if signal is None:
            return None

        out = np.asarray(signal.data) * self.params.gain.gain.value

        meta = {}
        if "sfreq" in signal.meta:
            meta["sfreq"] = signal.meta["sfreq"]

        return {"out": (out, meta)}
