"""Stitcher — the past of its input, carried by `goofi.Stream`, so a one-sample frame can still
look four samples back."""

import numpy as np
import goofi

BACK = 4


class Stitcher(goofi.Node):
    """Emit, per sample, how far the stream moved over the last four samples."""

    TAGS = ["transform"]
    INPUTS = {"input": goofi.DataType.ARRAY}
    OUTPUTS = {"out": goofi.DataType.ARRAY}
    PARAMS = {
        "stitcher": {
            "history": goofi.IntParam(16, 0, 10000, doc="Steps of the past to stitch against."),
            "reset": goofi.PulseParam(doc="Forget the past."),
        }
    }

    def setup(self):
        self.stream = goofi.Stream()

    def pulse_stitcher_reset(self):
        self.stream.reset()

    def process(self, input):
        if input is None:
            return
        stitched, at = self.stream.push(input.data, -1, self.params.stitcher.history)
        head = np.repeat(stitched[..., :1], BACK, axis=-1)
        past = np.concatenate([head, stitched], axis=-1)
        n = input.data.shape[-1]
        return stitched[..., at:] - past[..., at : at + n], input.meta
