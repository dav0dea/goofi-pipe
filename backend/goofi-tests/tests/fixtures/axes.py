"""Axes — one meta rule per pass over a labelled, rated frame, so the rules the pymod owns can be
read off the wire."""

import numpy as np
import goofi


class Axes(goofi.Node):
    """Apply one of the meta rules to a `[2, 3]` labelled frame."""

    TAGS = ["transform"]
    OUTPUTS = {"out": goofi.DataType.ARRAY}
    PRODUCER = True
    PARAMS = {
        "axes": {
            "rule": goofi.StringParam(
                "drop_last",
                options=["drop_last", "drop_first", "keep", "insert_first", "insert_last", "concat"],
                doc="Which meta rule to apply to the frame.",
            )
        }
    }

    def process(self):
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        src = goofi.Data(
            arr,
            {"sfreq": 250.0, "channels": {"dim0": ["a", "b"], "dim1": [1.0, 2.0, 3.0]}},
        )
        rule = self.params.axes.rule
        if rule == "drop_last":
            return arr.mean(axis=-1), src.drop_axis(-1)
        if rule == "drop_first":
            return arr.mean(axis=0), src.drop_axis(0)
        if rule == "keep":
            return arr[:, [0, 2]], src.keep(-1, [0, 2])
        if rule == "insert_first":
            return arr[None, ...], src.insert_axis(0, ["only"])
        if rule == "insert_last":
            return arr[..., None], src.insert_axis(2)
        return np.concatenate([arr, arr], axis=-1), src.concat([src], -1)
