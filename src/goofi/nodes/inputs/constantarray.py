import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.nodes.inputs._overwrite import _OverwriteHold
from goofi.params import FloatParam, StringParam


class ConstantArray(_OverwriteHold, Node):

    def config_params():
        return {
            "constant": {
                "value": FloatParam(1.0, -10.0, 10.0),
                "shape": "1",
                "graph": StringParam("none", options=["none", "ring", "random"]),
                "overwrite_timeout": FloatParam(
                    5,
                    0,
                    30,
                    doc="Duration within which the overwrite input data is used, revert to constant data after (0 never clears the overwrite).",
                ),
            },
            "common": {"autotrigger": True, "max_frequency": 30},
        }

    def config_input_slots():
        return {"overwrite": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def setup(self):
        self.setup_overwrite()

    def process(self, overwrite: Data):
        held = self.held_override(overwrite)
        if held is not None:
            return {"out": (held.data, held.meta)}

        if self.params.constant.graph.value == "ring":
            matrix = ring_graph_adjacency_matrix(int(self.params.constant.shape.value))
            return {"out": (matrix, {"sfreq": self.params.common.max_frequency.value})}

        if self.params.constant.graph.value == "random":
            return {
                "out": (
                    np.random.rand(int(self.params.constant.shape.value), int(self.params.constant.shape.value)),
                    {"sfreq": self.params.common.max_frequency.value},
                )
            }

        parts = [p for p in self.params.constant.shape.value.split(",") if len(p) > 0]
        shape = list(map(int, parts))
        return {
            "out": (
                np.ones(shape) * self.params.constant.value.value,
                {"sfreq": self.params.common.max_frequency.value},
            )
        }


def ring_graph_adjacency_matrix(n):
    # Create an nxn zero matrix
    adjacency = np.zeros((n, n), dtype=int)

    # Set values for the ring connections
    for i in range(n):
        adjacency[i][(i + 1) % n] = 1  # Next vertex in the ring
        adjacency[i][(i - 1) % n] = 1  # Previous vertex in the ring

    return adjacency
