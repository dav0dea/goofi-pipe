import numpy as np

from goofi.data import DataType
from goofi.node import Node
from goofi.params import FloatParam, StringParam


class ConstantArray(Node):
    """
    This node generates an array output whose contents and structure depend on the selected mode. It can produce a constant-valued array of any specified shape, a ring graph adjacency matrix, or a random matrix. The node can be used to create standard data arrays or generate simple graph structures for further processing in the pipeline.

    Outputs:
    - out: An array (NumPy ndarray). The array can be a constant array of given shape filled with a specified value, a ring graph adjacency matrix, or a random matrix, depending on configuration. The output includes metadata with the sample frequency if available.
    """

    def config_params():
        return {
            "constant": {
                "value": FloatParam(1.0, -10.0, 10.0),
                "shape": "1",
                "graph": StringParam("none", options=["none", "ring", "random"]),
            },
            "common": {"autotrigger": True},
        }

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def process(self):
        if self.params.constant.graph.value == "ring":
            matrix = ring_graph_adjacency_matrix(int(self.params.constant.shape.value))
            return {"out": (matrix, {"sfreq": self.params.common.max_frequency.value})}
        elif self.params.constant.graph.value == "random":
            return {
                "out": (
                    np.random.rand(int(self.params.constant.shape.value), int(self.params.constant.shape.value)),
                    {"sfreq": self.params.common.max_frequency.value},
                )
            }
        else:
            parts = [p for p in self.params.constant.shape.value.split(",") if len(p) > 0]
            shape = list(map(int, parts))
            return {
                "out": (np.ones(shape) * self.params.constant.value.value, {"sfreq": self.params.common.max_frequency.value})
            }


def ring_graph_adjacency_matrix(n):
    # Create an nxn zero matrix
    adjacency = np.zeros((n, n), dtype=int)

    # Set values for the ring connections
    for i in range(n):
        adjacency[i][(i + 1) % n] = 1  # Next vertex in the ring
        adjacency[i][(i - 1) % n] = 1  # Previous vertex in the ring

    return adjacency
